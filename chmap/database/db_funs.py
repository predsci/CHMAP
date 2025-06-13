"""
Create functions for initializing connection to Database,
adding entries, querying, etc.
"""

import os
import sys
import socket
import urllib.parse
import pandas as pd
import numpy as np
import astropy.time as astro_time
import collections  # for checking if an input is iterable
import sunpy.map
import datetime
import getpass
from scipy import interpolate

from sqlalchemy import create_engine, func, or_, union_all, case, distinct
from sqlalchemy.orm import sessionmaker, aliased
from chmap.database.db_classes import *
from chmap.data.download.euv_utils import get_metadata
from chmap.utilities.file_io import io_helpers
from chmap.data.download import euv_utils
from chmap.utilities.datatypes import datatypes
import chmap.utilities.file_io.psi_hdf as psihdf
import chmap.utilities.cred_funs as creds


def init_db_conn(db_type, chd_base, db_loc, db_name='chd', port="3306", user="",
                 password="", cred_dir=None):
    """
    Connect to the database specified by db_name.
    Then establish the SQLAlchemy declarative base
    and engine.
    db_type - should be 'sqlite' or 'mysql'.
    chd_base - sqlalchemy declarative base that
    defines SQL table structures.
    db_loc - character string. sqlite: a path to the local DB file
                               mysql: an IP address or host name
    db_name - for mysql, designate which database to open
    port - for mysql, allows user to designate a different port
           where appropriate.
    :return: an SQLAlchemy session
    """

    # first define the engine connection
    if db_type == 'mysql':
        db_user = user
        if password == "":
            db_psswd = creds.recover_pwd(cred_dir)
        else:
            db_psswd = password
        url_psswd = urllib.parse.quote_plus(db_psswd)
        connect_string = 'mysql://' + db_user + ':' + url_psswd + '@' + db_loc + \
                         ':' + port + '/' + db_name
        print_string = 'mysql://' + db_user + ':****pwd****@' + db_loc + ':' + \
                       port + '/' + db_name
        print("Attempting to connect to DB server ", print_string)  # print to check
    elif db_type == 'sqlite':
        print("Attempting to connect to SQLite DB server " + db_loc)
        connect_string = 'sqlite:///' + db_loc
    else:
        sys.exit("At this time, 'db_type' must be either 'sqlite' or 'mysql'.")

    # now establish the engine
    engine = create_engine(connect_string)
    # import base declarative table definitions into the engine. Also instructs the
    # engine to create tables if they don't exist.
    chd_base.metadata.create_all(engine)
    # define the session
    session = sessionmaker(bind=engine)
    # open session/connection to DB
    db = session()
    print("Connection successful\n")
    return db


def init_db_conn_old(db_name, chd_base, sqlite_path="", user="", password="",
                     cred_dir=None):
    """
    Connect to the database specified by db_name.
    Then establish the SQLAlchemy declarative base
    and engine.
    db_name - should be 'sqlite' or 'mysql-shadow'.
    chd_base - sqlalchemy declarative base that
    defines SQL table structures.
    sqlite_fn - local sqlite filename. File should/will
    be located in App.DATABASE_HOME.
    :return: an SQLAlchemy session
    """

    # first define the engine connection
    if db_name == 'sqlite':
        print("Attempting to connect to SQLite DB server " + sqlite_path)
        connect_string = 'sqlite:///' + sqlite_path
    elif db_name == 'chd-db':
        database_dir = App.DATABASE_HOME
        sqlite_filename = App.DATABASE_FNAME
        sqlite_path = os.path.join(database_dir, sqlite_filename)
        print('Attempting to connect to CHD_DB file ' + sqlite_path)
        connect_string = 'sqlite:///' + sqlite_path
    elif db_name == 'mysql-Q':
        db_user = user
        if password == "":
            db_psswd = creds.recover_pwd(cred_dir)
        else:
            db_psswd = password
        url_psswd = urllib.parse.quote_plus(db_psswd)
        # connect_string = 'mysql://' + db_user + ':' + url_psswd + '@q.predsci.com:3306/chd'
        # print_string = 'mysql://' + db_user + ':****pwd****@q.predsci.com:3306/chd'
        if socket.gethostname() == "Q":
            hostname = "Q"
        else:
            hostname = "q.predsci.com"
        connect_string = 'mysql://' + db_user + ':' + url_psswd + '@' + hostname + ':3306/chd'
        print_string = 'mysql://' + db_user + ':****pwd****@' + hostname + ':3306/chd'
        print("Attempting to connect to DB server ", print_string)  # print to check
    elif db_name == 'mysql-Q_test':
        db_user = user
        if password == "":
            db_psswd = creds.recover_pwd(cred_dir)
        else:
            db_psswd = password
        url_psswd = urllib.parse.quote_plus(db_psswd)
        connect_string = 'mysql://' + db_user + ':' + url_psswd + '@q.predsci.com:3306/chd_test'
        print_string = 'mysql://' + db_user + ':****pwd****@q.predsci.com:3306/chd_test'
        print("Attempting to connect to DB server ", print_string)  # print to check
    else:
        sys.exit("At this time, 'db_name' must be either 'sqlite', 'chd-db', 'mysql-Q', or 'mysql-shadow'.")

    # now establish the engine
    engine = create_engine(connect_string)
    # import base declarative table definitions into the engine. Also instructs the engine to create tables if they
    # don't exist.
    chd_base.metadata.create_all(engine)
    # define the session
    session = sessionmaker(bind=engine)
    # open session/connection to DB
    db = session()
    print("Connection successful\n")
    return db


def query_euv_images(db_session, time_min=None, time_max=None, instrument=None, wavelength=None, flag=0):
    """
    User-friendly query function.  Allows user to avoid using SQLAlchemy or SQL
    syntax.  Default behavior is to return the entire database.
    Update (02/2021) - function updated to return a join of the EUV_Images and
    Data_Files tables.

    time_min, time_max - datetime objects that define the time interval to search.
    instrument - list of spacecraft to include (characters)
    wavelength - list of wavelengths to include (integer)
    flag - integer (default 0)
    db_session - database sqlalchemy session object
    """

    base_query = db_session.query(EUV_Images, Data_Files.fname_raw, Data_Files.fname_hdf).filter(
            EUV_Images.data_id == Data_Files.data_id).order_by(
            EUV_Images.instrument, EUV_Images.date_obs)

    if time_min is None and time_max is None:
        # get entire DB
        query_out = base_query
    elif not isinstance(time_min, datetime.datetime) or not isinstance(time_max, datetime.datetime):
        sys.exit("Error: time_min and time_max must have matching entries of 'None' or of type Datetime.")
    elif wavelength is None:
        if instrument is None:
            query_out = base_query.filter(EUV_Images.date_obs >= time_min,
                                          EUV_Images.date_obs <= time_max, EUV_Images.flag == flag)
        else:
            query_out = base_query.filter(EUV_Images.date_obs >= time_min, EUV_Images.date_obs <= time_max,
                                          EUV_Images.instrument.in_(instrument), EUV_Images.flag == flag)
    else:
        if instrument is None:
            query_out = base_query.filter(EUV_Images.date_obs >= time_min, EUV_Images.date_obs <= time_max,
                                          EUV_Images.wavelength.in_(wavelength), EUV_Images.flag == flag)
        else:
            query_out = base_query.filter(EUV_Images.date_obs >= time_min, EUV_Images.date_obs <= time_max,
                                          EUV_Images.instrument.in_(instrument),
                                          EUV_Images.wavelength.in_(wavelength), EUV_Images.flag == flag)

    pd_out = pd.read_sql(query_out.statement, db_session.bind)

    return pd_out


def query_euv_images_rot(db_session, rot_min=None, rot_max=None, instrument=None, wavelength=None, flag=0):
    """
    function to query euv_images by carrington rotation
    @param db_session:
    @param rot_min:
    @param rot_max:
    @param instrument:
    @param wavelength:
    @param flag: integer (default 0)
    @return:
    """

    base_query = db_session.query(EUV_Images, Data_Files.fname_raw, Data_Files.fname_hdf).filter(
        EUV_Images.data_id == Data_Files.data_id, EUV_Images.flag == flag).order_by(
        EUV_Images.instrument, EUV_Images.cr_rot)

    if rot_min is None and rot_max is None:
        # get entire DB
        query_out = base_query
    elif wavelength is None:
        if instrument is None:
            query_out = base_query.filter(EUV_Images.cr_rot >= rot_min, EUV_Images.cr_rot <= rot_max)
        else:
            query_out = base_query.filter(EUV_Images.cr_rot >= rot_min, EUV_Images.cr_rot <= rot_max,
                                          EUV_Images.instrument.in_(instrument))
    else:
        if instrument is None:
            query_out = base_query.filter(EUV_Images.cr_rot >= rot_min, EUV_Images.cr_rot <= rot_max,
                                          EUV_Images.wavelength.in_(wavelength))
        else:
            query_out = base_query.filter(EUV_Images.cr_rot >= rot_min, EUV_Images.cr_rot <= rot_max,
                                          EUV_Images.instrument.in_(instrument),
                                          EUV_Images.wavelength.in_(wavelength))

    pd_out = pd.read_sql(query_out.statement, db_session.bind)

    return pd_out


def query_data_file(db_session, time_min=None, time_max=None, datatype=None, provider=None, flag=0):
    """
    Query the database for records in the data_files table.

    Default behavior is to return all database entries for flag=0. In general,
    any input that is left empty will simpy be ignored during the query.

    Parameters
    ----------
    db_session : sqlalchemy.orm.session.Session
        Active sqlalchemy session that includes database connection.
    time_min : datetime.datetime
        Query minimum time for 'date_obs' column
    time_max : datetime.datetime
        Query maximum time for 'date_obs' column
    datatype : list of strings
        File type to be queried (ex. 'EUV_Image')
    provider : list of strings
        Providers to be included (ex. 'LMSAL-EUVI_A', 'LMSAL-EUVI_B', 'LMSAL-AIA')
    flag: int
        Single flag value to be queried

    Returns
    -------
    pd_out : pandas.DataFrame
        Query results are returned in the same format as they appear in the 'data_files' table.
    """

    base_query = db_session.query(Data_Files).filter(
        Data_Files.flag == flag).order_by(
        Data_Files.type, Data_Files.provider, Data_Files.date_obs)

    if time_min is None and time_max is None:
        # get entire DB
        query_out = base_query
    elif not isinstance(time_min, datetime.datetime) or not isinstance(time_max, datetime.datetime):
        sys.exit("Error: time_min and time_max must have matching entries of 'None' or of type Datetime.")
    elif datatype is None:
        if provider is None:
            query_out = base_query.filter(Data_Files.date_obs >= time_min,
                                          Data_Files.date_obs <= time_max, Data_Files.flag == flag)
        else:
            query_out = base_query.filter(Data_Files.date_obs >= time_min, Data_Files.date_obs <= time_max,
                                          Data_Files.provider.in_(provider), Data_Files.flag == flag)
    else:
        if provider is None:
            query_out = base_query.filter(Data_Files.date_obs >= time_min, Data_Files.date_obs <= time_max,
                                          Data_Files.type.in_(datatype), Data_Files.flag == flag)
        else:
            query_out = base_query.filter(Data_Files.date_obs >= time_min, Data_Files.date_obs <= time_max,
                                          Data_Files.provider.in_(provider),
                                          Data_Files.type.in_(datatype), Data_Files.flag == flag)

    pd_out = pd.read_sql(query_out.statement, db_session.bind)

    return pd_out


def add_datafile2session(db_session, date_obs, data_provider, data_type, fname_raw,
                         fname_hdf="", flag=0):
    """

    :param db_session: sqlalchemy.orm.session.Session
                       SQLAlchemy session object
    :param date_obs: datetime.datetime
                     The observation timestamp
    :param data_provider: str
                          A brief label for the data provider
    :param data_type: str
                      A brief label for the data type
    :param fname_raw: str
                      The relative path of the raw data file
    :param fname_hdf: str (default: "")
                      The relative path of the processed data file (optional)
    :param flag: int (default: 0)
                 Simple flag for good data, bad data, saturated image, etc.
    :return:
            db_session: sqlalchemy.orm.session.Session
                        The updated session with new record added to session,
                        but not yet committed to the database.
            out_flag: int
                      Database write result: 0 - record written to session
                                             1 - record already exists in DB.
                                             Record not added to session.
                                             2 - filename_raw already exists in
                                             DB. Record not added to session.
    """
    # check if exists in DB
    out_flag = 0
    existing_record = db_session.query(Data_Files).filter(
        Data_Files.date_obs == date_obs,
        Data_Files.provider == data_provider,
        Data_Files.type == data_type
    ).all()
    if existing_record.__len__() > 0:
        print("A database record already exists for provider", data_provider,
              ", data-type", data_type, "and observation date", date_obs, ". SKIPPING.\n")
        out_flag = 1
        return db_session, out_flag
    existing_fname = db_session.query(Data_Files).filter(
        Data_Files.fname_raw == fname_raw
    ).all()
    if existing_fname.__len__() > 0:
        print("A database record already exists for relative path:", fname_raw, ". SKIPPING.\n")
        out_flag = 2
        return db_session, out_flag

    # save to database
    new_record = Data_Files(date_obs=date_obs, provider=data_provider, type=data_type,
                            fname_raw=fname_raw, fname_hdf=fname_hdf, flag=flag)
    db_session.add(new_record)

    return db_session, out_flag


def add_image2session(data_dir, subdir, fname, db_session, datatype="EUV_Image", provider=None):
    """
    Adds a row to the database session (euv_images table) that references the image location and metadata.
    The updated session will need to be committed - db_session.commit() - in order to
    write the new row to the DB.  See example101_DB-workflow.py
    Additionally, add an entry to the data_files table
    :param data_dir: The local location of data directory.  If using an SQLite DB,
    then it should be located in data_dir as well.
    :param subdir: File location relative to data_dir.
    :param fname: Image file name.
    :param db_session: The SQLAlchemy database session.
    :return: the updated SQLAlchemy session.
    """

    DB_path = os.path.join(subdir, fname)
    fits_path = os.path.join(data_dir, DB_path)
    fits_map = sunpy.map.Map(fits_path)
    file_meta = get_metadata(fits_map)

    # check if row already exists in DB
    existing_row_id = db_session.query(EUV_Images.data_id, Data_Files.fname_raw).filter(
        EUV_Images.instrument == file_meta['instrument'],
        EUV_Images.date_obs == euv_utils.roundSeconds(file_meta['datetime']),
        EUV_Images.wavelength == file_meta['wavelength'],
        EUV_Images.data_id == Data_Files.data_id).all()
    if len(existing_row_id) == 1:
        if DB_path != existing_row_id[0][1]:
            # this is a problem.  We now have two different files for the same image
            print(("Current download: " + DB_path + " already exists in the database under a different file name:" +
                   existing_row_id[0][1]))
            sys.exit()
        else:
            # file has already been downloaded and entered into DB. do nothing
            print("File is already logged in database.  Nothing added.")
            pass
    elif len(existing_row_id) > 1:
        # This image already exists in the DB in MORE THAN ONE PLACE!
        print("Current download: " + DB_path + " already exists in the database MULTIPLE times. " +
              "Something is fundamentally wrong. DB unique index should " +
              "prevent this from happening.")
        sys.exit()
    else:
        # Add new entry to data_files
        if provider is None:
            # automate provider name for back-compatibility
            inst_string = file_meta['instrument']
            inst_string = inst_string.replace("-", "_")
            provider = "LMSAL-" + inst_string
        data_file_add = Data_Files(date_obs=file_meta['datetime'], provider=provider,
                                   type=datatype, fname_raw=DB_path, fname_hdf="")
        db_session.add(data_file_add)
        # get data_id
        db_session.flush()
        new_data_id = data_file_add.data_id
        # Construct now DB table row
        image_add = EUV_Images(data_id=new_data_id, date_obs=file_meta['datetime'],
                               instrument=file_meta['instrument'], wavelength=file_meta['wavelength'],
                               distance=file_meta['distance'], cr_lon=file_meta['cr_lon'],
                               cr_lat=file_meta['cr_lat'], cr_rot=file_meta['cr_rot'],
                               time_of_download=datetime.datetime.now())
        # Append to the list of rows to be added
        db_session.add(image_add)

        print(("Database row added for " + file_meta['instrument'] + ", wavelength: " + str(file_meta['wavelength']) +
               ", timestamp: " + file_meta['date_string']))

    return db_session


def remove_euv_image(db_session, raw_series, raw_dir, hdf_dir, proc_only=False):
    """
    Simultaneously delete image from filesystem and remove metadata row
    from EUV_Images table.
    raw_series - expects a pandas series that results from one row of
    the EUV_Images DB table.
    proc_only - boolean {False} To delete only the proccessed image file,
                set to True.  This will leave an entry in the database,
                but set fname_hdf to "".
    Ex.
    test_pd = query_euv_images(db_session=db_session, time_min=query_time_min,
                                time_max=query_time_max)
    remove_euv_image(raw_series=test_pd.iloc[0])

    ToDo:
        - Vectorize functionality.  If 'raw_series' is a pandas DataFrame
            delete all rows and their corresponding files.
    """

    raw_id = raw_series['data_id']
    raw_fname = raw_series['fname_raw']
    raw_full_path = os.path.join(raw_dir, raw_fname)
    hdf_fname = raw_series['fname_hdf']
    hdf_full_path = os.path.join(hdf_dir, hdf_fname)

    if not proc_only:
        # check if file exists in filesystem
        if os.path.exists(raw_full_path):
            os.remove(raw_full_path)
            print("Deleted file: " + raw_full_path)
            exit_status = 0
        else:
            print("\nWarning: Image file not found at location: " + raw_full_path +
                  ". This may be the symptom of a larger problem.")
            exit_status = 1
    else:
        exit_status = 0

    # first check if there is an hdf file listed
    if hdf_fname != '':
        # check if file exists in filesystem
        if os.path.exists(hdf_full_path):
            os.remove(hdf_full_path)
            print("Deleted file: " + hdf_full_path)
        else:
            print("\nWarning: Processed HDF file not found at location: " + hdf_full_path +
                  ". This may be the symptom of a larger problem.")
            exit_status = exit_status + 2

    if not proc_only:
        # delete row where id = raw_id.  Use int to recover an INT from numpy.int64
        out_flag = db_session.query(EUV_Images).filter(EUV_Images.data_id == int(raw_id)).delete()
        out_flag2 = db_session.query(Data_Files).filter(Data_Files.data_id == int(raw_id)).delete()
        if out_flag == 0:
            exit_status = exit_status + 4
        elif out_flag == 1:
            db_session.commit()
            print("Row deleted from DB for data_id=" + str(raw_id))
    else:
        # delete fname_hdf from Data_Files table
        print("fname_hdf field of data_files table set to '' for data_id=" + str(raw_id))
        db_session.query(Data_Files).filter(Data_Files.data_id == raw_id).update(dict(fname_hdf=""))
        db_session.commit()

    return exit_status, db_session


def update_image_val(db_session, raw_series, col_name, new_val):
    """
    Change value for EUV_Images in row referenced from raw_series and column referenced in col_name
    raw_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("data_id", "date_obs", "instrument", "wavelength"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions. Alternatively one could use remove_euv_image() followed " +
              "by euvi.download_image_fixed_format(), add_image2session(), and db_session.commit()")
    else:
        raw_id = int(raw_series['data_id'])
        # for back-compatibility, redirect filename changes to data_files table
        if col_name in ("fname_raw", "fname_hdf"):
            db_session.query(Data_Files).filter(Data_Files.data_id == raw_id).update({col_name: new_val})
        else:
            db_session.query(EUV_Images).filter(EUV_Images.data_id == raw_id).update({col_name: new_val})
        db_session.commit()

    return db_session


def build_euvimages_from_fits(db_session, raw_data_dir, hdf_data_dir):
    """
    Iterate through the fits file directory and add each entry to the DB.

    :param db_session: SQLAlchemy database session
    :param raw_data_dir: base directory for all fits files
    :param hdf_data_dir: base directory for all processed hdf images
    :return: the updated database session
    """

    # walk through all subdirectories of raw_data_dir
    for (root, dirs, files) in os.walk(raw_data_dir, topdown=True):
        for filename in files:
            # look for fits files
            if filename.endswith(".fits"):
                # extract relative path
                relative_path = root.replace(raw_data_dir + '/', "")
                # extract metadata from file and write a row to the database (session)
                db_session = add_image2session(data_dir=raw_data_dir, subdir=relative_path, fname=filename,
                                               db_session=db_session)
        # commit changes to the DB
        db_session.commit()



    # now look for matching hdf files
    # first query image DB for all records
    result = pd.read_sql(db_session.query(EUV_Images).statement, db_session.bind)
    # iterate over each row
    for index, row in result.iterrows():
        # Extract metadata from fits file
        fits_path = os.path.join(raw_data_dir, row.fname_raw)
        fits_map = sunpy.map.Map(fits_path)
        chd_meta = get_metadata(fits_map)
        prefix, postfix, extension = io_helpers.construct_hdf5_pre_and_post(chd_meta)
        sub_dir, fname = io_helpers.construct_path_and_fname(
            hdf_data_dir, fits_map.date.datetime, prefix, postfix, extension,
            mkdir=False)
        hdf_rel_path = sub_dir.replace(hdf_data_dir + os.path.sep, '')

        hdf_full_path = os.path.join(hdf_data_dir, hdf_rel_path, fname)
        # check for matching hdf5 file
        if os.path.exists(hdf_full_path):
            # assume the file is good (??) and record in DB
            db_session = update_image_val(db_session=db_session, raw_series=row,
                                          col_name="fname_hdf", new_val=os.path.join(hdf_rel_path, fname))

    return db_session


def add_euv_map_old(db_session, combo_id, meth_id, fname, var_dict=None, time_of_compute=None):
    """
    Simultaneously add record to EUV_Maps and Var_Vals_Map.  Currently does not require
    var_dict or time_of_compute.
    :param db_session: SQLAlchemy database session
    :param combo_id: The combination ID associated with the map
    :param meth_id: Method ID
    :param fname: relative filepath and name
    :param var_dict: Python dictionary containing variable values indexed by name
    :param time_of_compute: Timestamp for when the map was generated.
    :return: The updated session;
             exit_status: 1 - filename already exists. Delete existing record before
             adding a new one. 2 - record successfully added.
    """

    # check if filename already exists in DB
    existing_fname = pd.read_sql(db_session.query(EUV_Maps.map_id).filter(EUV_Maps.fname == fname).statement,
                                 db_session.bind)
    if len(existing_fname) > 0:
        exit_status = 1
        map_id = None
    else:
        # check if meth_id/combo_id/var_vals_map already exist in DB?

        # create EUV_Maps object and add to session
        map_add = EUV_Maps(combo_id=combo_id, meth_id=meth_id, fname=fname, time_of_compute=time_of_compute)
        db_session.add(map_add)
        # commit to DB and return map_id
        db_session.flush()
        map_id = map_add.map_id
        exit_status = 0

        # determine variable IDs
        n_vars = len(var_dict)
        if n_vars is not None:
            var_info = pd.read_sql(db_session.query(Var_Defs).filter(Var_Defs.var_name.in_(var_dict.keys())).statement,
                                   db_session.bind)
            # Write variable values to Var_Vals_Map
            for index, var_row in var_info.iterrows():
                var_val = var_dict[var_row.var_name]
                add_var_val = Var_Vals_Map(map_id=map_id, combo_id=combo_id, meth_id=meth_id, var_id=var_row.var_id,
                                           var_val=var_val)
                db_session.add(add_var_val)

        # now commit EUV_Map update and Var_Vals_Map update simultaneously
        db_session.commit()

    return db_session, exit_status, map_id


def add_euv_map(db_session, psi_map, base_path=None, map_type=None):
    """
    Simultaneously add record to EUV_Maps and Var_Vals_Map. Write object to file.
    :param db_session:
    :param psi_map: PsiMap object for which all methods, variables, method_combo, and
    image_combo have been created in the database.  All variable information should
    be in psi_map.method_info
    :param base_path:
    :param map_type:
    :return: The updated session,
             exit_status: 1 - filename or map already exists. Delete existing record before
             adding a new one. 0 - record successfully added,
             map_id: the appropriate map_id
    TODO: EUV_map uniqueness should be enforced by a combination of combo_id,
        meth_combo_id, and the associated set of var_vals from the var_vals_map
        table. Programmatically, this is fairly straightforward to implement, but
        a little kludgy (Done).  The difficulty of implementing this in SQL is that
        the var_vals column must transform from a data column to a unique indexing
        column.  Then check that the SET of var_vals is distinct from any other
        set of var_vals. Not sure this can be done within SQL.
          Related to this issue: generate unique filenames for maps with different
        var_vals. Ex: same map with two different final resolutions.
    """
    # set a floating point epsilon (relative)
    rel_eps = 1E-6

    time_of_compute = psi_map.map_info.loc[0, 'time_of_compute'].to_pydatetime()
    fname = psi_map.map_info.loc[0, 'fname']
    # combo_id = psi_map.map_info.loc[0, 'combo_id'].__int__()
    valid_combo_ind = psi_map.map_info['combo_id'].index.get_loc(psi_map.map_info['combo_id'].last_valid_index())
    combo_id = psi_map.map_info.loc[valid_combo_ind, 'combo_id'].__int__()
    meth_combo_id = psi_map.map_info.loc[0, 'meth_combo_id'].__int__()

    if fname is not None and ~np.isnan(fname):
        # check if filename already exists in DB
        existing_fname = pd.read_sql(db_session.query(EUV_Maps.map_id).filter(EUV_Maps.fname == fname).statement,
                                     db_session.bind)
    else:
        existing_fname = []

    if len(existing_fname) > 0:
        print("Map object already exists in database with filename: " + fname +
              ".\n No file written.  No EUV_Map record added to database.")
        exit_status = 1  # There is already a map record for this file
        # psi_map.map_info.loc[0, 'map_id'] = existing_fname.loc[0, 'map_id']

    else:
        # Check that this map does not already exist in the DB
        # Do this by checking for matching combo_id and meth_combo_id.  If one or more matches are found,
        # check if the accompanying variable values match.
        combo_matches_query = db_session.query(EUV_Maps.map_id).filter(EUV_Maps.combo_id == combo_id,
                                                                       EUV_Maps.meth_combo_id == meth_combo_id)
        combo_matches = pd.read_sql(combo_matches_query.statement, db_session.bind)
        if len(combo_matches) > 0:
            map_matches = combo_matches.copy()
            no_match = False
            # check variable values
            for index, row in psi_map.method_info.iterrows():
                if np.isnan(row.var_id) or row.var_id is None or \
                        row.var_val is None or np.isnan(row.var_val):
                    # some methods have no associated variable, others have no-value variables,
                    # do not consider these
                    continue

                float_range = [row.var_val*(1 - rel_eps), row.var_val*(1 + rel_eps)]
                var_query = db_session.query(Var_Vals_Map.map_id).filter(Var_Vals_Map.map_id.in_(map_matches.map_id),
                                                                         Var_Vals_Map.var_id == row.var_id,
                                                                         Var_Vals_Map.var_val.between(
                                                                             float_range[0], float_range[1]))
                query_result = pd.read_sql(var_query.statement, db_session.bind)
                if len(query_result) == 0:
                    # if any variable is different, this map does not exist in the DB
                    no_match = True
                    break
                else:
                    # replace map_matches with maps that match this variable
                    map_matches = query_result
        else:
            no_match = True

        if no_match:
            # create EUV_Maps object and add to session
            map_add = EUV_Maps(combo_id=combo_id, meth_combo_id=meth_combo_id, fname=fname,
                               time_of_compute=time_of_compute)
            db_session.add(map_add)
            # commit to DB and return map_id
            db_session.flush()
            map_id = map_add.map_id
            # add map_id to map object
            psi_map.map_info.loc[0, 'map_id'] = map_id
            # if filename is not specified, auto-generate using map_id
            if fname is None or np.isnan(fname):
                # generate filename
                if len(psi_map.map_info) == 1:
                    inst = psi_map.data_info.instrument[0]
                else:
                    inst = None
                subdir, temp_fname = io_helpers.construct_map_path_and_fname(base_path, psi_map.map_info.date_mean[
                    valid_combo_ind], map_id, map_type, 'h5', inst=None, mkdir=True)
                h5_filename = os.path.join(subdir, temp_fname)
                # rel_file_path = h5_filename.replace(base_path, "")
                rel_file_path = os.path.relpath(h5_filename, base_path)
                # record file path in map object
                psi_map.map_info.loc[0, 'fname'] = rel_file_path
                # alter map record to reflect newly-generated filename
                map_add.fname = rel_file_path
            else:
                h5_filename = os.path.join(base_path, fname)

            # write map object to file
            psihdf.wrh5_fullmap(h5_filename, psi_map.x, psi_map.y, np.array([]), psi_map.data,
                                method_info=psi_map.method_info, data_info=psi_map.data_info,
                                map_info=psi_map.map_info, no_data_val=psi_map.no_data_val,
                                mu=psi_map.mu, origin_image=psi_map.origin_image, chd=psi_map.chd)

            # Loop over psi_map.method_info rows and insert variable values
            for index, var_row in psi_map.method_info.iterrows():
                if not np.isnan(var_row.var_val):
                    add_var_val = Var_Vals_Map(map_id=map_id, combo_id=combo_id, meth_id=var_row.meth_id,
                                               var_id=var_row.var_id, var_val=var_row.var_val)
                    # print(map_id, combo_id, var_row.meth_id, var_row.var_id, var_row.var_val)
                    db_session.add(add_var_val)

            # now commit EUV_Map update and Var_Vals update simultaneously
            db_session.commit()
            exit_status = 0  # New map record created
            print("PsiMap object written to:", h5_filename, "\nDatabase record created with map_id:", map_id, "\n")
        else:
            exit_status = 1
            print("This map already exists in the database with map_id:", map_matches.map_id[0],
                  "\nNo file written. No EUV_Maps record added.\n")

    return db_session, exit_status, psi_map


def update_euv_map(db_session, map_id, col_name, new_val):
    """
    Change value for EUV_Maps in row referenced from map_series and column referenced in col_name
    map_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("map_id", "combo_id", "meth_combo_id"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions.")
    else:
        db_session.query(EUV_Maps).filter(EUV_Maps.map_id == map_id).update({col_name: new_val})
        db_session.commit()

    return db_session


def remove_euv_map(db_session, map_info, map_dir):

    # loop through map_info rows
    for index, row in map_info.iterrows():
        # remove all var_vals_map records with matching map_id
        n_vals_del = db_session.query(Var_Vals_Map).filter(
            Var_Vals_Map.map_id == row.map_id).delete()
        # remove map from euv_maps
        n_maps_del = db_session.query(EUV_Maps).filter(
            EUV_Maps.map_id == row.map_id).delete()

        # delete map file
        if row.fname[0] == "/":
            rel_path = row.fname[1:]
        else:
            rel_path = row.fname
        file_path = os.path.join(map_dir, rel_path)
        # check that file exists
        if os.path.exists(file_path):
            # delete file
            print("Deleting file: ", rel_path)
            os.remove(file_path)
        else:
            print("No file associated with map_id", row.map_id, ". No file to delete.\n")

        # commit deletions to database
        db_session.commit()

    return db_session


def get_combo_id(db_session, meth_id, data_ids, create=False):
    """
    Function to query the database for an existing combination of images.  If it does not yet exist
    and create=True, the function will create a record for this combination.
    :param db_session: SQLAlchemy database session.
    :param meth_id: method id
    :param data_ids: a tuple of data_id values that correspond to records in the euv_images table.
    :param create: boolean flag to indicate if new combinations should written to DB.
    :return: integer value of combo_id. If create=False and the entered images are a new combination, return -1.
    """

    # query DB to determine if this combo exists.
    n_images = len(data_ids)
    # Return the number of matching images in each combo that has n_images and contains at least one of data_ids.
    # This version uses actual SQL for the subquery
    # match_groups = pd.read_sql("""SELECT combo_id, COUNT(data_id) AS i_count FROM Data_Combo_Assoc WHERE
    #                                   combo_id IN (SELECT combo_id FROM image_combos WHERE n_images=
    #                                   """ + str(n_images) + ") AND data_id IN (" + str(data_ids) +
    #                                   " GROUP BY combo_id;", db_session.bind)
    # This version uses SQLAlchemy to re-create the SQL
    match_groups = pd.read_sql(
        db_session.query(Data_Combo_Assoc.combo_id, func.count(Data_Combo_Assoc.data_id).label("i_count")).
            filter(Data_Combo_Assoc.combo_id.in_(
            db_session.query(Data_Combos.combo_id).filter(Data_Combos.n_images == n_images,
                                                          Data_Combos.meth_id == meth_id)
        ), Data_Combo_Assoc.data_id.in_(data_ids)
        ).group_by(Data_Combo_Assoc.combo_id).statement, db_session.bind)

    # for testing only
    # match_groups = pd.DataFrame(data={'combo_id': [1, 2], 'i_count': [2, 3]})
    # match_groups = pd.DataFrame(columns = ['combo_id', 'i_count'])

    if len(match_groups) > 0:
        # reduce match_groups to combos that match exactly
        match_groups = match_groups.loc[match_groups.i_count == n_images]
        if len(match_groups) == 1:
            # return the existing combo_id
            combo_id = match_groups.combo_id.values[0].item()
            no_match_exists = False
            # get combo date-times
            combo_info = pd.read_sql(db_session.query(Data_Combos).filter(Data_Combos.combo_id == combo_id,
                                                                          Data_Combos.meth_id == meth_id
                                                                          ).statement, db_session.bind)
            combo_times = {'date_mean': combo_info.date_mean[0].to_pydatetime(),
                           'date_max': combo_info.date_max[0].to_pydatetime(),
                           'date_min': combo_info.date_min[0].to_pydatetime()}
        else:
            no_match_exists = True
            combo_id = None
            combo_times = None

    else:
        no_match_exists = True
        combo_id = None
        combo_times = None

    if no_match_exists and create:
        # add record to image_combos
        # first retrieve records of images/data_files
        # image_pd = pd.read_sql(db_session.query(Data_Files, EUV_Images).filter(
        #     Data_Files.data_id.in_(data_ids),
        #     Data_Files.data_id == EUV_Images.data_id).statement, db_session.bind)
        image_pd = pd.read_sql(db_session.query(Data_Files, EUV_Images).filter(
            Data_Files.data_id.in_(data_ids)).join(
            EUV_Images, Data_Files.data_id == EUV_Images.data_id, isouter=True
            ).statement, db_session.bind)
        image_pd = image_pd.loc[:, ~image_pd.columns.duplicated()]
        # determine date range and mean
        date_max = image_pd.date_obs.max()
        date_min = image_pd.date_obs.min()
        date_mean = image_pd.date_obs.mean()
        # record in dict object
        combo_times = {'date_mean': date_mean, 'date_max': date_max, 'date_min': date_min}

        # determine instruments in combo
        all_instruments = image_pd.instrument.unique()
        if all_instruments.__len__() == 1 and all_instruments[0] is not None:
            # set this instrument value in 'instrument' column
            instrument = all_instruments[0]
        elif all_instruments.__len__() > 1:
            instrument = "MULTIPLE"
        else:
            instrument = None

        # generate record and add to session
        combo_add = Data_Combos(meth_id=meth_id, n_images=n_images, date_mean=date_mean, date_max=date_max,
                                date_min=date_min, instrument=instrument)
        db_session.add(combo_add)
        # Add record to DB
        db_session.commit()
        combo_id = combo_add.combo_id

        # Add image/combo associations to DB
        for data_id in data_ids:
            assoc_add = Data_Combo_Assoc(combo_id=combo_id, data_id=data_id)
            db_session.add(assoc_add)
        # commit changes to DB
        db_session.commit()

    return db_session, combo_id, combo_times


def add_combo_image_assoc(db_session, combo_id, data_id):
    """

    :param db_session:
    :param combo_id:
    :param data_id:
    :return:
    """

    # check if association already exists
    existing_assoc = pd.read_sql(db_session.query(Data_Combo_Assoc).filter(Data_Combo_Assoc.combo_id == combo_id,
                                                                           Data_Combo_Assoc.data_id == data_id
                                                                           ).statement, db_session.bind)

    # If association record does not exist, add it
    if len(existing_assoc.combo_id) == 0:
        assoc_add = Data_Combo_Assoc(combo_id=combo_id, data_id=data_id)
        db_session.add(assoc_add)
        # commit changes to DB
        db_session.commit()
        exit_flag = 1
    else:
        exit_flag = 0

    return db_session, exit_flag


def get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None, create=False):
    """
    Query DB for a method ID.  When create=True, if the method does not already exist,
    meth_desc and meth_vars are used to create the method definition in the DB.
    :param db_session: SQLAlchemy database session. Contains DB connection protocol.
    :param meth_name: Required. Method name as it appears/will appear in DB.
    :param meth_desc: Method description. Only used when creating a method record.
    :param create: Flag. If meth_name is not in DB and create==True, then a new method
    is created.
    :return: The new or existing method ID
    """

    # Query DB for existing method
    existing_meth = pd.read_sql(
        db_session.query(Method_Defs.meth_id).filter(Method_Defs.meth_name == meth_name).statement,
        db_session.bind)

    if len(existing_meth.meth_id) == 0:
        meth_exists = False
        meth_id = None
        var_ids = None
    else:
        # method already exists. lookup variable ids
        meth_exists = True
        meth_id = existing_meth.meth_id[0].item()
        if var_names != None:
            var_ids = [0] * len(var_names)
            for ii in range(len(var_names)):
                var_id_query = pd.read_sql(db_session.query(Var_Defs.var_id).filter(Var_Defs.meth_id == meth_id,
                                                                                    Var_Defs.var_name == var_names[ii]
                                                                                    ).statement, db_session.bind)
                # A method_id/var_name pair should return a unique var_id, so return a scalar--not a list
                # var_ids[ii] = var_id_query.var_id.to_list()
                var_ids[ii] = var_id_query.var_id.item()
        else:
            var_ids = None

    if create and not meth_exists:
        # create method record
        add_method = Method_Defs(meth_name=meth_name, meth_description=meth_desc)
        db_session.add(add_method)
        # push change to DB and return meth_id
        db_session.flush()
        meth_id = add_method.meth_id
        # add associated variables to var_defs
        if var_names != None:
            var_ids = [0] * len(var_names)
            for ii in range(len(var_names)):
                add_var = Var_Defs(meth_id=meth_id, var_name=var_names[ii], var_description=var_descs[ii])
                db_session.add(add_var)
                db_session.flush()
                var_ids[ii] = add_var.var_id
        else:
            var_ids = None
        db_session.commit()
    elif not meth_exists:
        var_ids = None

    return db_session, meth_id, var_ids


def get_var_id(db_session, meth_id, var_name=None, var_desc=None, create=False):
    """
    Query for variable id by var_name and meth_id
    :param db_session: db_session that connects to the database.
    :param meth_id method identification integer.
    :param var_name: string variable name.
    :param var_desc: Only required for create=True.
    :param create: If var_id does not exist, create it.
    :return:
    """

    # Query DB for existing variable
    existing_var = pd.read_sql(db_session.query(Var_Defs.var_id).filter(Var_Defs.var_name == var_name,
                                                                        Var_Defs.meth_id == meth_id).statement,
                               db_session.bind)

    if len(existing_var.var_id) == 0:
        var_exists = False
        var_id = None
    else:
        var_exists = True
        var_id = existing_var.var_id[0]

    if create and not var_exists:
        # create variable record
        add_var = Var_Defs(meth_id=meth_id, var_name=var_name, var_description=var_desc)
        db_session.add(add_var)
        # push change to DB and return var_id
        db_session.commit()
        var_id = add_var.var_id
    if not create and var_name is None:
        var_id = pd.read_sql(db_session.query(Var_Defs.var_id).filter(Var_Defs.meth_id == meth_id).statement,
                             db_session.bind)
    return db_session, var_id


def get_var_val(db_session, combo_id, meth_id, var_id, var_val=None, create=False):
    """
    Query for variable value by combo_id and var_id
    @param db_session: db_session that connects to the database.
    @param combo_id: combo_id from image_combos table
    @param meth_id: meth_id from method_defs table
    @param var_id: var_id from var_defs table
    @param var_val: float variable value
    @param create: If var_val does not exist, create it.
    @return:
    """

    # Query DB for existing variable value
    existing_var = pd.read_sql(db_session.query(Var_Vals).filter(Var_Vals.combo_id == combo_id,
                                                                 Var_Vals.var_id == var_id).statement,
                               db_session.bind)

    if len(existing_var) == 0:
        val_exists = False
    else:
        # value already exists
        val_exists = True
        if var_val is not None:
            db_session.query(Var_Vals).filter(Var_Vals.var_id == combo_id,
                                              Var_Vals.combo_id == var_id).update({Var_Vals.var_val: var_val})
            db_session.commit()
        else:
            var_val = existing_var.var_val[0]

    if create and not val_exists:
        # create variable value record
        add_val = Var_Vals(combo_id=combo_id, meth_id=meth_id, var_id=var_id, var_val=var_val)
        db_session.add(add_val)
        # push change to DB and return var_val
        db_session.commit()

    return db_session, var_val


def get_method_combo_id(db_session, meth_ids, create=False):
    # query DB to determine if this combo exists.
    n_meths = len(meth_ids)
    # Return the number of matching methods in each combo that has n_methods and contains at least one of meth_ids.
    match_groups = pd.read_sql(
        db_session.query(Method_Combo_Assoc.meth_combo_id, func.count(Method_Combo_Assoc.meth_combo_id).label(
            "m_count")).filter(Method_Combo_Assoc.meth_combo_id.in_(
            db_session.query(Method_Combos.meth_combo_id).filter(Method_Combos.n_methods == n_meths)
        ), Method_Combo_Assoc.meth_id.in_(meth_ids)
        ).group_by(Method_Combo_Assoc.meth_combo_id).statement, db_session.bind)

    if len(match_groups) > 0:
        # reduce match_groups to combos that match
        match_groups = match_groups.loc[match_groups.m_count == n_meths]
        if len(match_groups) == 1:
            # return the existing meth_combo_id
            meth_combo_id = match_groups.meth_combo_id.values[0]
            no_match_exists = False
        else:
            no_match_exists = True
            meth_combo_id = None

    else:
        no_match_exists = True
        meth_combo_id = None

    if no_match_exists and create:
        # add record to method_combos
        # generate record and add to session
        combo_add = Method_Combos(n_methods=n_meths)
        db_session.add(combo_add)
        db_session.flush()
        meth_combo_id = combo_add.meth_combo_id
        for id in meth_ids:
            # add method_id association to meth_combo. Be sure to convert to base python int
            assoc_add = Method_Combo_Assoc(meth_combo_id=meth_combo_id, meth_id=id)
            db_session.add(assoc_add)
        # Add record to DB
        db_session.commit()

    return db_session, meth_combo_id


def query_euv_maps(db_session, mean_time_range=None, extrema_time_range=None, n_images=None, data_ids=None,
                   methods=None, var_val_range=None, wavelength=None):
    """
    Query the database for maps that meet the input specifications.  db_session specifies database information.  All
    other inputs are query specifiers.  If more than one query specifier is 'not None', they are connected to one
    another using 'and' logic.
    :param db_session: SQLAlchemy database session.  Used for database connection and structure info.
    :param mean_time_range: Two element list of datetime values. Returns maps with mean_time in the range.
    :param extrema_time_range: Two element list of datetime values. Returns maps with min/max ranges that intersect
    extrema_time_range.
    :param n_images: An integer value. Returns maps made from n_images number of images.
    :param data_ids: A list of one or more integers.  Returns maps that include all images in data_ids.
    :param methods: A list of one or more character strings.  Returns maps that include all methods in 'methods'.
    :param var_val_range: A dict with variable names as element labels and a two-element list as values.  Ex
    {'par1': [min_par1, max_par1], 'par2': [min_par2, max_par2]}. Returns maps with parameter values in the
    specified range.
    :param wavelength: A list of one or more integer values. Returns maps with at least one image from each wavelength
    listed in 'wavelength'.
    :return: map_info, data_info, method_info - three pandas dataframes that summarize map details.
    """

    # determine map-combo method id
    map_combo_meth_id = db_session.query(Method_Defs.meth_id).filter(
        Method_Defs.meth_name == "Map - Data combo").first()[0]
    combo_query = db_session.query(Data_Combos.combo_id).filter(
            Data_Combos.meth_id == map_combo_meth_id)
    # combo query
    # n_images, data_ids, mean_time_range, extrema_time_range
    if mean_time_range is not None:
        # query combos by mean timestamp
        combo_query = combo_query.filter(
            Data_Combos.date_mean.between(mean_time_range[0], mean_time_range[1]))
        if extrema_time_range is not None:
            # AND combo must contain an image in extrema_time_range
            combo_query = combo_query.filter(or_(Data_Combos.date_min.between(extrema_time_range[0],
                                                                              extrema_time_range[1]),
                                                 Data_Combos.date_max.between(extrema_time_range[0],
                                                                              extrema_time_range[1])))
    elif extrema_time_range is not None:
        # query combos with an image in extrema_time_range
        combo_query = combo_query.filter(
            or_(Data_Combos.date_min.between(extrema_time_range[0], extrema_time_range[1]),
                Data_Combos.date_max.between(extrema_time_range[0], extrema_time_range[1])
                )
        )
    else:
        # this is a problem. mean_time_range or extrema_time_range must be defined
        sys.exit("query_euv_maps() requires that mean_time_range OR extrema_time_range be defined.")

    if n_images is not None:
        # AND combo must have Data_Combos.n_images IN(n_images)
        combo_query = combo_query.filter(Data_Combos.n_images == n_images)

    if data_ids is not None:
        # AND filter by combos that contain data_ids
        combo_ids_query = db_session.query(Data_Combo_Assoc.combo_id).filter(Data_Combo_Assoc.data_id.in_(data_ids))
        combo_query = combo_query.filter_by(Data_Combos.combo_id.in_(combo_ids_query))

    if wavelength is not None:
        # AND combo contains images with wavelength in 'wavelength'
        data_ids_query = db_session.query(EUV_Images.data_id).filter(EUV_Images.wavelength.in_(wavelength))
        combo_ids_query = db_session.query(Data_Combo_Assoc.combo_id).filter(Data_Combo_Assoc.data_id.in_(
            data_ids_query))
        combo_query = combo_query.filter_by(Data_Combos.combo_id.in_(combo_ids_query))

    # start the master EUV_Map query that references combo, method, and var queries. Includes an
    # join with data_combos
    euv_map_query = db_session.query(EUV_Maps, Data_Combos).filter(
        EUV_Maps.combo_id.in_(combo_query), EUV_Maps.combo_id == Data_Combos.combo_id
    ).order_by(Data_Combos.date_mean)

    # method query
    if methods is not None:
        # filter method_id by method names in 'methods' input
        method_ids_query = db_session.query(Method_Defs.meth_id).filter(Method_Defs.meth_name.in_(methods))
        # find method combos containing methods
        if len(methods) == 1:
            method_combo_query = db_session.query(Method_Combo_Assoc.meth_combo_id).filter(Method_Combo_Assoc.meth_id.in_(
                method_ids_query
            )).distinct()
        else:
            method_combo_query = db_session.query(Method_Combo_Assoc.meth_combo_id).filter(
                Method_Combo_Assoc.meth_id.in_(method_ids_query)).group_by(
                    Method_Combo_Assoc.meth_combo_id).having(func.count(
                        Method_Combo_Assoc.meth_id) == len(methods))
        # update master query
        euv_map_query = euv_map_query.filter(EUV_Maps.meth_combo_id.in_(method_combo_query))

    # variable value query
    if var_val_range is not None:
        # assume var_val_range is a dict of variable ranges with entries like: 'iter': [min, max]
        var_map_id_query = db_session.query(Var_Vals_Map.map_id)
        # setup a query to find a list of maps in the specified variable ranges
        for var_name in var_val_range:
            var_vals = var_val_range[var_name]
            var_id_query = db_session.query(Var_Defs.var_id).filter(Var_Defs.var_name == var_name)
            # Use intersect operator
            # temp_map_id_query = db_session.query(Var_Vals_Map.map_id).filter(Var_Vals_Map.var_id == var_id_query,
            #                                      Var_Vals_Map.var_val.between(var_vals[0], var_vals[1]))
            # var_map_id_query = var_map_id_query.intersect(temp_map_id_query)
            # Use inner join
            temp_map_id_subquery = db_session.query(Var_Vals_Map.map_id).filter(
                Var_Vals_Map.var_id.in_(var_id_query), Var_Vals_Map.var_val.between(
                    var_vals[0], var_vals[1])
            ).subquery()
            var_map_id_query = var_map_id_query.join(temp_map_id_subquery, Var_Vals_Map.map_id ==
                                                     temp_map_id_subquery.c.map_id)
            # var_map_id_query = var_map_id_query.filter(Var_Vals_Map.var_id == var_id_query,
            #                                            Var_Vals_Map.var_val.between(var_vals[0], var_vals[1]))

        # update master query
        euv_map_query = euv_map_query.filter(EUV_Maps.map_id.in_(var_map_id_query))

    # Need three output tables from SQL
    # 1. map_info: euv_maps joined with image_combos
    # 2. method_info: Var_Vals_Map joined with method_defs and var_defs
    # 3. data_info: euv_images joined with Data_Combo_Assoc

    # return map_info table using a join
    map_info = pd.read_sql(euv_map_query.join(Data_Combos).statement, db_session.bind)
    # remove duplicate combo_id columns
    map_info = map_info.T.groupby(level=0).first().T

    # return image info. also keep image/combo associations for map-object building below
    image_assoc = pd.read_sql(
        db_session.query(Data_Combo_Assoc).filter(Data_Combo_Assoc.combo_id.in_(map_info.combo_id)
                                                   ).statement, db_session.bind)
    data_info = pd.read_sql(db_session.query(EUV_Images, Data_Files).filter(
        EUV_Images.data_id.in_(image_assoc.data_id), EUV_Images.data_id == Data_Files.data_id
                                                     ).statement, db_session.bind)

    # return Var_Vals_Map joined with var_defs. they are not directly connected tables, so use explicit join syntax
    var_query = db_session.query(Var_Vals_Map, Var_Defs).join(Var_Defs, Var_Vals_Map.var_id == Var_Defs.var_id
                                                              ).filter(Var_Vals_Map.map_id.in_(map_info.map_id))
    joint_query = db_session.query(Method_Defs, Var_Defs, Var_Vals_Map).filter(
        Method_Defs.meth_id == Var_Defs.meth_id).filter(
        Var_Defs.var_id == Var_Vals_Map.var_id).filter(Var_Vals_Map.map_id.in_(map_info.map_id))
    method_info = pd.read_sql(joint_query.statement, db_session.bind)
    # remove duplicate var_id columns
    method_info = method_info.T.groupby(level=0).first().T

    return map_info, data_info, method_info, image_assoc


def get_euv_map_list(map_info, data_info, method_info, image_assoc):
    """
    This function returns a list of map objects. This could grow large quickly if too many maps are requested.
    The inputs are pandas dataframes as expected from the output of query_euv_maps()
    :param map_info:
    :param data_info:
    :param method_info:
    :param image_assoc:
    :return: a list of PSI_Map objects
    """
    map_list = [datatypes.PsiMap()]*len(map_info)
    map_data_dir = App.MAP_FILE_HOME
    for index, map_series in map_info.iterrows():
        print("Reading in map with map id:", map_series.map_id)
        hdf_path = map_data_dir + map_series.fname
        temp_map = datatypes.read_psi_map(hdf_path)
        # map_info
        map_list[index] = temp_map
        map_list[index].append_map_info(map_series)
        #     # data_info
        combo_id = map_series.combo_id
        data_ids = image_assoc.data_id[image_assoc.combo_id == combo_id]
        images_slice = data_info.loc[data_info.data_id.isin(data_ids)]
        map_list[index].append_data_info(images_slice)
        #     # method info
        map_list[index].append_method_info(method_info.loc[method_info.meth_id == map_series.meth_id])

    return map_list


def query_euv_map_list(db_session, mean_time_range=None, extrema_time_range=None, n_images=None, data_ids=None,
                       methods=None, var_val_range=None, wavelength=None):
    """
    Query the database for maps that meet the input specifications.  db_session specifies database information.  All
    other inputs are query specifiers.  If more than one query specifier is 'not None', they are connected to one
    another using 'and' logic.
    :param db_session: SQLAlchemy database session.  Used for database connection and structure info.
    :param mean_time_range: Two element list of datetime values. Returns maps with mean_time in the range.
    :param extrema_time_range: Two element list of datetime values. Returns maps with min/max ranges that intersect
    extrema_time_range.
    :param n_images: An integer value. Returns maps made from n_images number of images.
    :param data_ids: A list of one or more integers.  Returns maps that include all images in data_ids.
    :param methods: A list of one or more character strings.  Returns maps that include all methods in 'methods'.
    :param var_val_range: A dict with variable names as element labels and a two-element list as values.  Ex
    {'par1': [min_par1, max_par1], 'par2': [min_par2, max_par2]}. Returns maps with parameter values in the
    specified range.
    :param wavelength: A list of one or more integer values. Returns maps with at least one image from each wavelength
    listed in 'wavelength'.
    :return: map_info, data_info, method_info - three dataframes that summarize map details
             map_list - a list of the map objects loaded from query results
    """

    map_info, data_info, method_info, image_assoc = query_euv_maps(db_session, mean_time_range=mean_time_range,
                                                       extrema_time_range=extrema_time_range, n_images=n_images,
                                                       data_ids=data_ids, methods=methods,
                                                       var_val_range=var_val_range, wavelength=wavelength)

    # Then divide results up into a list of Map objects
    map_list = get_euv_map_list(map_info, data_info, method_info, image_assoc)

    return map_info, data_info, method_info, map_list


def create_map_input_object(new_map, fname, image_df, var_vals, method_name, time_of_compute=None):
    """
    This function generates a Map object and populates it with necessary info for creating a
    new map record in the DB.
    :param new_map: PsiMap object that has not been added to DB yet
    :param fname: relative file path, including filename
    :param time_of_compute: Datetime when the map was made
    :param image_df: DataFrame with columns 'data_id' describing the constituent images
    :param var_vals: DataFrame with columns 'var_name' and 'var_val' describing variable values
    :param method_name: Name of method used.  Must match method_defs.meth_name column in DB
    :return: The partially filled Map object needed to create a new map record in the DB
    """

    # construct method_info df
    method_info = pd.DataFrame(data=[[method_name, ]], columns=["meth_name", ])
    new_map.append_method_info(method_info)

    # add var_vals
    # Expects the input 'var_vals' to be a pandas DataFrame.  Requires fields 'var_name' and 'var_val'.
    new_map.append_var_info(var_vals)

    # add data_info
    # Expects the input 'image_df' to be a pandas DataFrame.  The only required column is 'data_id' and
    # should contain all the data_ids of this map's constituent images.  Can also be a complete euv_images-
    # structured DF.
    new_map.append_data_info(image_df)

    # construct map_info df
    map_info_df = pd.DataFrame(data=[[len(image_df), fname, time_of_compute], ],
                               columns=["n_images", "fname", "time_of_compute"])
    new_map.append_map_info(map_info_df)

    return new_map


def add_map_dbase_record(db_session, psi_map, base_path=None, map_type=None):
    """
    Add map record to database and write to hdf file.
    :param db_session:
    :param psi_map:
    :param base_path:
    :param map_type:
    :return:
    """
    # add generic Map - Data combo method
    psi_map.append_method_info(pd.DataFrame(data={'meth_name': ["Map - Data combo", ],
            'meth_description': ["Generic method for a map's combo_id", ],
            'var_name': ["combo_id", ], 'var_description':
            ["combo_id placeholder", ], 'var_val': [0, ]}))
    # generate method dataframes
    psi_map.method_info = psi_map.method_info.drop_duplicates()
    methods_df = psi_map.method_info.drop_duplicates(subset="meth_name")
    methods_df_cp = methods_df.copy()
    # extract/create method_id(s)
    for index, row in methods_df.iterrows():
        if row.meth_id is None or np.isnan(row.meth_id):
            var_index = psi_map.method_info.meth_name == row.meth_name
            temp_var_names = psi_map.method_info.var_name[var_index].to_list()
            temp_var_names = list(dict.fromkeys(temp_var_names))
            temp_var_descs = psi_map.method_info.var_description[var_index].to_list()
            temp_var_descs = list(dict.fromkeys(temp_var_descs))
            # if method_id is not passed in map object, query method id from existing methods table. Create if new
            if np.isnan(row.var_val):
                temp_var_names = None
                temp_var_descs = None
            db_session, temp_meth_id, temp_var_ids = get_method_id(db_session, row.meth_name,
                                                                   meth_desc=row.meth_description,
                                                                   var_names=temp_var_names,
                                                                   var_descs=temp_var_descs, create=True)
            methods_df_cp.loc[index, 'meth_id'] = temp_meth_id
            if temp_var_ids is not None:
                psi_map.method_info.loc[var_index, 'var_id'] = temp_var_ids
            # add method id back to psi_map.method_info
            if temp_meth_id is not None:
                psi_map.method_info.loc[var_index, 'meth_id'] = temp_meth_id
            # new_index = np.arange(0, len(psi_map.method_info))
            # psi_map.method_info = psi_map.method_info.reindex(new_index)
            # for index2, row2 in psi_map.method_info.iterrows():
            #     if row2.meth_name == row.meth_name:
            #         psi_map.method_info.loc[index2, 'meth_id'] = temp_meth_id
        else:
            # do nothing. method id is already defined
            pass

    # Get method combo_id. Create if it doesn't already exist
    meth_ids = methods_df_cp.meth_id.to_list()
    meth_ids = list(dict.fromkeys(meth_ids))
    db_session, meth_combo_id = get_method_combo_id(db_session, meth_ids, create=True)
    psi_map.map_info.loc[:, "meth_combo_id"] = meth_combo_id

    # Get image combo_id. Create if it doesn't already exist.
    data_ids = psi_map.data_info.data_id.to_list()
    data_ids = tuple(data_ids)
    combo_meth_id = get_method_id(db_session, meth_name="Map - Data combo", create=False)
    db_session, combo_id, combo_times = get_combo_id(
        db_session=db_session, meth_id=combo_meth_id[1], data_ids=data_ids, create=True)
    # record data_combo info in map object
    psi_map.map_info.loc[:, "combo_id"] = combo_id
    psi_map.map_info.loc[:, "date_mean"] = combo_times['date_mean']
    psi_map.map_info.loc[:, "date_max"] = combo_times['date_max']
    psi_map.map_info.loc[:, "date_min"] = combo_times['date_min']

    db_session, exit_status, psi_map = add_euv_map(db_session=db_session, psi_map=psi_map,
                                                   base_path=base_path,
                                                   map_type=map_type)

    return db_session, psi_map


def delete_map_dbase_record(db_session, map_object, data_dir=None):
    # determine map_id
    map_id = map_object.map_info.map_id[0]
    # determine filename
    fname_query = db_session.query(EUV_Maps.fname).filter(EUV_Maps.map_id == map_id).all()
    hdf_fname = fname_query[0].__getitem__(0)

    if data_dir is None:
        data_dir = App.PROCESSED_DATA_HOME
    data_full_path = os.path.join(data_dir, hdf_fname)

    # check if file exists in filesystem
    if os.path.exists(data_full_path):
        os.remove(data_full_path)
        print("Deleted file: " + data_full_path)
        exit_status = 0
    else:
        print("\nWarning: Map file not found at location: " + data_full_path +
              ". This may be the symptom of a larger problem.")
        exit_status = 1

    # delete variable values
    out_flag = db_session.query(Var_Vals_Map).filter(Var_Vals_Map.map_id == map_id).delete()
    if out_flag == 0:
        exit_status = exit_status + 2
    else:
        db_session.commit()
        print(str(out_flag) + " row(s) deleted from 'var_vals' for map_id=" + str(map_id))
    # delete map record
    out_flag = db_session.query(EUV_Maps).filter(EUV_Maps.map_id == map_id).delete()
    if out_flag == 0:
        exit_status = exit_status + 4
    else:
        db_session.commit()
        print(str(out_flag) + " row(s) deleted from 'euv_maps' for map_id=" + str(map_id))
    exit_status = 0

    return exit_status


def update_method_existing_map(db_session, map_id, psi_map, map_path):
    """
    Given an existing map and DB map record, update method info/combo.  A method change implies a change to the data,
    mu, origin_image, or chd fields. So in addition to updating the database, this function also overwrites the map
    file with the contents of 'psi_map'.
    :param db_session: SQLAlchemy database session.  Used for database connection and structure info.
    :param map_id: integer from DB map table specifying the existing map record
    :param psi_map: PsiMap class. Map object MUST INCLUDE new CHD grid and detection method.
    :param map_path: Local path to maps directory (not a complete path to the file).
    :return: flag indicating process result
                    0 - Operation success
                    1 -
                    2 - Failure, undefined method id
    """
    exit_status = 0
    # --- assign a new meth_combo_id ---
    # check that all methods in psi_map.method_info have method-id numbers
    if psi_map.method_info.meth_id.isnull().any():
        print("One or more null method_ids in psi_map. In DB_funs.update_method_existing_map(), all methods must be "
              "previously defined and meth_id inserted into map object prior to function call. Exiting function "
              "with status 2 - Failure, undefined method id.")
        exit_status = 2
        return exit_status

    # determine if any methods are new?
    existing_record = pd.read_sql(db_session.query(EUV_Maps).filter(
        EUV_Maps.map_id == map_id).statement, db_session.bind)
    if existing_record.shape[0] == 0:
        # map_id in psi_map does not match with a database record, search for different map_id?
        exit_status = 3
        print("No map record to modify. In DB_funs.update_method_existing_map(), 'map_id' must match an existing"
              "database map record. Exiting function with status 3 - Failure, undefined no database record for map_id.")
        return exit_status

    existing_methods_query = pd.read_sql(db_session.query(Method_Combo_Assoc).filter(
        Method_Combo_Assoc.meth_combo_id == existing_record.combo_id[0]).statement, db_session.bind)
    existing_methods = existing_methods_query.meth_id
    new_map_methods = psi_map.method_info.meth_id.unique()
    new_methods = set(new_map_methods) - set(existing_methods)

    # get a combo_id for the new combination of methods
    new_combo_id = get_method_combo_id(db_session, new_map_methods, create=True)
    # update map object
    psi_map.map_info.meth_combo_id = new_combo_id

    # --- update map record ---
    db_session.query(EUV_Maps).filter(EUV_Maps.map_id == map_id).update({"meth_combo_id": new_combo_id})

    # --- write variable values for new methods ---
    for method_id in new_methods:
        # loop through methods and write variable values
        method_index = np.where(psi_map.method_info.meth_id == method_id)
        for method_row in method_index:
            # write variable value
            add_var_val = Var_Vals_Map(map_id=map_id, combo_id=existing_record.combo_id[0],
                                       meth_id=psi_map.method_info.meth_id[method_row],
                                       var_id=psi_map.method_info.var_id[method_row],
                                       var_val=psi_map.method_info.var_val[method_row])
            db_session.add(add_var_val)

    # now commit the changes
    db_session.commit()

    # --- overwrite map file ---
    psi_map.write_to_file(map_path, filename=existing_record.fname[0], db_session=None)

    return exit_status


def read_sql2pandas(sql_query):
    """
    trial function to compare speeds with pd.read_sql()
    This function would maintain basic python datatypes (SQLAlchemy compatible). pd.read_sql() uses
    numpy numeric-types and a pandas datetime-type which are not SQLAlchemy compatible.
    :param sql_query:
    :return:
    """
    # execute query
    query_list = sql_query.all()
    result_type = type(query_list[0]).__name__
    # extract column names and datatypes
    if result_type == "result":
        column_names = query_list[0].keys()
        column_dtypes = []
        for element in query_list[0]:
            column_dtypes.append(type(element).__name__)
    else:
        column_names = []
        column_dtypes = []
        for table_column in query_list[0].__table__.columns:
            column_names.append(table_column.key)
        for element in query_list[0]:
            column_dtypes.append(type(element).__name__)

    # initialize pandas dataframe
    type_dict = dict(zip(column_names, column_dtypes))
    pd_out = pd.DataFrame(np.full((len(query_list), len(query_list[0])), 0), columns=column_names)
    pd_out.astype(type_dict)

    # loop through query results and index into dataframe

    return


def safe_datetime(unknown_datetime):
    """
    SQLAlchemy inputs and queries to the database require Timestamps in a datetime.datetime() format.
    Input a scalar or list/tuple/pd.Series of unknown TimeStamp-type.
    Convert the scalar/vector to a scalar/list of type datetime.datetime()
    :param unknown_datetime: a scalar or list/tuple/pd.Series of DateTimes with unknown type.
    :return: a scalar or list of type datetime.datetime()
    """

    not_list = False
    # if input is not iterable, try to make it a tuple
    if not isinstance(unknown_datetime, collections.Iterable):
        # assume that the input is a scalar and convert to tuple
        unknown_datetime = (unknown_datetime, )
        not_list = True

    # initialize output datetime.datetime() list
    datetime_out = [datetime.datetime(1, 1, 1, 0, 0, 0)] * len(unknown_datetime)

    for index, unknown_element in enumerate(unknown_datetime):
        # check for common package datetime classes and convert to datetime.datetime()
        if type(unknown_element) == datetime.datetime:
            element_out = unknown_element
        elif type(unknown_element) == pd._libs.tslibs.timestamps.Timestamp:
            element_out = unknown_element.to_pydatetime()
        elif type(unknown_element) == astro_time.core.Time:
            element_out = unknown_element.to_datetime()

        # check if successful
        if type(element_out) != datetime.datetime:
            sys.exit("Timestamp object could not be converted to datetime.datetime format.  The "
                     "datetime.datetime class is required for SQLAlchemy interaction with the "
                     "database.")

        datetime_out[index] = element_out

    if not_list:
        # convert back to single value
        datetime_out = datetime_out[0]

    return datetime_out


def pdseries_tohdf(pd_series):
    """
    :param pd_series: panda series for image to convert to hdf file
    :return: hdf5 filename of type string
    """
    f_name = pd_series.fname_hdf
    return f_name


def query_hist(db_session, meth_id, n_mu_bins=None, n_intensity_bins=None, lat_band=None,
               time_min=None, time_max=None, instrument=None, wavelength=None):
    """
    query histogram based on time frame
    @param db_session:
    @param meth_id:
    @param n_mu_bins:
    @param n_intensity_bins:
    @param lat_band:
    @param time_min:
    @param time_max:
    @param instrument:
    @param wavelength:
    @return: pandas data frame
    """

    if lat_band is None:
        # This default is not assigned in the function statement do to the problems associated
        # with 'mutable' default arguments
        lat_band = [-np.pi / 64., np.pi / 64.]
    # convert lat_band to float value
    lat_band_float = float(np.max(lat_band))

    if time_min is None and time_max is None:
        # get entire DB
        query_out = db_session.query(Histogram)
    elif not isinstance(time_min, datetime.datetime) or not isinstance(time_max, datetime.datetime):
        sys.exit("Error: time_min and time_max must have matching entries of 'None' or of type Datetime.")
    elif wavelength is None:
        if instrument is None:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.n_mu_bins == n_mu_bins,
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )
        elif n_mu_bins is None:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.instrument.in_(
                                                               instrument),
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )
        else:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.instrument.in_(
                                                               instrument),
                                                           Histogram.n_mu_bins == n_mu_bins,
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )
    else:
        if instrument is None:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.wavelength.in_(
                                                               wavelength),
                                                           Histogram.n_mu_bins == n_mu_bins,
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )
        elif n_mu_bins is None:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.instrument.in_(
                                                               instrument),
                                                           Histogram.wavelength.in_(
                                                               wavelength),
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )

        else:
            query_out = db_session.query(Histogram).filter(Histogram.date_obs >= time_min,
                                                           Histogram.date_obs <= time_max,
                                                           Histogram.meth_id == meth_id,
                                                           Histogram.instrument.in_(
                                                               instrument),
                                                           Histogram.wavelength.in_(
                                                               wavelength),
                                                           Histogram.n_mu_bins == n_mu_bins,
                                                           Histogram.n_intensity_bins == n_intensity_bins,
                                                           Histogram.lat_band.between(
                                                               (1. - 1e-4)*lat_band_float,
                                                               (1. + 1e-4)*lat_band_float)
                                                           )
    pd_out = pd.read_sql(query_out.order_by(Histogram.instrument, Histogram.date_obs)\
                         .with_hint(Histogram, 'USE INDEX (hist_index)').statement,
                         db_session.bind)
    # pd_out = pd.read_sql(query_out.statement, db_session.bind)

    return pd_out


def add_hist(db_session, histogram, overwrite=False):
    """
    Adds a row to the database session that references the hist location and metadata.
    The updated session will need to be committed - db_session.commit() - in order to
    write the new row to the DB.

    :param db_session: The SQLAlchemy database session.
    :param histogram: histogram class object
    :return: the updated SQLAlchemy session.
    """

    hist_identifier = histogram.instrument + " observed at " + str(histogram.date_obs)


    # convert arrays to correct binary format
    intensity_bin_edges, mu_bin_edges, hist = datatypes.hist_to_binary(histogram)

    # check if row already exists in DB
    existing_row_id = db_session.query(Histogram.hist_id).filter(
        Histogram.image_id == histogram.image_id,
        Histogram.meth_id == histogram.meth_id,
        Histogram.n_mu_bins == histogram.n_mu_bins,
        Histogram.n_intensity_bins == histogram.n_intensity_bins,
        Histogram.lat_band.between((1.-1e-4)*histogram.lat_band, (1.+1e-4)*histogram.lat_band)).all()
        # Histogram.image_id == histogram.image_id,
        # Histogram.meth_id == histogram.meth_id,
        # Histogram.n_mu_bins == histogram.n_mu_bins,
        # Histogram.n_intensity_bins == histogram.n_intensity_bins,
        # Histogram.instrument == histogram.instrument,
        # Histogram.date_obs == histogram.date_obs,
        # Histogram.lat_band == histogram.lat_band,
        # Histogram.wavelength == histogram.wavelength).all()
    if len(existing_row_id) == 1:
        existing_row_row = existing_row_id[0]
        existing_row_int = existing_row_row[0]
        if overwrite:
            print("Histogram currently exists for " + histogram.instrument + " " + str(histogram.date_obs) +
                  ".  Overwriting.")
            db_session.query(Histogram).filter(Histogram.hist_id == existing_row_int).update({Histogram.hist: hist})
        else:
            # histogram has already been downloaded and entered into DB. do nothing
            print("Histogram is already logged in database.  Nothing added.")
            pass
    elif len(existing_row_id) > 1:
        # This histogram already exists in the DB in MORE THAN ONE PLACE!
        print("Current download: " + hist_identifier + " already exists in the database MULTIPLE times. " +
              "Something is fundamentally wrong. DB unique index should " +
              "prevent this from happening.")
        sys.exit(0)
    else:
        # Add new entry to DB
        # Construct new DB table row
        hist_add = Histogram(image_id=histogram.image_id, meth_id=histogram.meth_id,
                             date_obs=histogram.date_obs,
                             instrument=histogram.instrument, wavelength=histogram.wavelength,
                             n_mu_bins=histogram.n_mu_bins, n_intensity_bins=histogram.n_intensity_bins,
                             lat_band=histogram.lat_band, intensity_bin_edges=intensity_bin_edges,
                             mu_bin_edges=mu_bin_edges, hist=hist)
        # Append to the list of rows to be added
        db_session.add(hist_add)
        print("Database row added for " + histogram.instrument + ", wavelength: " +
              str(histogram.wavelength) + ", timestamp: " + str(histogram.date_obs))
    db_session.commit()
    return db_session


def combo_bracket_date(db_session, target_date, meth_id, instrument="any"):

    # method_id = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
    #                           create=True)

    if instrument == "any":
        below_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean <= target_date,
            Data_Combos.meth_id == meth_id).order_by(Data_Combos.date_mean.desc())
        above_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean >= target_date,
            Data_Combos.meth_id == meth_id).order_by(Data_Combos.date_mean)
    else:
        below_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean <= target_date,
            Data_Combos.meth_id == meth_id,
            Data_Combos.instrument == instrument).order_by(Data_Combos.date_mean.desc())
        above_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean >= target_date,
            Data_Combos.meth_id == meth_id,
            Data_Combos.instrument == instrument).order_by(Data_Combos.date_mean)

    below_pd = pd.read_sql(below_query.limit(1).statement, db_session.bind)
    above_pd = pd.read_sql(above_query.limit(1).statement, db_session.bind)

    return below_pd, above_pd


def query_inst_combo(db_session, query_time_min, query_time_max, meth_name, instrument="any"):
    """
    Query database for combos with date_mean in time range and with specific method and instrument.
    :param db_session: database session with connection info embedded
    :param query_time_min: datetime minimum query time for combo date_mean
    :param query_time_max: datetime maximum query time for combo date_mean
    :param meth_name: method name as character string
    :param instrument: default value "any" ignores instrument column. Otherwise, acceptable values
    include "AIA", "EUVI-A", "EUVI-B", and None.
    :return: pandas dataframe of combos
    """
    method_id = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                              create=True)

    if instrument == "any":
        combo_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean.between(query_time_min, query_time_max),
            Data_Combos.meth_id == method_id[1])
    else:
        combo_query = db_session.query(Data_Combos).filter(
            Data_Combos.date_mean.between(query_time_min, query_time_max),
            Data_Combos.meth_id == method_id[1],
            Data_Combos.instrument == instrument)

    # retrieve query result (ordered by date_mean)
    combo_result = pd.read_sql(combo_query.order_by(Data_Combos.date_mean).statement,
                               db_session.bind)

    return combo_result


def query_inst_combo_old(db_session, query_time_min, query_time_max, meth_name, instrument):
    """
    query correct combination of image combos for certain instrument and method
    """
    method_id = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                              create=True)

    # inst_combo_query = pd.read_sql(
    #     db_session.query(Data_Combo_Assoc.combo_id).filter(Data_Combo_Assoc.data_id.in_(
    #         db_session.query(EUV_Images.data_id).filter(EUV_Images.instrument == instrument))).statement,
    #     db_session.bind)
    # combo_query = pd.read_sql(db_session.query(Data_Combos).filter(Data_Combos.date_mean <= query_time_max).
    #                           filter(Data_Combos.date_mean >= query_time_min).filter(
    #     Data_Combos.meth_id == method_id[1],
    #     Data_Combos.combo_id.in_(inst_combo_query.combo_id)).statement,
    #                           db_session.bind)

    # try to speed this up by first querying combos:
    combo_query = db_session.query(Data_Combos.combo_id).filter(Data_Combos.date_mean <= query_time_max).filter(
        Data_Combos.date_mean >= query_time_min).filter(Data_Combos.meth_id == method_id[1])

    # join instrument to image_combo info for candidate combos from combo_query
    image_inst_join = db_session.query(Data_Combo_Assoc, EUV_Images.instrument).filter(
        Data_Combo_Assoc.combo_id.in_(combo_query), EUV_Images.data_id == Data_Combo_Assoc.data_id)

    # execute query
    image_combo_instrument = pd.read_sql(image_inst_join.statement, db_session.bind)

    # Find two unique lists of combos
    inst_index = image_combo_instrument['instrument'].eq(instrument)
    list_A = image_combo_instrument.combo_id[inst_index].unique()
    list_B = image_combo_instrument.combo_id[~inst_index].unique()
    # determine correct combos by determining which exist in A but not in B
    combos_keep = np.setdiff1d(list_A, list_B, assume_unique=True)
    # return correct pandas format for back-compatibility (ordered by date_mean)
    combo_result = pd.read_sql(db_session.query(Data_Combos).filter(
        Data_Combos.combo_id.in_(combos_keep.tolist())).order_by(Data_Combos.date_mean).statement, db_session.bind)

    return combo_result


def query_var_val(db_session, meth_name, date_obs, inst_combo_query):
    """
    query variable value corresponding to image
    @param db_session:
    @param meth_name:
    @param date_obs:
    @param inst_combo_query:
    @return:
    """
    if type(date_obs) == str:
        date_obs = datetime.datetime.strptime(date_obs, "%Y-%m-%dT%H:%M:%S.%f")
    # query method_defs for method id
    method_id_info = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                   create=False)
    # query var defs for variable id
    var_id = get_var_id(db_session, method_id_info[1], var_name=None, var_desc=None, create=False)

    ### find correct combo id ###
    if date_obs <= inst_combo_query.date_mean.iloc[0]:
        correct_combo = inst_combo_query[(inst_combo_query['date_mean'] == str(inst_combo_query.date_mean.iloc[0]))]
        if len(inst_combo_query) > 1:
            correct_combo = correct_combo.append(
                inst_combo_query[(inst_combo_query['date_mean'] == str(inst_combo_query.date_mean.iloc[1]))])
    elif date_obs >= inst_combo_query.date_mean.iloc[-1]:
        correct_combo = inst_combo_query[(inst_combo_query['date_mean'] == str(inst_combo_query.date_mean.iloc[-1]))]
        if len(inst_combo_query) > 1:
            correct_combo = correct_combo.append(
                inst_combo_query[(inst_combo_query['date_mean'] == str(inst_combo_query.date_mean.iloc[-2]))])
    else:
        correct_combo = inst_combo_query[
            (inst_combo_query['date_mean'] >= str(date_obs - datetime.timedelta(days=7))) & (
                    inst_combo_query['date_mean'] <= str(date_obs + datetime.timedelta(days=7)))]

    # check for combo
    if correct_combo.shape[0] == 0:
        # if no combos in 2-week window, return NaN
        var_vals = np.zeros((1, var_id[1].size))
        var_vals.fill(np.nan)
        return var_vals

    # create empty arrays
    var_vals = np.zeros((len(correct_combo.combo_id), var_id[1].size))
    date_mean = np.zeros((len(correct_combo.combo_id)))
    # query var_vals for variable values
    for i, combo_id in enumerate(correct_combo.combo_id):
        # query variable values
        var_val_query = pd.read_sql(db_session.query(Var_Vals).filter(Var_Vals.meth_id == method_id_info[1],
                                                                      Var_Vals.combo_id == combo_id
                                                                      ).order_by(Var_Vals.var_id).statement,
                                    db_session.bind)
        if var_val_query.size == 0:
            var_vals[i, :] = np.nan * var_id[1].size
        else:
            var_vals[i, :] = var_val_query.var_val
            timeutc = datetime.datetime.utcfromtimestamp(0)
            date_float = (correct_combo.date_mean.iloc[i] - timeutc).total_seconds()
            date_mean[i] = date_float

    # interpolate
    var_val = np.zeros(var_id[1].size)
    if len(correct_combo.combo_id) > 1:
        for i in range(var_id[1].size):
            interp_values = interpolate.interp1d(x=date_mean, y=var_vals[:, i], fill_value='extrapolate')
            date_obs_fl = (date_obs - timeutc).total_seconds()
            var_val[i] = interp_values(date_obs_fl)
    else:
        var_val[:] = var_vals[:, :]
    return var_val


def get_correction_pars(db_session, meth_name, date_obs, instrument="any"):

    if type(date_obs) == str:
        date_obs = datetime.datetime.strptime(date_obs, "%Y-%m-%dT%H:%M:%S.%f")
    # query method_defs for method id
    method_id_info = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None,
                                   create=False)
    # query var defs for variable id
    var_id = get_var_id(db_session, method_id_info[1], var_name=None, var_desc=None, create=False)

    ### find correct combo id(s) ###
    before_combo, after_combo = combo_bracket_date(db_session, target_date=date_obs, meth_id=method_id_info[1],
                                                   instrument=instrument)

    if before_combo.empty:
        query_combos = after_combo
    elif after_combo.empty:
        query_combos = before_combo
    else:
        query_combos = pd.concat([before_combo, after_combo], sort=False, ignore_index=True)

    # query variable values
    var_val_query = pd.read_sql(db_session.query(Var_Vals).filter(Var_Vals.meth_id == method_id_info[1],
                                                                  Var_Vals.combo_id.in_(query_combos.combo_id)
                                                                  ).order_by(Var_Vals.var_id).statement,
                                db_session.bind)

    var_val = np.zeros(var_id[1].size)
    if query_combos.shape[0] == 1:
        var_val[:] = var_val_query.var_val
    else:
        # interpolate
        v1 = var_val_query.var_val.loc[var_val_query.combo_id.eq(query_combos.combo_id.iloc[0])]
        v1 = v1.to_list()
        v2 = var_val_query.var_val.loc[var_val_query.combo_id.eq(query_combos.combo_id.iloc[1])]
        v2 = v2.to_list()
        # convert combo dates to number of seconds
        timeutc = datetime.datetime.utcfromtimestamp(0)
        date_means_float = [(query_combos.date_mean.iloc[0] - timeutc).total_seconds(),
                            (query_combos.date_mean.iloc[1] - timeutc).total_seconds()]
        # convert target date to number of seconds
        date_obs_float = (date_obs - timeutc).total_seconds()

        for ii in range(var_id[1].size):
            interp_values = interpolate.interp1d(x=date_means_float, y=[v1[ii], v2[ii]], fill_value='extrapolate')
            var_val[ii] = interp_values(date_obs_float)

    return var_val


def return_closest_combo(db_session, class_name, class_column, meth_id, inst_query, time):
    """
    function to return combo_id with mean date closest to that of date observed
    @param db_session:
    @param class_name: name of sqlalchemy class
    @param class_column: class_name.column_name
    @param meth_id: method id
    @param inst_query: return of query for combo_ids of instrument
    @param time: date observed for image
    @return:
    """

    greater = db_session.query(class_name).filter(class_column > time,
                                                  class_name.meth_id == meth_id,
                                                  class_name.combo_id.in_(inst_query.combo_id)). \
        order_by(class_name.date_mean.asc()).limit(1).subquery().select()

    lesser = db_session.query(class_name).filter(class_column <= time,
                                                 class_name.meth_id == meth_id,
                                                 class_name.combo_id.in_(inst_query.combo_id)). \
        order_by(class_column.desc()).limit(1).subquery().select()

    the_union = union_all(lesser, greater).alias()
    the_alias = aliased(class_name, the_union)
    the_diff = getattr(the_alias, class_column.name) - time
    abs_diff = case([(the_diff < datetime.timedelta(0), -the_diff)], else_=the_diff)

    image_combo_query = pd.read_sql(db_session.query(the_alias).order_by(abs_diff.asc()).statement, db_session.bind)
    return image_combo_query


def store_lbcc_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index, inst_index, optim_vals,
                      results, create=False):
    """

    :param db_session:
    :param pd_hist:
    :param meth_name:
    :param meth_desc:
    :param var_name:
    :param var_desc:
    :param date_index:
    :param inst_index:
    :param optim_vals:
    :param results:
    :param create:
    :return:
    """
    # create image combos in db table
    # get image_ids from queried histograms - same as ids in euv_images table
    image_ids = tuple(pd_hist['image_id'])
    # get method id
    method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=create)
    method_id = method_id_info[1]
    # combo id
    combo_id_info = get_combo_id(db_session, method_id, image_ids, create=create)
    combo_id = combo_id_info[1]

    # create association between combo and image_id
    # this is now handled in get_combo_id()
    # for image_id in image_ids:
    #     add_combo_image_assoc(db_session, combo_id, image_id)

    # create variables in database
    for i in range(len(optim_vals)):
        #### definitions #####
        # create variable definitions
        var_name_i = var_name + str(i)
        var_desc_i = var_desc + str(i)

        ##### store values #####
        # add method to db
        # method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=create)
        # method_id = method_id_info[1]
        # add variable to db
        var_id_info = get_var_id(db_session, method_id, var_name_i, var_desc_i, create=create)
        var_id = int(var_id_info[1])
        # add variable value to database
        var_val = results[date_index, inst_index, i]
        db_session, var_val_db = get_var_val(db_session, combo_id, method_id, var_id, var_val, create=create)
        # test if input val and output val are within 4 sig figs
        # if ~np.isclose(var_val, var_val_db, rtol=1e-4):
        #     print("WARNING: LBCC variable value already exists in DB and was NOT overwritten. ",
        #           "In DB_funs.store_lbcc_values(), for combo_id: ", combo_id, ", method_id: ", method_id,
        #           ", and variable: ", var_name_i, sep="")

    return db_session


def store_iit_values(db_session, pd_hist, meth_name, meth_desc, alpha_x_parameters, create=False):
    """

    :param db_session:
    :param pd_hist:
    :param meth_name:
    :param meth_desc:
    :param alpha_x_parameters:
    :param create:
    :return:
    """
    # create image combos in db table
    # get image_ids from queried histograms - same as ids in euv_images table
    image_ids = tuple(pd_hist['image_id'])
    # add method to db
    var_names = ["alpha", "x"]
    var_descs = ["IIT scale factor", "IIT offset"]
    method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=var_names, var_descs=var_descs,
                                   create=create)
    method_id = method_id_info[1]
    # get combo_ids
    combo_id_info = get_combo_id(db_session, meth_id=method_id, data_ids=image_ids, create=create)
    combo_id = combo_id_info[1]

    # create association between combo and image_id
    # this is now handled in get_combo_id()
    # for image_id in image_ids:
    #     add_combo_image_assoc(db_session, combo_id, image_id)

    # create variables in database
    for i in range(len(alpha_x_parameters)):
        #### definitions #####
        # create variable definitions
        # if i == 0:
        #     var_name = "alpha"
        #     var_desc = "IIT scale factor"
        # elif i == 1:
        #     var_name = "x"
        #     var_desc = "IIT offset"
        var_name = var_names[i]
        var_desc = var_descs[i]
        ##### store values #####
        # add method to db
        # method_id_info = get_method_id(db_session, meth_name, meth_desc, var_name, var_desc, create=create)
        # method_id = method_id_info[1]
        # recover variable id
        var_id_info = get_var_id(db_session, method_id, var_name, var_desc, create=create)
        var_id = int(var_id_info[1])
        # add variable value to database
        var_val = alpha_x_parameters[i]
        db_session, var_val_db = get_var_val(db_session, combo_id, method_id, var_id, var_val, create=create)
        # test if input val and output val are within 4 sig figs (rough check to see if value already existed)
        # if ~np.isclose(var_val, var_val_db, rtol=1e-4):
        #     print("WARNING: IIT variable value already exists in DB and was NOT overwritten. ",
        #           "In DB_funs.store_iit_values(), for combo_id: ", combo_id, ", method_id: ", method_id,
        #           ", and variable: ", var_name, sep="")

    return db_session


def generate_methdf(query_pd):
    meth_columns = []
    for column in Method_Defs.__table__.columns:
        meth_columns.append(column.key)
    defs_columns = []
    for column in Var_Defs.__table__.columns:
        defs_columns.append(column.key)
    df_cols = set().union(meth_columns, defs_columns, ("var_val",))
    methods_template = pd.DataFrame(data=None, columns=list(df_cols))
    # generate a list of methods dataframes
    methods_list = [methods_template] * query_pd.__len__()
    return methods_list


#### NOT CURRENTLY USED ####


def add_corrected_image(db_session, corrected_image):
    corrected_image_id = corrected_image.instrument + " observed at " + str(corrected_image.date_obs)

    # check if row already exists in DB
    existing_row_id = db_session.query(Corrected_Images.image_id).filter(
        Corrected_Images.image_id == corrected_image.image_id,
        Corrected_Images.meth_id == corrected_image.meth_id,
        # Corrected_Images.n_intensity_bins == corrected_image.n_intensity_bins,
        Corrected_Images.instrument == corrected_image.instrument,
        Corrected_Images.date_obs == corrected_image.date_obs,
        Corrected_Images.wavelength == corrected_image.wavelength).all()

    if len(existing_row_id) == 1:
        # lbcc image has already been downloaded and entered into DB. do nothing
        print("LBCC Image is already logged in database.  Nothing added.")
        pass
    elif len(existing_row_id) > 1:
        # This lbcc image already exists in the DB in MORE THAN ONE PLACE!
        print("Current download: " + corrected_image_id + " already exists in the database MULTIPLE times. " +
              "Something is fundamentally wrong. DB unique index should prevent this from happening.")
        sys.exit(0)
    else:
        # add this lbcc image to the database
        # convert LBCC Image to binary
        mu_array, lat_array, corrected_data = datatypes.LBCCImage.lbcc_to_binary(corrected_image)
        corrected_image_add = Corrected_Images(image_id=corrected_image.image_id, meth_id=corrected_image.meth_id,
                                               date_obs=corrected_image.date_obs, instrument=corrected_image.instrument,
                                               wavelength=corrected_image.wavelength, distance=corrected_image.distance,
                                               cr_lon=corrected_image.cr_lon, cr_lat=corrected_image.cr_lat,
                                               cr_rot=corrected_image.cr_rot, lat_array=lat_array,
                                               mu_array=mu_array, corrected_data=corrected_data)
        # Append to the list of rows to be added
        db_session.add(corrected_image_add)
        print("Database row added for", corrected_image.instrument, "at time", corrected_image.date_obs)
    # commit to database
    db_session.commit()
    return db_session


def query_corrected_images(db_session, time_min=None, time_max=None, instrument=None, wavelength=None):
    if time_min is None and time_max is None:
        # get entire DB
        query_out = pd.read_sql(db_session.query(Corrected_Images).statement, db_session.bind)
    elif not isinstance(time_min, datetime.datetime) or not isinstance(time_max, datetime.datetime):
        sys.exit("Error: time_min and time_max must have matching entries of 'None' or of type Datetime.")
    elif wavelength is None:
        if instrument is None:
            query_out = pd.read_sql(db_session.query(Corrected_Images).filter(Corrected_Images.date_obs >= time_min,
                                                                              Corrected_Images.date_obs <= time_max).statement,
                                    db_session.bind)
        else:
            query_out = pd.read_sql(db_session.query(Corrected_Images).filter(Corrected_Images.date_obs >= time_min,
                                                                              Corrected_Images.date_obs <= time_max,
                                                                              Corrected_Images.instrument.in_(
                                                                                  instrument)
                                                                              ).statement,
                                    db_session.bind)
    else:
        if instrument is None:
            query_out = pd.read_sql(db_session.query(Corrected_Images).filter(Corrected_Images.date_obs >= time_min,
                                                                              Corrected_Images.date_obs <= time_max,
                                                                              Corrected_Images.wavelength.in_(
                                                                                  wavelength)).statement,
                                    db_session.bind)
        else:
            query_out = pd.read_sql(db_session.query(Corrected_Images).filter(Corrected_Images.date_obs >= time_min,
                                                                              Corrected_Images.date_obs <= time_max,
                                                                              Corrected_Images.instrument.in_(
                                                                                  instrument),
                                                                              Corrected_Images.wavelength.in_(
                                                                                  wavelength)).statement,
                                    db_session.bind)
    return query_out


def store_mu_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, date_index,
                    inst_index, ii, optim_vals, results, create=False):
    # create image combos in db table
    # get image_ids from queried histograms - same as ids in euv_images table
    image_ids = tuple(pd_hist['image_id'])
    method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=create)
    method_id = method_id_info[1]
    # get combo id
    combo_id_info = get_combo_id(db_session, method_id, image_ids, create)
    combo_id = combo_id_info[1]

    # create association between combo and image_id
    # this is now handled in get_combo_id()
    # for image_id in image_ids:
    #     add_combo_image_assoc(db_session, combo_id, image_id)

    # create variables in database
    for i in range(len(optim_vals)):
        #### definitions #####
        # create variable definitions
        var_name_i = var_name + str(i) + str(ii)
        var_desc_i = var_desc + str(i) + str(ii)

        ##### store values #####
        # add variable to db
        # add method to db
        method_id_info = get_method_id(db_session, meth_name, meth_desc, var_name_i, var_desc_i, create=create)
        method_id = method_id_info[1]
        var_id_info = get_var_id(db_session, method_id, var_name_i, var_desc_i, create)
        var_id = int(var_id_info[1])
        # add variable value to database
        var_val = results[date_index, inst_index, ii, i]
        get_var_val(db_session, combo_id, method_id, var_id, var_val, create)

    return db_session


def store_beta_y_values(db_session, pd_hist, meth_name, meth_desc, var_name, var_desc, beta_y_parameters, create=False):
    # create image combos in db table
    # get image_ids from queried histograms - same as ids in euv_images table
    image_ids = tuple(pd_hist['image_id'])
    # get method id
    method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=create)
    method_id = method_id_info[1]
    # get combo id
    combo_id_info = get_combo_id(db_session, image_ids, create)
    combo_id = combo_id_info[1]

    # create association between combo and image_id
        # this is now handled in get_combo_id()
    # for image_id in image_ids:
    #     add_combo_image_assoc(db_session, combo_id, image_id)

    # create variables in database]
    for i in range(len(beta_y_parameters)):
        #### definitions #####
        # create variable definitions
        if i == 0:
            beta_y = "Beta"
        elif i == 1:
            beta_y = "Y"

        var_name_i = var_name + beta_y
        var_desc_i = var_desc + beta_y

        ##### store values #####
        # add method to db
        method_id_info = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=create)
        method_id = method_id_info[1]
        # add variable to db
        var_id_info = get_var_id(db_session, method_id, var_name_i, var_desc_i, create=create)
        var_id = int(var_id_info[1])
        # add variable value to database
        # beta_y_binary = beta_y_parameters[i].tobytes()
        var_val = beta_y_parameters[i]
        get_var_val(db_session, combo_id, method_id, var_id, var_val, create=create)

    return db_session


def query_lbcc_fit(db_session, image, meth_name):
    data_id = image.data_id
    date_obs = image.date_obs
    instrument = image.instrument

    # get method id based on method name
    method_id_info = get_method_id(db_session, meth_name, meth_desc=None, var_names=None, var_descs=None, create=create)

    # query for combo id
    # check date_obs against mean date of image combo
    image_combo_query = pd.read_sql(db_session.query(Data_Combos).filter(Data_Combos.date_mean == date_obs
                                                                          ).statement, db_session.bind)

    # if none, then need to find closest center date
    if len(image_combo_query) == 0:
        image_combo_query = return_closest_combo(db_session, Data_Combos, Data_Combos.date_mean, date_obs)

    # get variable id info based on method and combo id
    var_id_query = pd.read_sql(db_session.query(Var_Vals.var_id).filter(Var_Vals.meth_id == method_id_info.meth_id,
                                                                        Var_Vals.combo_id == image_combo_query.combo_id).statement,
                               db_session.bind)

    # query var_val for variables values near the date_obs
    var_vals = np.zeros((len(var_id_query)))
    for i, var_id in enumerate(var_id_query.var_id):
        var_val_query = pd.read_sql(db_session.query(Var_Vals).filter(Var_Vals.combo_id == image_combo_query.combo_id,
                                                                      Var_Vals.var_id == var_id
                                                                      ).statement,
                                    db_session.bind)
        if i == 0:
            beta = var_val_query.var_val[0]
        if i == 1:
            y = var_val_query.var_val[0]

    return beta, y


def sync_local_filesystem(base_local_path, base_remote_path, user, raw_image=True, processed_image=True, euv_map=True,
                          raw_mmap=True, dry_run=True, verbose=True, size_only=False):
    """

    :param base_local_path:
    :param base_remote_path:
    :param user:
    :param raw_image:
    :param processed_image:
    :param euv_map:
    :param raw_mmap:
    :param dry_run:
    :param verbose:
    :param size_only:
    :return:
    TODO: rsync does not take password as an argument. This is not going to work. Just print appropriate shell commands?
    """

    # prompt user for their password
    user_pass = getpass.getpass("Enter password for Q:")

    base_rsync_string = "rsync -a"
    if dry_run:
        base_rsync_string = base_rsync_string + " --dry-run"
    if verbose:
        base_rsync_string = base_rsync_string + " -v"
    if size_only:
        base_rsync_string = base_rsync_string + " --size_only"

    base_rsync_string = base_rsync_string + " --exclude='*.AppleDouble*' "

    if raw_image:
        remote_path = user + "@q.predsci.com" + base_remote_path + "raw_images "
        local_path = base_local_path
        rsync_string = base_rsync_string + remote_path + local_path
        os.system(rsync_string)

    if processed_image:
        remote_path = user + "@q.predsci.com" + base_remote_path + "processed_images "
        local_path = base_local_path
        rsync_string = base_rsync_string + remote_path + local_path
        os.system(rsync_string)

    if euv_map:
        remote_path = user + "@q.predsci.com" + base_remote_path + "maps "
        local_path = base_local_path
        rsync_string = base_rsync_string + remote_path + local_path
        os.system(rsync_string)

    if raw_mmap:
        remote_path = user + "@q.predsci.com" + base_remote_path + "raw_mmaps "
        local_path = base_local_path
        rsync_string = base_rsync_string + remote_path + local_path
        os.system(rsync_string)


def combo_clean_up(db_session):

    # query for used combo_ids
    var_vals_combo_ids = pd.read_sql(db_session.query(distinct(Var_Vals.combo_id).label("combo_id")).statement,
                                     db_session.bind)
    var_vals_map_combo_ids = pd.read_sql(db_session.query(distinct(Var_Vals_Map.combo_id).label("combo_id")).statement,
                                         db_session.bind)
    euv_maps_combo_ids = pd.read_sql(db_session.query(distinct(EUV_Maps.combo_id).label("combo_id")).statement,
                                     db_session.bind)
    # collect a list of unique values
    used_combos = var_vals_combo_ids.combo_id.append(var_vals_map_combo_ids.combo_id).unique()
    used_combos = pd.Series(used_combos).append(euv_maps_combo_ids.combo_id).unique()

    # query for unused combo_ids
    unused_ids = pd.read_sql(db_session.query(Data_Combos.combo_id).filter(
        Data_Combos.combo_id.notin_(used_combos.tolist())
    ).statement, db_session.bind)
    delete_ids = unused_ids.combo_id.to_list()

    # delete unused combo_ids
    if delete_ids.__len__() > 0:
        ica_rows_removed = db_session.query(Data_Combo_Assoc).filter(
            Data_Combo_Assoc.combo_id.in_(delete_ids)
            ).delete(synchronize_session=False)
        ic_rows_removed = db_session.query(Data_Combos).filter(
            Data_Combos.combo_id.in_(delete_ids)
            ).delete(synchronize_session=False)

        db_session.commit()
    else:
        ica_rows_removed = 0
        ic_rows_removed = 0

    return delete_ids, ic_rows_removed, ica_rows_removed


def init_data_files_from_images(db_session):
    """
    Use the contents of 'euv_images' table to populate 'data_files' table.

    This function assumes 'data_files' to be empty and is really only useful
    when transitioning from an older version of the DB to current format. The
    data_files.provider column defaults to euv_images.instrument here, but
    this may get updated to something more specific later.

    Parameters
    ----------
    db_session : sqlalchemy.orm.session.Session
                An active sqlalchemy session that connects to the database.

    Returns
    -------
    0
    """
    # read EUV-Images table
    images_query = db_session.query(EUV_Images)
    images_pd = pd.read_sql(images_query.statement, db_session.bind)

    # use images info to write to data_files table
    for index, row in images_pd.iterrows():
        image_add = EUV_Images(data_id=row.data_id, date_obs=row.date_obs, provider=row.instrument,
                               type="EUV Image", fname_raw=row.fname_raw, fname_hdf=row.fname_hdf,
                               flag=row.flag)
        # Append to the list of rows to be added
        db_session.add(image_add)

    # commit the rows to the DB
    db_session.commit()

    return 0
