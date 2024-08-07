"""
Create functions for initializing connection to Database,
adding entries, querying, etc.
"""

import os
import sys
import urllib.parse
import pandas as pd
import datetime


import sunpy.map

from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import sessionmaker

from chmap.settings.app import App
from chmap.database.db_classes import *
from chmap.data.download.euv_utils import get_metadata
from chmap.utilities.file_io import io_helpers


# from modules import datatypes


def init_db_conn(db_name, chd_base, sqlite_path=""):
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
    if db_name=='mysql-shadow':
        db_user = "test_beta"
        db_psswd = "ATlz8d40gh^W7Ge6"
        url_psswd = urllib.parse.quote_plus(db_psswd)
        connect_string = 'mysql://'+db_user+':'+url_psswd+'@shadow.predsci.com:3306/CHD'
    elif db_name=='sqlite':
        print("Attempting to connect to SQLite DB file " + sqlite_path)
        connect_string = 'sqlite:///' + sqlite_path
    else:
        sys.exit("At this time, 'db_name' must be either 'sqlite' or 'mysql-shadow'.")

    # now establish the engine
    engine = create_engine(connect_string)
    # import base declarative table definitions into the engine
    chd_base.metadata.create_all(engine)
    # define the session
    session = sessionmaker(bind=engine)
    # open session/connection to DB
    db = session()

    return db


def query_euv_images(db_session, time_min=None, time_max=None, instrument=None, wavelength=None):
    """User-friendly query function.  Allows user to avoid using SQLAlchemy or SQL
    syntax.  Default behavior is to return the entire database.
    time_min, time_max - datetime objects that define the time interval to search.
    instrument - list of spacecraft to include (characters)
    wavelength - list of wavelengths to include (integer)
    db_session - database sqlalchemy session object
    """

    if time_min is None and time_max is None:
        # get entire DB
        query_out = pd.read_sql(db_session.query(EUV_Images).statement, db_session.bind)
    elif not isinstance(time_min, datetime.datetime) or not isinstance(time_max, datetime.datetime):
        sys.exit("Error: time_min and time_max must have matching entries of 'None' or of type Datetime.")
    elif wavelength is None:
        if instrument is None:
            query_out = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.date_obs>=time_min,
                                                                        EUV_Images.date_obs<=time_max).statement,
                                    db_session.bind)
        else:
            query_out = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.date_obs >= time_min,
                                                                        EUV_Images.date_obs <= time_max,
                                                                        EUV_Images.instrument.in_(instrument)).statement,
                                    db_session.bind)
    else:
        if instrument is None:
            query_out = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.date_obs>=time_min,
                                                                        EUV_Images.date_obs<=time_max,
                                                                        EUV_Images.wavelength.in_(wavelength)).statement,
                                    db_session.bind)
        else:
            query_out = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.date_obs >= time_min,
                                                                        EUV_Images.date_obs <= time_max,
                                                                        EUV_Images.instrument.in_(instrument),
                                                                        EUV_Images.wavelength.in_(wavelength)).statement,
                                    db_session.bind)

    return query_out


def add_image2session(data_dir, subdir, fname, db_session):
    """
    Adds a row to the database session that references the image location and metadata.
    The updated session will need to be committed - db_session.commit() - in order to
    write the new row to the DB.  See example101_DB-workflow.py
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
    existing_row_id = db_session.query(EUV_Images.image_id, EUV_Images.fname_raw).filter(
        EUV_Images.instrument == file_meta['instrument'],
        EUV_Images.date_obs == file_meta['datetime'],
        EUV_Images.wavelength == file_meta['wavelength']).all()
    if len(existing_row_id) == 1:
        if DB_path != existing_row_id[0][1]:
            # this is a problem.  We now have two different files for the same image
            sys.exit(("Current download: " + DB_path + " already exists in the database under a different file name:" +
                     existing_row_id[0][1]))
        else:
            # file has already been downloaded and entered into DB. do nothing
            print("File is already logged in database.  Nothing added.")
            pass
    elif len(existing_row_id) > 1:
        # This image already exists in the DB in MORE THAN ONE PLACE!
        sys.exit(("Current download: " + DB_path + " already exists in the database MULTIPLE times. " +
                 "Something is fundamentally wrong. DB unique index should " +
                 "prevent this from happening."))
    else:
        # Add new entry to DB
        # Construct now DB table row
        image_add = EUV_Images(date_obs=file_meta['datetime'], instrument=file_meta['instrument'],
                               wavelength=file_meta['wavelength'], fname_raw=DB_path,
                               fname_hdf="", distance=file_meta['distance'], cr_lon=file_meta['cr_lon'],
                               cr_lat=file_meta['cr_lat'], cr_rot=file_meta['cr_rot'],
                               time_of_download=datetime.datetime.now())
        # Append to the list of rows to be added
        db_session.add(image_add)
        print(("Database row added for " + file_meta['instrument'] + ", wavelength: " + str(file_meta['wavelength']) +
              ", timestamp: " + file_meta['date_string']))

    return db_session


def remove_euv_image(db_session, raw_series, raw_dir, hdf_dir):
    """
    Simultaneously delete image from filesystem and remove metadata row
    from EUV_Images table.
    raw_series - expects a pandas series that results from one row of
    the EUV_Images DB table.
    Ex.
    test_pd = query_euv_images(db_session=db_session, time_min=query_time_min,
                                time_max=query_time_max)
    remove_euv_image(raw_series=test_pd.iloc[0])

    ToDo:
        - Vectorize functionality.  If 'raw_series' is a pandas DataFrame
            delete all rows and their corresponding files.
    """

    raw_id = raw_series['image_id']
    raw_fname = raw_series['fname_raw']
    raw_full_path = os.path.join(raw_dir, raw_fname)
    hdf_fname = raw_series['fname_hdf']
    hdf_full_path = os.path.join(hdf_dir, hdf_fname)

    # check if file exists in filesystem
    if os.path.exists(raw_full_path):
        os.remove(raw_full_path)
        print("Deleted file: " + raw_full_path)
        exit_status = 0
    else:
        print("\nWarning: Image file not found at location: " + raw_full_path +
              ". This may be the symptom of a larger problem.")
        exit_status = 1

    # first check if there is an hdf file listed
    if hdf_fname!='':
        # check if file exists in filesystem
        if os.path.exists(hdf_full_path):
            os.remove(hdf_full_path)
            print("Deleted file: " + hdf_full_path)
        else:
            print("\nWarning: Processed HDF file not found at location: " + hdf_full_path +
                  ". This may be the symptom of a larger problem.")
            exit_status = exit_status + 2

    # delete row where id = raw_id.  Use .item() to recover an INT from numpy.int64
    out_flag = db_session.query(EUV_Images).filter(EUV_Images.image_id == raw_id.item()).delete()
    if out_flag==0:
        exit_status = exit_status + 4
    elif out_flag==1:
        db_session.commit()
        print("Row deleted from DB for image_id=" + str(raw_id))

    return exit_status, db_session


def update_image_val(db_session, raw_series, col_name, new_val):
    """
    Change value for EUV_Images in row referenced from raw_series and column referenced in col_name
    raw_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("image_id", "obs_time", "instrument", "wavelength"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions. Alternatively one could use remove_euv_image() followed " +
              "by euvi.download_image_fixed_format(), add_image2session(), and db_session.commit()")
    else:
        raw_id = raw_series['image_id']
        db_session.query(EUV_Images).filter(EUV_Images.image_id == raw_id).update({col_name : new_val})
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
                relative_path = root.replace(raw_data_dir+'/', "")
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


def add_euv_map(db_session, combo_id, meth_id, fname, var_dict=None, time_of_compute=None):
    """
    Simultaneously add record to EUV_Maps and Var_Vals.  Currently does not require
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
    existing_fname = pd.read_sql(db_session.query(EUV_Maps.map_id).filter(EUV_Maps.fname==fname).statement,
                                 db_session.bind)
    if len(existing_fname)>0:
        exit_status = 1
        map_id = None
    else:
        # check if meth_id/combo_id/var_vals already exist in DB?

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

            for index, var_row in var_info.iterrows():
                var_val = var_dict[var_row.var_name]
                add_var_val = Var_Vals(map_id=map_id, meth_id=meth_id, var_id=var_row.var_id, var_val=var_val)
                db_session.add(add_var_val)

        # now commit EUV_Map update and Var_Vals update simultaneously
        db_session.commit()

    return db_session, exit_status, map_id


# def add_map_image_assoc(db_session, combo_id, image_ids):
#
#     # add rows to MapImageAssoc table (session)
#     for image_id in image_ids:
#         assoc_add = Map_Image_Assoc(combo_id=combo_id, image_id=image_id)
#         # Append to the list of rows to be added
#         db_session.add(assoc_add)
#
#     return db_session


def update_euv_map(db_session, map_series, col_name, new_val):
    """
    Change value for EUV_Maps in row referenced from map_series and column referenced in col_name
    map_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("map_id", "obs_time", "instrument", "wavelength"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions. Alternatively one could use remove_euv_image() followed " +
              "by euvi.download_image_fixed_format(), add_image2session(), and db_session.commit()")
    else:
        raw_id = map_series['map_id']
        db_session.query(EUV_Images).filter(EUV_Images.image_id == raw_id).update({col_name : new_val})
        db_session.commit()

    return db_session


def get_combo_id(db_session, image_ids, create=False):
    """
    Function to query the database for an existing combination of images.  If it does not yet exist
    and create=True, the function will create a record for this combination.
    :param db_session: SQLAlchemy database session.
    :param image_ids: a tuple of image_id values that correspond to records in the euv_images table.
    :param create: boolean flag to indicate if new combinations should written to DB.
    :return: integer value of combo_id. If create=False and the entered images are a new combination, return -1.
    """

    # query DB to determine if this combo exists.
    n_images = len(image_ids)
    # Return the number of matching images in each combo that has n_images and contains at least one of image_ids.
    # This version uses actual SQL for the subquery
    # match_groups = pd.read_sql("""SELECT combo_id, COUNT(image_id) AS i_count FROM map_image_assoc WHERE
    #                                   combo_id IN (SELECT combo_id FROM image_combos WHERE n_images=
    #                                   """ + str(n_images) + ") AND image_id IN (" + str(image_ids) +
    #                                   " GROUP BY combo_id;", db_session.bind)
    # This version uses SQLAlchemy to re-create the SQL
    match_groups = pd.read_sql(
        db_session.query(Map_Image_Assoc.combo_id, func.count(Map_Image_Assoc.image_id).label("i_count")).
        filter(Map_Image_Assoc.combo_id.in_(
                    db_session.query(Image_Combos.combo_id).filter(Image_Combos.n_images==n_images)
                                            ), Map_Image_Assoc.image_id.in_(image_ids)
               ).group_by(Map_Image_Assoc.combo_id).statement, db_session.bind)

    # for testing only
    # match_groups = pd.DataFrame(data={'combo_id': [1, 2], 'i_count': [2, 3]})
    # match_groups = pd.DataFrame(columns = ['combo_id', 'i_count'])


    if len(match_groups)>0:
        # reduce match_groups to combos that match
        match_groups = match_groups.loc[match_groups.i_count==n_images]
        if len(match_groups)==1:
            # return the existing combo_id
            combo_id = match_groups.combo_id.values[0]
            no_match_exists = False
        else:
            no_match_exists = True
            combo_id = None

    else:
        no_match_exists = True
        combo_id = None

    if no_match_exists and create:
        # add record to image_combos
        # first retrieve records of images
        image_pd = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.image_id.in_(image_ids)).statement,
                               db_session.bind)
        # determine date range and mean
        date_max = image_pd.date_obs.max()
        date_min = image_pd.date_obs.min()
        date_mean = image_pd.date_obs.mean()
        # generate record and add to session
        combo_add = Image_Combos(n_images=n_images, date_mean=date_mean, date_max=date_max, date_min=date_min)
        db_session.add(combo_add)
        # Add record to DB
        db_session.commit()
        combo_id = combo_add.combo_id

    return db_session, combo_id


def add_combo_image_assoc(db_session, combo_id, image_id):
    """

    :param db_session:
    :param combo_id:
    :param image_id:
    :return:
    """

    # check if association already exists
    existing_assoc = pd.read_sql(db_session.query(Map_Image_Assoc).filter(Map_Image_Assoc.combo_id == combo_id,
                                                                         Map_Image_Assoc.image_id == image_id).statement,
                                 db_session.bind)

    # If association record does not exist, add it
    if len(existing_assoc.combo_id) == 0:
        assoc_add = Map_Image_Assoc(combo_id=combo_id, image_id=image_id)
        db_session.add(assoc_add)
        # commit changes to DB
        db_session.commit()
        exit_flag = 1
    else:
        exit_flag = 0

    return db_session, exit_flag


def get_method_id(db_session, meth_name, meth_desc=None, create=False):
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
    existing_meth = pd.read_sql(db_session.query(Meth_Defs.meth_id).filter(Meth_Defs.meth_name==meth_name).statement,
                                db_session.bind)

    if len(existing_meth.meth_id)==0:
        meth_exists = False
        meth_id = None
    else:
        meth_exists = True
        meth_id = existing_meth.meth_id[0]

    if create and not meth_exists:
        # create method record
        add_method = Meth_Defs(meth_name=meth_name, meth_description=meth_desc)
        db_session.add(add_method)
        # push change to DB and return meth_id
        db_session.commit()
        meth_id = add_method.meth_id

    return db_session, meth_id


def get_var_id(db_session, var_name, var_desc, create=False):
    """

    :param db_session:
    :param var_name:
    :param var_desc:
    :param create:
    :return:
    """

    # Query DB for existing variable
    existing_var = pd.read_sql(db_session.query(Var_Defs.var_id).filter(Var_Defs.var_name == var_name).statement,
                               db_session.bind)

    if len(existing_var.var_id) == 0:
        var_exists = False
        var_id = None
    else:
        var_exists = True
        var_id = existing_var.var_id[0]

    if create and not var_exists:
        # create variable record
        add_var = Var_Defs(var_name=var_name, var_description=var_desc)
        db_session.add(add_var)
        # push change to DB and return var_id
        db_session.commit()
        var_id = add_var.var_id

    return db_session, var_id


def add_meth_var_assoc(db_session, var_id, meth_id):
    """
    Check if meth/var association exists.  If not, add it.
    :param db_session: SQLAlchemy DB session
    :param var_id: variable ID
    :param meth_id: method ID
    :return: 0 - association already exists; 1 - association added
    """

    # Query DB for existing association
    existing_assoc = pd.read_sql(db_session.query(Meth_Var_Assoc).filter(Meth_Var_Assoc.var_id==var_id,
                                                                        Meth_Var_Assoc.meth_id==meth_id).statement,
                                 db_session.bind)
    # If association record does not exist, add it
    if len(existing_assoc.var_id) == 0:
        add_assoc = Meth_Var_Assoc(meth_id=meth_id, var_id=var_id)
        db_session.add(add_assoc)
        db_session.commit()
        exit_flag = 1
    else:
        exit_flag = 0

    return db_session, exit_flag


def query_euv_maps(db_session, mean_time_range=None, extrema_time_range=None, n_images=None, image_ids=None,
                   methods=None, var_val_range=None, wavelength=None):

    # combo query
    # n_images, image_ids, mean_time_range, extrema_time_range
    if mean_time_range is not None:
        # query combos by mean timestamp
        combo_query = db_session.query(Image_Combos.combo_id).filter(Image_Combos.date_mean.between(mean_time_range[0],
                                                                                                    mean_time_range[1]))
        if extrema_time_range is not None:
            # AND combo must contain an image in extrema_time_range
            combo_query = combo_query.filter_by(or_(Image_Combos.date_min.between(extrema_time_range[0],
                                                                                  extrema_time_range[1]),
                                                    Image_Combos.date_max.between(extrema_time_range[0],
                                                                                  extrema_time_range[1])))
    elif extrema_time_range is not None:
        # query combos with an image in extrema_time_range
        combo_query = db_session.query(Image_Combos.combo_id).filter_by(
            or_(Image_Combos.date_min.between(extrema_time_range[0], extrema_time_range[1]),
                Image_Combos.date_max.between(extrema_time_range[0], extrema_time_range[1])
                )
        )
    else:
        # this is a problem. mean_time_range or extrema_time_range must be defined
        sys.exit("query_euv_maps() requires that mean_time_range OR extrema_time_range be defined.")

    if n_images is not None:
        # AND combo must have Image_Combos.n_images IN(n_images)
        combo_query = combo_query.filter_by(Image_Combos.n_images.in_(n_images))

    if image_ids is not None:
        # AND filter by combos that contain image_ids
        combo_ids_query = db_session.query(Map_Image_Assoc.combo_id).filter(Map_Image_Assoc.image_id.in_(image_ids))
        combo_query = combo_query.filter_by(Image_Combos.combo_id.in_(combo_ids_query))

    if wavelength is not None:
        # AND combo contains images with wavelength in 'wavelength'
        image_ids_query = db_session.query(EUV_Images.image_id).filter(EUV_Images.wavelength.in_(wavelength))
        combo_ids_query = db_session.query(Map_Image_Assoc.combo_id).filter(Map_Image_Assoc.image_id.in_(
            image_ids_query))
        combo_query = combo_query.filter_by(Image_Combos.combo_id.in_(combo_ids_query))

    # start the master EUV_Map query that references combo, method, and var queries. Includes an
    # implicit join with image_combos
    euv_map_query = db_session.query(EUV_Maps, Image_Combos).filter(EUV_Maps.combo_id.in_(combo_query))

    # method query
    if methods is not None:
        # filter method_id by method names in 'methods'
        method_query = db_session.query(Meth_Defs.meth_id).filter(Meth_Defs.meth_name.in_(methods))
        # update master query
        euv_map_query = euv_map_query.filter_by(EUV_Maps.meth_id.in_(method_query))

    # variable value query
    if var_val_range is not None:
        # assume var_val_range is a dict of variable ranges with entries like: 'iter': [min, max]
        var_map_id_query = db_session.query(Var_Vals.map_id)
        # setup a query to find a list of maps in the specified variable ranges
        for var_name in var_val_range:
            var_id_query = db_session.query(Var_Defs.var_id).filter(Var_Defs.var_name==var_name)
            var_map_id_query = var_map_id_query.filter_by(Var_Vals.var_id==var_id_query,
                                                          Var_Vals.var_val.between(var_val_range[var_name]))
        # update master query
        euv_map_query = euv_map_query.filter_by(EUV_Maps.map_id.in_(var_map_id_query))

    # Need three output tables from SQL
        # 1. euv_maps joined with image_combos
        # 2. var_vals joined with meth_defs and var_defs
        # 3. euv_images joined with map_image_assoc

    # return map_info table using a join
    map_info = pd.read_sql(euv_map_query.statement, db_session.bind)
    # remove duplicate combo_id columns
    map_info = map_info.T.groupby(level=0).first().T

    # return image info. also keep image/combo associations for map-object building below
    image_assoc = pd.read_sql(db_session.query(Map_Image_Assoc).filter(Map_Image_Assoc.combo_id.in_(map_info.combo_id)
                                                                       ).statement, db_session.bind)
    image_info = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.image_id.in_(image_assoc.image_id)
                                                                 ).statement, db_session.bind)

    # return var_vals joined with var_defs. they are not directly connected tables, so use explicit join syntax
    var_query = db_session.query(Var_Vals, Var_Defs).join(Var_Defs, Var_Vals.var_id==Var_Defs.var_id
                                                ).filter(Var_Vals.map_id.in_(map_info.map_id))
    var_info = pd.read_sql(var_query.statement, db_session.bind)
    # remove duplicate var_id columns
    var_info = var_info.T.groupby(level=0).first().T

    # also get method info
    method_info = pd.read_sql(db_session.query(Meth_Defs).statement, db_session.bind)

    # This doesn't make sense.  We want a dataframe detailing each map record
    # Then divide results up into a list of Map objects
    # map_list = [datatypes.PsiMap()]*len(map_info)
    # for index, map_series in map_info.iterrows():
    #     # map_info
    #     map_list[index].append_map_info(map_series)
    #     # image_info
    #     combo_id = map_series.combo_id
    #     image_ids = image_assoc.image_id[image_assoc.combo_id==combo_id]
    #     images_slice = image_info.loc[image_info.image_id.isin(image_ids)]
    #     map_list[index].append_image_info(images_slice)
    #     # var_info
    #     var_slice = var_info.loc[var_info.map_id==map_series.map_id]
    #     map_list[index].append_var_info(var_slice)
    #     # method info
    #     map_list[index].append_method_info(method_info.loc[method_info.meth_id==map_series.meth_id])
    #
    # return map_list

    return map_info, image_info, var_info, method_info


def create_map_input_object(new_map, fname, image_df, var_vals, method_name, time_of_compute=None):
    """
    This function generates a Map object and populates it with necessary info for creating a
    new map record in the DB.
    :param new_map: PsiMap object that has not been added to DB yet
    :param fname: relative file path, including filename
    :param time_of_compute: Datetime when the map was made
    :param image_df: DataFrame with columns 'image_id' describing the constituent images
    :param var_vals: DataFrame with columns 'var_name' and 'var_val' describing variable values
    :param method_name: Name of method used.  Must match meth_defs.meth_name column in DB
    :return: The partially filled Map object needed to create a new map record in the DB
    """

    # construct method_info df
    method_info = pd.DataFrame(data=[[method_name, ]], columns=["meth_name", ])
    new_map.append_method_info(method_info)

    # add var_vals
    # Expects the input 'var_vals' to be a pandas DataFrame.  Requires fields 'var_name' and 'var_val'.
    new_map.append_var_info(var_vals)

    # add image_info
    # Expects the input 'image_df' to be a pandas DataFrame.  The only required column is 'image_id' and
    # should contain all the image_ids of this map's constituent images.  Can also be a complete euv_images-
    # structured DF.
    new_map.append_image_info(image_df)

    # construct map_info df
    map_info_df = pd.DataFrame(data=[[len(image_df), fname, time_of_compute], ],
                                     columns=["n_images", "fname", "time_of_compute"])
    new_map.append_map_info(map_info_df)

    return new_map


def create_method(db_session, meth_name, meth_desc, meth_vars, var_descs):
    """
    Create a new method record in the DB
    :param db_session: SQLAlchemy session
    :param meth_name: string scalar
    :param meth_desc: string scalar
    :param meth_vars: list of strings
    :param var_descs: list of strings
    :return: db_session, meth_id
    """

    # First create needed variables; Or return var_ids if they already exist for another method
    var_ids = [None]*len(meth_vars)
    for index, var in enumerate(meth_vars, start=0):
        db_session, var_ids[index] = get_var_id(db_session=db_session, var_name=var, var_desc=var_descs[index],
                                                create=True)
    # Then create a method; Or return meth_id if it already exists
    db_session, meth_id = get_method_id(db_session=db_session, meth_name=meth_name, meth_desc=meth_desc, create=True)
    # Now make method-variable associations
    for var_id in var_ids:
        db_session, exit_flag = add_meth_var_assoc(db_session=db_session, var_id=var_id, meth_id=meth_id)

    return db_session, meth_id


def add_map_dbase_record(db_session, psi_map):

    # extract method_id
    if psi_map.map_info.meth_id is None:
        # if method_id is not passed in map object, query method id from existing methods table
        db_session, meth_id = get_method_id(db_session=db_session, meth_name=psi_map.method_info.meth_name[0],
                                        create=False)
    else:
        meth_id = psi_map.map_info.meth_id[0]

    # Get combo_id. Create if it doesn't already exist.
    image_ids = psi_map.image_info.image_id.to_list()
    db_session, combo_id = get_combo_id(db_session=db_session, image_ids=image_ids, create=True)
    # add combo-image associations
    for image in image_ids:
        db_session, exit_flag = add_combo_image_assoc(db_session=db_session, combo_id=combo_id, image_id=image)

    # Add EUV_map record
    var_dict = dict(zip(psi_map.var_info.var_name, psi_map.var_info.var_val))
    # When writing to SQL, SQLAlchemy wants native python datatypes
    time_of_compute = psi_map.map_info.time_of_compute[0]
    if type(time_of_compute).__module__=='pandas._libs.tslibs.timestamps':
        time_of_compute = time_of_compute.to_pydatetime()
    if type(meth_id).__module__=='numpy':
        meth_id = meth_id.item()
    db_session, exit_status, map_id = add_euv_map(db_session=db_session, combo_id=combo_id, meth_id=meth_id,
                                                  fname=psi_map.map_info.fname[0], var_dict=var_dict,
                                                  time_of_compute=time_of_compute)

    return db_session, map_id


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
    out_flag = db_session.query(Var_Vals).filter(Var_Vals.map_id == map_id).delete()
    if out_flag==0:
        exit_status = exit_status + 2
    else:
        db_session.commit()
        print(str(out_flag) + " row(s) deleted from 'var_vals' for map_id=" + str(map_id))
    # delete map record
    out_flag = db_session.query(EUV_Maps).filter(EUV_Maps.map_id == map_id).delete()
    if out_flag==0:
        exit_status = exit_status + 4
    else:
        db_session.commit()
        print(str(out_flag) + " row(s) deleted from 'euv_maps' for map_id=" + str(map_id))
    exit_status = 0

    return exit_status


def write_map_object(map_in, filename=None, db_session=None):
    """

    :param map_in: should be a PsiMap object with full image and method/variable info
    :param filename: if 'None' will use standardized file naming convention to name the file.
    :param db_session: if 'None' will not attempt to write an entry to the database.
    :return: a map object that includes filename and database ids where appropriate.
    """
