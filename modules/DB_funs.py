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

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from settings.app import App
from modules.DB_classes import *
from modules.misc_funs import get_metadata


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
    existing_row_id = db_session.query(EUV_Images.id, EUV_Images.fname_raw).filter(
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
        image_add = EUV_Images(date_obs=file_meta['datetime'], jd=file_meta['jd'], instrument=file_meta['instrument'],
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

    raw_id = raw_series['id']
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
    out_flag = db_session.query(EUV_Images).filter(EUV_Images.id==raw_id.item()).delete()
    if out_flag==0:
        exit_status = exit_status + 4
    elif out_flag==1:
        db_session.commit()
        print("Row deleted from DB for id=" + str(raw_id))

    return exit_status, db_session


def update_image_val(db_session, raw_series, col_name, new_val):
    """
    Change value for EUV_Images in row referenced from raw_series and column referenced in col_name
    raw_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("id", "obs_time", "instrument", "wavelength"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions. Alternatively one could use remove_euv_image() followed " +
              "by euvi.download_image_fixed_format(), add_image2session(), and db_session.commit()")
    else:
        raw_id = raw_series['id']
        db_session.query(EUV_Images).filter(EUV_Images.id==raw_id).update({col_name : new_val})
        db_session.commit()

    return(db_session)


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
        # check for matching hdf5 file
        hdf_full_path = os.path.join(hdf_data_dir, row.fname_raw)
        hdf_full_path = hdf_full_path.replace(".fits", ".hdf5")
        if os.path.exists(hdf_full_path):
            hdf_rel_path = hdf_full_path.replace(hdf_data_dir + "/", "")
            # assume the file is good (??) and record in DB
            db_session = update_image_val(db_session=db_session, raw_series=row,
                                          col_name="fname_hdf", new_val=hdf_rel_path)

    return(db_session)


def add_euv_map(db_session, map_fname):

    # check if row or filename already exists in DB

    # create EUV_Maps object and add to session

    # create MapImageAssoc row(s) and add to session add_map_image_assoc()

    return(db_session)


def add_map_image_assoc(db_session, combo_id, image_ids):

    # add rows to MapImageAssoc table (session)
    for image_id in image_ids:
        assoc_add = Map_Image_Assoc(combo_id=combo_id, image_id=image_id)
        # Append to the list of rows to be added
        db_session.add(assoc_add)

    return(db_session)


def update_euv_map(db_session, map_series, col_name, new_val):
    """
    Change value for EUV_Maps in row referenced from map_series and column referenced in col_name
    map_series - pandas series for the row to be updated
    :return: the SQLAlchemy database session
    """

    if col_name in ("id", "obs_time", "instrument", "wavelength"):
        print("This is a restricted column and will not be updated by this function. Values can be changed " +
              "directly using SQLAlchemy functions. Alternatively one could use remove_euv_image() followed " +
              "by euvi.download_image_fixed_format(), add_image2session(), and db_session.commit()")
    else:
        raw_id = map_series['id']
        db_session.query(EUV_Images).filter(EUV_Images.id==raw_id).update({col_name : new_val})
        db_session.commit()

    return(db_session)


def get_combo_id(db_session, images, create=False):
    """
    Function to query the database for an existing combination of images.  If it does not yet exist
    and create=True, the function will create a record for this combination.
    :param db_session: SQLAlchemy database session.
    :param images: a tuple of image_id values that correspond to records in the euv_images table.
    :param create: boolean flag to indicate if new combinations should written to DB.
    :return: integer value of combo_id. If create=False and the entered images are a new combination, return -1.
    """

    # query DB to determine if this combo exists.
    n_images = len(images)
    # returns the number of matching images in each combo that contains at least one matching image.
    # This version uses actual SQL for the subquery
    # match_groups = pd.read_sql("""SELECT combo_id, COUNT(image_id) AS i_count FROM map_image_assoc WHERE
    #                                   combo_id IN (SELECT combo_id FROM image_combos WHERE n_images=
    #                                   """ + str(n_images) + ") GROUP BY combo_id;", db_session.bind)
    # This version uses SQLAlchemy to re-create the SQL
    match_groups = pd.read_sql(
        db_session.query(Map_Image_Assoc.combo_id, func.count(Map_Image_Assoc.image_id).label("i_count")).\
        filter(Map_Image_Assoc.combo_id==\
                    db_session.query(Image_Combos.combo_id).filter(Image_Combos.n_images==n_images)
               ).group_by(Map_Image_Assoc.combo_id).statement,
    db_session.bind)

    # for testing only
    # match_groups = pd.DataFrame(data={'combo_id': [1, 2], 'i_count': [2, 3]})
    # match_groups = pd.DataFrame(columns = ['combo_id', 'i_count'])

    no_match_exists=False
    if len(match_groups)>0:
        # reduce match_groups to combos that match
        match_groups = match_groups.loc[match_groups.i_count==n_images]
        if len(match_groups)==1:
            # return the existing combo_id
            combo_id = match_groups.combo_id.values[0]
        else:
            no_match_exists=True
            combo_id = None

    else:
        no_match_exists=True
        combo_id = None

    # if no_match_exists and create:
    #     # first add record to image_combos
    #     # calc min, max, obs, jd times
    #     combo_add = Image_Combos(n_images=n_images, date_obs=, date_max=, date_min=, jd=)
    #     db_session.add(combo_add)
    #     # sync session and DB 'flush()' which also returns the new combo_id
    #     db_session.flush()
    #     combo_id = combo_add.combo_id
    #     # then add combo-image associations to map_image_assoc
    #     for image in images:
    #         assoc_add = Map_Image_Assoc(combo_id=combo_id, image_id=image)
    #         db_session.add(assoc_add)
    #     # commit changes to DB
    #     db_session.commit()


    return(combo_id)

