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

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from settings.app import App
from modules.DB_classes import *
from modules.misc_funs import get_metadata


def init_DB_conn(db_name, chd_base, sqlite_fn=""):
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
        dbfile = os.path.join(App.DATABASE_HOME, sqlite_fn)
        connect_string = 'sqlite:///' + dbfile
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


def query_euv_images(db_session, time_min, time_max, instrument=None, wavelength=None):
    """User-friendly query function.  Allows user to avoid using SQLAlchemy or SQL
    syntax.
    time_min, time_max - datetime objects that define the time interval to search.
    instrument - list of spacecraft to include (characters)
    wavelength - list of wavelengths to include (integer)
    db_session - database sqlalchemy session object
    """

    if wavelength is None:
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
    fits_path = os.path.join(data_dir, subdir, fname)
    fits_map = sunpy.map.Map(fits_path)
    file_meta = get_metadata(fits_map)

    # check if row already exists in DB
    existing_row_id = db_session.query(EUV_Images.id, EUV_Images.fname_raw).filter(
        EUV_Images.instrument == file_meta['instrument'],
        EUV_Images.date_obs == file_meta['datetime'],
        EUV_Images.wavelength == file_meta['wavelength']).all()
    if len(existing_row_id) == 1:
        if fits_path != existing_row_id[0][1]:
            # this is a problem.  We now have two different files for the same image
            sys.exit(("Current download: " + fits_path + " already exists in the database under a different file name:" +
                     existing_row_id[0][1]))
        else:
            # file has already been downloaded and entered into DB. do nothing
            print("File is already logged in database.  Nothing added.")
            pass
    elif len(existing_row_id) > 1:
        # This image already exists in the DB in MORE THAN ONE PLACE!
        sys.exit(("Current download: " + fits_path + " already exists in the database MULTIPLE times. " +
                 "Something is fundamentally wrong. DB unique index should " +
                 "prevent this from happening."))
    else:
        # Add new entry to DB
        # Construct now DB table row
        image_add = EUV_Images(date_obs=file_meta['datetime'], jd=file_meta['jd'], instrument=file_meta['instrument'],
                               wavelength=file_meta['wavelength'], fname_raw=fits_path,
                               fname_hdf="", distance=file_meta['distance'], cr_lon=file_meta['cr_lon'],
                               cr_lat=file_meta['cr_lat'], cr_rot=file_meta['cr_rot'],
                               time_of_download=datetime.datetime.now())
        # Append to the list of rows to be added
        db_session.add(image_add)
        print(("Database row added for " + file_meta['instrument'] + ", wavelength: " + str(file_meta['wavelength']) +
              ", timestamp: " + file_meta['date_string']))

    return db_session


def remove_euv_image(db_session, raw_series):
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
    hdf_fname = raw_series['fname_hdf']

    # check if file exists in filesystem
    if os.path.exists(raw_fname):
        os.remove(raw_fname)
        print("Deleted file: " + raw_fname)
        exit_status = 0
    else:
        print("\nWarning: Image file not found at location: " + raw_fname +
              ". This may be the symptom of a larger problem.")
        exit_status = 1

    # first check if there is an hdf file listed
    if hdf_fname!='':
        # check if file exists in filesystem
        if os.path.exists(hdf_fname):
            os.remove(raw_fname)
            print("Deleted file: " + raw_fname)
        else:
            print("\nWarning: Processed HDF file not found at location: " + raw_fname +
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
        db_session.query(EUV_Images).filter(EUV_Images.id==raw_id.item()).update({col_name : new_val})
        db_session.commit()

    return(db_session)
