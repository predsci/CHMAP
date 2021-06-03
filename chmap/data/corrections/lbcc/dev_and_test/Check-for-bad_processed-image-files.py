
# A few images from 2020/07/30 did not process correctly.
# remove the files and clear their hdf_filename in the DB

import os
import datetime

from chmap.settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funs

# In this example we can use the 'reference_data' fits files supplied with repo or the directories setup in App.py
# data-file dirs
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
# database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = App.DATABASE_FNAME


# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


# first image (AIA)
query_min = datetime.datetime(2020, 7, 30, 15, 0, 0)
query_max = datetime.datetime(2020, 7, 30, 17, 0, 0)
instrument = ["AIA", ]

image_pd = db_funs.query_euv_images(db_session, time_min=query_min, time_max=query_max,
                                    instrument=instrument)

delete_path = os.path.join(hdf_data_dir, image_pd.fname_hdf.iloc[0])
os.remove(delete_path)

db_session = db_funs.update_image_val(db_session, image_pd.iloc[0], "fname_hdf", "")


# other two images (EUVI-A)
query_min = datetime.datetime(2020, 7, 30, 13, 0, 0)
query_max = datetime.datetime(2020, 7, 30, 17, 0, 0)
instrument = ["EUVI-A", ]

image_pd = db_funs.query_euv_images(db_session, time_min=query_min, time_max=query_max,
                                    instrument=instrument)

for index, row in image_pd.iterrows():
    delete_path = os.path.join(hdf_data_dir, row.fname_hdf)
    os.remove(delete_path)

    db_session = db_funs.update_image_val(db_session, row, "fname_hdf", "")


db_session.close()