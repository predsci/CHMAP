
"""
Assuming we have a directory of fits files, this example walks through the files and
re-creates a database from scratch.
This example uses the fits files included in reference_data/
"""


import os

from chmap.settings.app import App
from chmap.database.db_classes import *
from chmap.database.db_funs import init_db_conn_old, build_euvimages_from_fits, query_euv_images

# In this example we can use the 'reference_data' fits files supplied with repo or the directories setup in App.py
# data-file dirs
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
# database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = App.DATABASE_FNAME

# To recreate the SQLite database file in reference_data/, use these:
# raw_data_dir = os.path.join(App.APP_HOME, "reference_data", "raw")
# hdf_data_dir = os.path.join(App.APP_HOME, "reference_data", "processed")
# database_dir = os.path.join(App.APP_HOME, "reference_data")
# sqlite_filename = "dbtest.db"

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "tervin"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print("\nPrevious file ", sqlite_filename, " deleted.\n")

    db_session = init_db_conn_old(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=Base, user=user, password=password)


# build database
print("\nNow build database image records for each existing file:")
db_session = build_euvimages_from_fits(db_session=db_session, raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir)

print("\nProcess complete.")

if use_db == 'sqlite':
    # recover all image records and print
    print("\nCheck that 'euv_images' table contains records:")
    test_pd = query_euv_images(db_session=db_session)
    print(test_pd)
elif use_db == 'mysql-Q':
    # recover all image records and print the count
    test_pd = query_euv_images(db_session=db_session)
    num_rows = test_pd.shape[0]
    print("\nTotal number of images in database: " + str(num_rows) + "\n")

db_session.close()
