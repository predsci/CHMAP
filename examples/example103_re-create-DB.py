
"""
Assuming we have a directory of fits files, this example walks through the files and
re-creates a database from scratch.
This example uses the fits files included in reference_data/
"""


import os

from settings.app import App
from modules.DB_classes import *
from modules.DB_funs import init_db_conn, build_euvimages_from_fits, query_euv_images

# In this example we use the 'reference_data' fits files supplied with repo
# manually set the data-file dirs
raw_data_dir = os.path.join(App.APP_HOME, "reference_data", "raw")
hdf_data_dir = os.path.join(App.APP_HOME, "reference_data", "processed")
# manually set the database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = "db_create-test.db"

# To recreate the SQLite database file in reference_data/, use these:
# database_dir = os.path.join(App.APP_HOME, "reference_data")
# sqlite_filename = "dbtest.db"

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)

if os.path.exists(sqlite_path):
    os.remove(sqlite_path)
    print("\nPrevious file ", sqlite_filename, " deleted.\n")

db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# build database
print("\nNow build database image records for each existing file:")
db_session = build_euvimages_from_fits(db_session=db_session, raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir)

# recover entire DB and print
print("\nCheck that 'euv_images' table contains records:")
test_pd = query_euv_images(db_session=db_session)
print(test_pd)

db_session.close()
