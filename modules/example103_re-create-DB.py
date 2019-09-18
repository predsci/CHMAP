
"""
Assuming we have a directory of fits files, this example walks through the files and
re-creates a database from scratch.
Additional Assumptions:
  - hdf_data_dir mirrors raw_data_dir.  If a matching hdf file exists, it is
      located in a position that mirrors the fits file.
  - sqlite_filename does not currently exist.
"""


import os

from settings.app import App
from modules.DB_classes import *
from modules.DB_funs import init_db_conn, build_euvimages_from_fits, query_euv_images

# In this example we use the 'reference_data' fits files supplied with repo
# manually set the data-file dirs
raw_data_dir = os.path.join(App.APP_HOME, "reference_data/raw")
hdf_data_dir = os.path.join(App.APP_HOME, "reference_data/processed")
# manually set the database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = "db_create-test.db"

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# build database
db_session = build_euvimages_from_fits(db_session=db_session, raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir)

# recover entire DB and print
test_pd = query_euv_images(db_session=db_session)
print(test_pd)

db_session.close()
