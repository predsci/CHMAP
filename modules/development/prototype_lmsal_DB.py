

import os

from settings.app import App
import modules.DB_classes as DBClass
from modules.DB_funs import init_db_conn


# data-file dirs
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
# database location
database_dir = App.DATABASE_HOME
# give the sqlite file a unique name
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q_test" # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)







