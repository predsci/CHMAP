
"""
Code for testing and setting up DB changes to include magnetic maps
"""

import os
import datetime
from settings.app import App
import database.db_funs as db_funcs
import database.db_classes as db_class
import modules.datatypes as psi_d_types

####### -------- updateable parameters ------ #######

# TIME RANGE FOR FIT PARAMETER CALCULATION
calc_query_time_min = datetime.datetime(2011, 10, 1, 0, 0, 0)
calc_query_time_max = datetime.datetime(2012, 4, 1, 0, 0, 0)

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q_test" # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


test = db_funcs.query_euv_images(db_session)


# read a psi_map file
test_map = psi_d_types.read_psi_map(
    "/Users/turtle/GitReps/CHD/test_data/maps/single/2014/04/13/single_20140413T193530_MID4.h5")
test_map.data_info.keys()

# test new image write process
# Add new entry to data_files
data_file_add = Data_Files(date_obs=datetime.datetime(2000, 1, 1), provider="abc",
                           type="abc", fname_raw="abc/", fname_hdf="")
db_session.add(data_file_add)
# get data_id
db_session.flush()
new_data_id = data_file_add.data_id
# be sure not to commit these changes
db_session.rollback()

db_session.close()
