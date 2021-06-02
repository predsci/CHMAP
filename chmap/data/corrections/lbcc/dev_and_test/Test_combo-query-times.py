
"""
Check the scaling rate of LBCC queries
"""

import os
import time
import datetime
from sqlalchemy.sql import func

from settings.app import App
from chmap.database.db_funs import init_db_conn
import chmap.database.db_funs as db_funcs
import chmap.database.db_classes as db_class

###### ------ UPDATEABLE PARAMETERS ------- #######
# TIME RANGE FOR LBC CORRECTION AND HISTOGRAM CREATION
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2012, 10, 1, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database parameters
create = True  # true if you want to add to database
# designate which database to connect to

# mysql
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

###### ------- NOTHING TO UPDATE BELOW ------- #######

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


instrument = inst_list[1]
center_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
time_window = datetime.timedelta(weeks=1)
min_time = center_time - time_window/2
max_time = center_time + time_window/2

start_time = time.time()
combo_query = db_funcs.query_inst_combo(db_session, min_time, max_time,
                                        meth_name="LBCC", instrument=instrument)
query_time = time.time() - start_time
print("LBCC query time for 1-week: ", query_time)


center_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
time_window = datetime.timedelta(weeks=2)
min_time = center_time - time_window/2
max_time = center_time + time_window/2

start_time = time.time()
combo_query = db_funcs.query_inst_combo(db_session, min_time, max_time,
                                        meth_name="LBCC", instrument=instrument)
query_time = time.time() - start_time
print("LBCC query time for 2-week: ", query_time)


center_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
time_window = datetime.timedelta(weeks=4)
min_time = center_time - time_window/2
max_time = center_time + time_window/2

start_time = time.time()
combo_query = db_funcs.query_inst_combo(db_session, min_time, max_time,
                                        meth_name="LBCC", instrument=instrument)
query_time = time.time() - start_time
print("LBCC query time for 4-week: ", query_time)


center_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
time_window = datetime.timedelta(weeks=8)
min_time = center_time - time_window/2
max_time = center_time + time_window/2

start_time = time.time()
combo_query = db_funcs.query_inst_combo(db_session, min_time, max_time,
                                        meth_name="LBCC", instrument=instrument)
query_time = time.time() - start_time
print("LBCC query time for 8-week: ", query_time)


center_time = datetime.datetime(2011, 1, 1, 0, 0, 0)
time_window = datetime.timedelta(weeks=64)
min_time = center_time - time_window/2
max_time = center_time + time_window/2

start_time = time.time()
combo_query = db_funcs.query_inst_combo(db_session, min_time, max_time,
                                        meth_name="LBCC", instrument=instrument)
query_time = time.time() - start_time
print("LBCC query time for 64-week: ", query_time)

db_session.close()

image_min_query = db_session.query(db_class.EUV_Images).filter(
    db_class.EUV_Images.instrument == instrument,
    db_class.EUV_Images.date_obs.in_(db_session.query(func.min(db_class.EUV_Images.date_obs)).filter(
        db_class.EUV_Images.instrument == instrument
    ))
).all()

image_max_query = db_session.query(db_class.EUV_Images).filter(
    db_class.EUV_Images.instrument == instrument,
    db_class.EUV_Images.date_obs.in_(db_session.query(func.max(db_class.EUV_Images.date_obs)).filter(
        db_class.EUV_Images.instrument == instrument
    ))
).all()

combo_min_query = db_session.query(db_class.Data_Combos).filter(
    db_class.Data_Combos.combo_id == db_class.Data_Combo_Assoc.combo_id,
    db_class.EUV_Images.date_obs
)


