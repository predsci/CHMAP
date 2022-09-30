
# Query and Delete selected processed images.
# Leave the corresponding raw images in place.


import os
import datetime
import pandas as pd

import chmap.database.db_classes as DBClass
import chmap.database.db_funs as db_funs


# test setting umask for writing to Q over AFP
os.umask(0o002)

# Paths to the database filesystem
raw_data_home = '/Volumes/extdata2/CHD_DB/raw_images'
processed_data_home = '/Volumes/extdata2/CHD_DB/processed_images'

# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db
# 'mysql-Q' Use the remote MySQL database on Q
user = "cdowns"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# setup database connection to MySQL database on Q
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

query_start = datetime.datetime(2020, 7, 30, 0)
query_end = datetime.datetime(2020, 7, 31, 0)
query1 = db_funs.query_euv_images(db_session=db_session, time_min=query_start, time_max=query_end)

query_start = datetime.datetime(2020, 12, 27, 0)
query_end = datetime.datetime(2020, 12, 28, 0)
query2 = db_funs.query_euv_images(db_session=db_session, time_min=query_start, time_max=query_end)

del_pd = pd.concat((query1, query2))

for index, row in del_pd.iterrows():
    exit_flag = db_funs.remove_euv_image(db_session, row, raw_data_home,
                                         processed_data_home, proc_only=True)


db_session.close()
