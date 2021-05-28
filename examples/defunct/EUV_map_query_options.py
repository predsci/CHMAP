

import os
import datetime
import pandas as pd

from settings.app import App
from database.db_classes import *
from database.deprecated.db_funs import init_db_conn
from sqlalchemy.orm import joinedload


# Assume that we are using images from the 'reference_data' setup supplied with repo
# manually set the data dir
raw_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'raw')
hdf_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'processed')
# manually set the database location using the installed app settings.
database_dir = os.path.join(App.APP_HOME, 'reference_data')


# setup database path
use_db = "sqlite"
sqlite_filename = "dbtest.db"
sqlite_path = os.path.join(database_dir, sqlite_filename)
# re-initialize database file and establish a connection/session
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# define query time range
query_time_min = datetime.datetime(2000, 1, 1, 1, 0, 0)
query_time_max = datetime.datetime(2019, 1, 1, 1, 0, 0)
mean_time_range = [query_time_min, query_time_max]

combo_query = db_session.query(Data_Combos.combo_id).filter(Data_Combos.date_mean.between(mean_time_range[0],
                                                                                            mean_time_range[1]))
euv_map_query = db_session.query(EUV_Maps).filter(EUV_Maps.combo_id.in_(combo_query))
euv_map_query_join = db_session.query(EUV_Maps, Data_Combos).filter(EUV_Maps.combo_id.in_(combo_query))
euv_map_rel_join = db_session.query(EUV_Maps).options(joinedload(EUV_Maps.combos)).filter(EUV_Maps.combo_id.in_(combo_query))
test = euv_map_query.all()

len(test)

# lazyload combo info
test[0].combos.combo_id
# lazyload which images the combo is made up of
for row in test[0].combos.images:
    print(row.data_id)


test2 = pd.read_sql(euv_map_query.statement, db_session.bind)
test2_2 = pd.read_sql(euv_map_query_join.statement, db_session.bind)
test2_3 = pd.read_sql(euv_map_rel_join.statement, db_session.bind)

db_session.close()
