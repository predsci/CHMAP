
# go through all combos and assign a value to the 'instrument' column

import os
import pandas as pd

from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_classes as db_class


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

# query all combos
image_combos = pd.read_sql(db_session.query(db_class.Image_Combos).statement, db_session.bind)

total_combos = image_combos.shape[0]

for index, row in image_combos.iterrows():
    print("Updating", index, "of", total_combos, "combos.")
    # join all images/instruments to this combo
    image_inst_join = db_session.query(db_class.Image_Combo_Assoc, db_class.EUV_Images.instrument).join(
        db_class.EUV_Images).filter(
        db_class.Image_Combo_Assoc.combo_id == row.combo_id)
    join_query = pd.read_sql(image_inst_join.statement, db_session.bind)

    all_instruments = join_query.instrument.unique()
    if all_instruments.__len__() == 1:
        # set this instrument value in 'instrument' column
        set_val = all_instruments[0]
    elif all_instruments.__len__() > 1:
        set_val = "MULTIPLE"
    else:
        set_val = None

    if set_val is not None:
        # write to database
        db_session.query(db_class.Image_Combos).filter(
            db_class.Image_Combos.combo_id == row.combo_id).update(
            {db_class.Image_Combos.instrument: set_val}, synchronize_session=False
        )
        # commit change
        db_session.commit()


db_session.close()


