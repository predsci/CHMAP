"""
Trim first and last 3 months (90 days) of LBCCs
Also trim Stereo A 2015 LBCCS
"""

import os
import datetime
from sqlalchemy.sql import func

from settings.app import App
import chmap.database.db_classes as db_class
from chmap.database.db_funs import init_db_conn, get_method_id, query_inst_combo


###### ------ PARAMETERS TO UPDATE -------- ########

# TIME RANGE
stereoA_trim_min = datetime.datetime(2014, 9, 1, 0, 0, 0)
stereoA_trim_max = datetime.datetime(2015, 12, 1, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# define window to remove at begining and end
trim_window = datetime.timedelta(days=90)

# recover local filesystem paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME


# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.
# setup local database paths (only used for use_db='sqlite')
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# setup database connection
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print("\nPrevious file ", sqlite_filename, " deleted.\n")

    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# query LBC method id
meth_name = 'LBCC'
meth_desc = 'LBCC Theoretic Fit Method'
method_id = get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

for instrument in inst_list:
    # determine minimum image date
    min_date_query = db_session.query(func.min(db_class.EUV_Images.date_obs)).filter(
        db_class.EUV_Images.instrument == instrument).all()
    # extract datetime from weird sqlalchemy construct
    min_date = min_date_query[0][0]
    # set lower trim max date
    lower_trim_max = min_date + trim_window

    # determine minimum image date
    max_date_query = db_session.query(func.max(db_class.EUV_Images.date_obs)).filter(
        db_class.EUV_Images.instrument == instrument).all()
    # extract datetime from weird sqlalchemy construct
    max_date = max_date_query[0][0]
    # set lower trim max date
    upper_trim_min = max_date - trim_window

    # delete LBCCs before lower_trim_max ------------
    del_combos = query_inst_combo(db_session, query_time_min=datetime.datetime(2000, 1, 1, 0, 0, 0),
                                  query_time_max=lower_trim_max, meth_name=meth_name, instrument=instrument)
    if del_combos.shape[0] > 0:
        print("Trimming early LBCC values for ", instrument)
        # First delete the variable values
        nvals = db_session.query(db_class.Var_Vals).filter(
            db_class.Var_Vals.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)
        # Then delete from image_combo_assoc
        nassoc = db_session.query(db_class.Data_Combo_Assoc).filter(
            db_class.Data_Combo_Assoc.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)
        # Last delete from image_combos
        ncombos = db_session.query(db_class.Data_Combos).filter(
            db_class.Data_Combos.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)

        # commit changes to DB
        db_session.commit()
        print("Data_Combos deleted from DB: ", ncombos)

    # delete LBCCs after upper_trim_min ------------
    del_combos = query_inst_combo(db_session, query_time_min=upper_trim_min,
                                  query_time_max=datetime.datetime(2100, 1, 1, 0, 0, 0), meth_name=meth_name,
                                  instrument=instrument)
    if del_combos.shape[0] > 0:
        print("Trimming late LBCC values for ", instrument)
        # First delete the variable values
        nvals = db_session.query(db_class.Var_Vals).filter(
            db_class.Var_Vals.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)
        # Then delete from image_combo_assoc
        nassoc = db_session.query(db_class.Data_Combo_Assoc).filter(
            db_class.Data_Combo_Assoc.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)
        # Last delete from image_combos
        ncombos = db_session.query(db_class.Data_Combos).filter(
            db_class.Data_Combos.combo_id.in_(del_combos.combo_id.to_list())
        ).delete(synchronize_session=False)

        # commit changes to DB
        db_session.commit()
        print("Data_Combos deleted from DB: ", ncombos)

# delete LBCCs in Stereo A gap ------------
instrument = "EUVI-A"
del_combos = query_inst_combo(db_session, query_time_min=stereoA_trim_min,
                              query_time_max=stereoA_trim_max, meth_name=meth_name, instrument=instrument)
if del_combos.shape[0] > 0:
    print("Trimming 2015 LBCC values for ", instrument)
    # First delete the variable values
    nvals = db_session.query(db_class.Var_Vals).filter(
        db_class.Var_Vals.combo_id.in_(del_combos.combo_id.to_list())
    ).delete(synchronize_session=False)
    # Then delete from image_combo_assoc
    nassoc = db_session.query(db_class.Data_Combo_Assoc).filter(
        db_class.Data_Combo_Assoc.combo_id.in_(del_combos.combo_id.to_list())
    ).delete(synchronize_session=False)
    # Last delete from image_combos
    ncombos = db_session.query(db_class.Data_Combos).filter(
        db_class.Data_Combos.combo_id.in_(del_combos.combo_id.to_list())
    ).delete(synchronize_session=False)

    # commit changes to DB
    db_session.commit()
    print("Data_Combos deleted from DB: ", ncombos)
