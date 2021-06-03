"""
Generate some test-case EUV_map info and push to the DB
"""


import os
import datetime
# import pandas as pd

from chmap.settings.app import App
from chmap.database.db_classes import *
from chmap.database.deprecated.db_funs import init_db_conn, get_var_id, get_method_id, add_meth_var_assoc, get_combo_id, \
    add_combo_image_assoc, add_euv_map, build_euvimages_from_fits


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

# delete old sqlite file
if os.path.exists(sqlite_filename):
    os.remove(sqlite_path)

# re-initialize database file and establish a connection/session
db_session = init_db_conn(db_name=use_db, chd_base=Base, sqlite_path=sqlite_path)

# re-build euv_images table
db_session = build_euvimages_from_fits(db_session=db_session, raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir)


# Generate several sets of euv map info
map1 = {'images': [1,2,3], 'method': 'meth_101', 'fname': "/test1.h5", 'time_of_compute': datetime.datetime(2019, 9, 19,
                                                                                                            12, 00, 1),
        'var_vals': {'iters': 120, 'x1': 3.7, 'x2': 151}}
map2 = {'images': [4,5,6], 'method': 'meth_101', 'fname': "/test2.h5", 'time_of_compute': datetime.datetime(2019, 9, 19,
                                                                                                            12, 00, 2),
        'var_vals': {'iters': 127, 'x1': 3.7, 'x2': 151}}
map3 = {'images': [7,8,9], 'method': 'meth_101', 'fname': "/test3.h5", 'time_of_compute': datetime.datetime(2019, 9, 19,
                                                                                                            12, 00, 3),
        'var_vals': {'iters': 12, 'x1': 3.7}}
map_list = [map1, map2, map3]


# Generate method info
meth_name = "meth_101"
meth_desc = "Although Method 101 is quite impressive, this entry is for testing purposes only."
meth_vars = ("iters", "x1", "x2")
var_descs = ("Number of iterations", "The renowned variable X the 1st", "Reserve Grand Champion X")


# First create needed variables
var_ids = [None] * len(meth_vars)
for index, var in enumerate(meth_vars, start=0):
    db_session, var_ids[index] = get_var_id(db_session=db_session, var_name=var, var_desc=var_descs[index], create=True)
# Then create a method
db_session, meth_id = get_method_id(db_session=db_session, meth_name=meth_name, meth_desc=meth_desc, create=True)
# Now make method-variable associations
for var_id in var_ids:
    db_session, exit_flag = add_meth_var_assoc(db_session=db_session, var_id=var_id, meth_id=meth_id)

for map_info in map_list:
    # Get combo_id. Create if it doesn't already exist.
    image_ids = map_info['images']
    db_session, combo_id = get_combo_id(db_session=db_session, data_ids=image_ids, create=True)
    # add combo-image associations
    for image in image_ids:
        db_session, exit_flag = add_combo_image_assoc(db_session=db_session, combo_id=combo_id, data_id=image)

    # Add EUV_map record
    db_session, exit_status, map_id = add_euv_map(db_session=db_session, combo_id=combo_id, meth_id=meth_id,
                fname=map_info['fname'], var_dict=map_info['var_vals'], time_of_compute=map_info['time_of_compute'])



db_session.close()
