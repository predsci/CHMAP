"""
This example demonstrates the process of adding a map to the database.
    1. Create a test SQLite database
    2. create minimal 'Map' object from method, var_vals, image associations, and map filename
    3. if new_method: add method definition to DB
    4. pass 'Map' to map adding function
    5. demonstrate map query
    6. demonstrate map record deletion
"""


import datetime
import pandas as pd
import os
import numpy as np

from settings.app import App
from modules import DB_classes
from modules.deprecated.DB_funs import create_map_input_object, init_db_conn, create_method, add_map_dbase_record,\
    query_euv_maps, delete_map_dbase_record
from modules.datatypes import PsiMap


# --- 1. Create a test SQLite database ----------------
# Assume that we are using images from the 'reference_data' setup supplied with repo
# manually set the data dir
raw_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'raw')
hdf_data_dir = os.path.join(App.APP_HOME, 'reference_data', 'processed')
# manually set the database location using the installed app settings.
database_dir = os.path.join(App.APP_HOME, 'test_data')

sqlite_filename = "db_map_test.db"
sqlite_path = os.path.join(database_dir, sqlite_filename)
# delete old sqlite file
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)
    print("Previous file ", sqlite_filename, " deleted.")

use_db = "sqlite"
db_session = init_db_conn(db_name=use_db, chd_base=DB_classes.Base, sqlite_path=sqlite_path)

# re-build euv_images table
# print("\nGenerating images database from existing fits files. \n")
# db_session = build_euvimages_from_fits(db_session=db_session, raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir)


# --- 2. create minimal 'Map' object from method, var_vals, image associations, and map filename -------------
fname = "/test/fname1.h5"
time_of_compute = datetime.datetime(2000, 1, 1, 1, 1, 1)
meth_name = "meth_101"
# method is new to the DB. Add method definition before adding map
new_method = True
# in practice image_df will usually be the output of query_euv_images(), but here
# we show that only the image_id column is needed for map record creation
image_df = pd.DataFrame(data=[1, 2, 3], columns=["image_id", ])
# variable values must be a DataFrame with columns var_name and var_val
var_vals = pd.DataFrame(data=[['x1', 1], ['x2', 10.1]], columns=["var_name", "var_val"])

# --- generate a Map object ----------
# example grid
x = np.array(range(10))
y = x
data = np.full((10, 10), 1.)
new_map = PsiMap(data=data, x=x, y=y, mu=None, origin_image=None, no_data_val=-9999.0)
map_input = create_map_input_object(new_map=new_map, fname=fname, image_df=image_df, var_vals=var_vals,
                                    method_name=meth_name, time_of_compute=time_of_compute)


# --- 3. if new_method: add method definition to DB -----
if new_method:
    print("\nWriting records to define the new method 'meth_101'.")
    # method 'meth_101' needs a description, list of associated variables and descriptions for the variables
    meth_desc = "Although Method 101 is quite impressive, this entry is for testing purposes only."
    meth_vars = ("iters", "x1", "x2")
    var_descs = ("Number of iterations", "The renowned variable X the 1st", "Reserve Grand Champion X")
    # this function creates the method definition, variable associations, and variable definitions (as needed)
    db_session, meth_id = create_method(db_session=db_session, meth_name=meth_name, meth_desc=meth_desc,
                                        meth_vars=meth_vars, var_descs=var_descs)
    # update map_input with method id
    map_input.map_info.meth_id = meth_id


# --- 4. pass 'Map' to map adding function ----------------
db_session, map_id = add_map_dbase_record(db_session, psi_map=map_input)
print("\nNew map record made with map_id=" + str(map_id), ".\n")


# --- 5. demonstrate map query ----------------------------
mean_time_range = [datetime.datetime(2000, 1, 1, 1, 1, 1), datetime.datetime(2020, 1, 1, 1, 1, 1)]
# For example search for all maps over a large time range.  For demonstration, all query options are listed
# query_par=None.  In practice, they default to None and do not need to be entered in the function call.  Ranges
# are expected as a list with length 2; others expect lists and use the IN() operator.
map_info, image_info, var_info, method_info = query_euv_maps(db_session, mean_time_range=mean_time_range,
                                                             extrema_time_range=None, n_images=None,
                                                             image_ids=None, methods=None, var_val_range=None,
                                                             wavelength=None)
# print("Querying for all maps in a large time range: \n" + str(mean_time_range) + ", \nReturns a list of maps of length "
#       + str(len(map_list)) + ".\n")

# --- 6. demonstrate map record deletion ------------------
map_dir = os.path.join(App.APP_HOME, 'reference_data', 'map')
exit_flag = delete_map_dbase_record(db_session, map_object=map_list[0], data_dir=map_dir)
# because there is not an actual file "/test/fname1.h5", this function will generate a warning

db_session.close()
