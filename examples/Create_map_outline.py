
"""
Working sketch of how we will combine some images into a map:
    1. Select images
    2. Limb-brightening correction
    3. Inter-instrument Transformation
    4. Coronal Hole Detection
    5. Convert to Map
    6. Combine Maps
    7. Save to DB
"""


import os
import datetime
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import scipy.interpolate as sp_interp
# import time
# import sunpy

from settings.app import App
import modules.DB_classes as db_class
from modules.DB_funs import init_db_conn, query_euv_images
import modules.datatypes as psi_d_types
from modules.map_manip import combine_maps
# import modules.coord_manip as coord
import modules.Plotting as EasyPlot

# --- 1. Select Images -----------------------------------------------------
# In this example we use the 'reference_data' fits files supplied with repo
# manually set the data-file dirs
raw_data_dir = os.path.join(App.APP_HOME, "reference_data", "raw")
hdf_data_dir = os.path.join(App.APP_HOME, "reference_data", "processed")
# manually set the database location
database_dir = os.path.join(App.APP_HOME, "reference_data")
sqlite_filename = "dbtest.db"

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)

db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)


# query some images
query_time_min = datetime.datetime(2014, 4, 13, 19, 35, 0)
query_time_max = datetime.datetime(2014, 4, 13, 19, 37, 0)
query_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

print(query_pd.instrument)
# use these three images (one from each instrument)
selected_images = query_pd

# read hdf file(s) to a list of LOS objects
los_list = [None]*selected_images.__len__()
image_plot_list = [None] * selected_images.__len__()
for index, row in selected_images.iterrows():
    hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
    los_list[index] = psi_d_types.read_los_image(hdf_path)
    # Plot for verification
    # image_plot_list[index] = EasyPlot.PlotImage(los_list[index], nfig=index, title="Image " + str(index))
    EasyPlot.PlotImage(los_list[index], nfig=index, title="Image " + str(index))


# --- 2. Limb-brightening correction ------------------------------------------

for ii in range(len(los_list)):
    # call function to de-limb los_list[ii]
    # los_list[ii] = limb_correct(los_list[ii])
    pass


# --- 3. Inter-instrument Transformation --------------------------------------

for ii in range(len(los_list)):
    # call function to correct los_list[ii]
    # los_list[ii] = inst_correct(los_list[ii])
    pass

# --- 4. Coronal Hole Detection -----------------------------------------------
# Store Coronal Hole Map with data map? or as separate map-object?
chd_list = [None]*len(los_list)
for ii in range(len(los_list)):
    # call function to ezseg los_list[ii]
    # chd_list[ii] = ezseg_wrapper(los_list[ii])
    pass

# --- 5. Convert to Map -------------------------------------------------------
# map parameter definitions.
R0 = 1.01
y_range = [-1, 1]
x_range = [0, 2*np.pi]
# The default behavior of los_image.interp_to_map() is to determine map vertical
# resolution from image vertical R0-resolution.  Here we want the maps that we
# intend to combine to have a fixed resolution.
map_nycoord = 1600
del_y = (y_range[1] - y_range[0])/(map_nycoord - 1)
map_nxcoord = (np.floor((x_range[1] - x_range[0])/del_y) + 1).astype(int)

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')

map_list = [None]*len(los_list)
for ii in range(len(los_list)):
    # use fixed map resolution
    map_list[ii] = los_list[ii].interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=selected_images.image_id[ii])
    # Alternatively, we could have resolution determined from image
    # map_list[ii] = los_list[ii].interp_to_map(R0=R0)
    EasyPlot.PlotMap(map_list[ii], nfig=10+ii, title="Map " + str(ii))

# --- 6. Combine Maps ---------------------------------------------------------
combined_map = combine_maps(map_list, del_mu=0.2)

EasyPlot.PlotMap(combined_map, nfig=20, title="Minimum Intensity Merge Map")
plt.show()

# --- 7. Save to DB -----------------------------------------------------------
# # add image info to map object
#
# # fname = gen_map_fname()
# fname = "/test/fname1.h5"
#
# time_of_compute = datetime.datetime.now()
# meth_name = "meth_101"
# # method is new to the DB. Add method definition before adding map
# new_method = True
# # in practice image_df will usually be the output of query_euv_images(), but here
# # we show that only the image_id column is needed for map record creation
# image_df = selected_images
# # variable values must be a DataFrame with columns var_name and var_val. These should
# # collected over steps 2-6 as necessary.
# var_vals = pd.DataFrame(data=[['x1', 1], ['x2', 10.1]], columns=["var_name", "var_val"])
#
# # --- generate a Map object ----------
# map_input = create_map_input_object(fname=fname, image_df=image_df, var_vals=var_vals, method_name=meth_name,
#                                     time_of_compute=time_of_compute)
#
# map_input.x = combined_map.x
# map_input.y = combined_map.y
# map_input.data = combined_map.data
# # send data to the DB
# db_session, map_id = add_map_dbase_record(db_session, psi_map=map_input)



