"""
outline to create combination EUV maps
- this method doesn't automatically save individual image maps to database, bc storage
- currently does not include CH Detection
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps
6. Save to DB
"""

sys.path.append("CHD")
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
from modules.map_manip import combine_maps
import modules.datatypes as datatypes
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs

import os
import time
import pandas as pd
import numpy as np
import datetime
from settings.app import App
import modules.Plotting as EasyPlot
import matplotlib.pyplot as plt

# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 4, 12, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 12, 12, 0, 0)
map_freq = 2  # number of hours... rename
save_single = False  # if you want to save single image maps to database (storage issue)

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
map_data_dir = App.MAP_FILE_HOME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
# initialize database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = db_funcs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01
del_mu = 0.2

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
map_nycoord = 1600
del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord, dtype='<f4')
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord, dtype='<f4')


# --- 1. Select Images -----------------------------------------------------
# query some images
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

# generate a dataframe to record methods
meth_columns = []
for column in db_class.Meth_Defs.__table__.columns:
    meth_columns.append(column.key)
defs_columns = []
for column in db_class.Var_Defs.__table__.columns:
    defs_columns.append(column.key)
df_cols = set().union(meth_columns, defs_columns, ("var_val", ))
methods_template = pd.DataFrame(data=None, columns=df_cols)
# generate a list of methods dataframes
methods_list = [methods_template] * query_pd.__len__()


# --- 2. Apply pre-processing corrections ------------------------------------------
# get dates
map_frequency = int((query_time_max - query_time_min).seconds / 3600 / map_freq)
moving_avg_centers = np.array(
    [np.datetime64(str(query_time_min)) + ii * np.timedelta64(map_freq, 'h') for ii in range(map_frequency+1)])

# query for combo ids within date range

lbc_combo_query = []
iit_combo_query = []
for inst_index, instrument in enumerate(inst_list):
    start = time.time()
    lbc_combo = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=7),
                                            query_time_max + datetime.timedelta(days=7),
                                            meth_name='LBCC Theoretic', instrument=instrument)
    iit_combo = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=7),
                                            query_time_max + datetime.timedelta(days=7), meth_name='IIT',
                                            instrument=instrument)
    end = time.time()
    print(end-start, "seconds for combo queries")
    lbc_combo_query.append(lbc_combo)
    iit_combo_query.append(iit_combo)


for date_ind, center in enumerate(moving_avg_centers):
    print("Starting corrections for", center, "images.\n")
    date_time = np.datetime64(center).astype(datetime.datetime)
    # create dataframe for date
    hist_date = query_pd['date_obs']
    date_pd = query_pd[
              (hist_date >= np.datetime64(date_time - datetime.timedelta(hours=map_freq/2))) &
              (hist_date <= np.datetime64(date_time + datetime.timedelta(hours=map_freq/2)))]
    if len(date_pd) == 0:
        print("No Images to Process for this date.")
        continue
    # create map list
    map_list = [datatypes.PsiMap()] * len(inst_list)
    image_info = []
    map_info = []
    for inst_ind, instrument in enumerate(inst_list):
        # query correct image combos
        hist_inst = date_pd['instrument']
        image_pd = date_pd[hist_inst == instrument]

        for image_ind, row in image_pd.iterrows():
            print("Processing image number", row.image_id, "for LBC and IIT Corrections.")
            # apply LBC
            original_los, lbcc_image, mu_indices, use_indices = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                     lbc_combo_query[inst_ind], image_row=row,
                                                                                     n_intensity_bins=n_intensity_bins,
                                                                                     R0=R0)
            # apply IIT
            lbcc_image, iit_image, use_indices = iit_funcs.apply_iit(db_session, hdf_data_dir, iit_combo_query[inst_ind],
                                                                     lbcc_image, use_indices, image_row=row, R0=R0)

            # --- 3. Coronal Hole Detection ------------------------------------------
            # TODO: implement CHD Fortran script

            # use fixed map resolution
            map_list[inst_ind] = iit_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
            # record image and map info
            map_list[inst_ind].append_image_info(row)
            image_info.append(row)
            map_info.append(map_list[inst_ind].map_info)

            # generate a record of the method and variable values used for interpolation
            new_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
                ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                          'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
            # add to the methods dataframe for this map
            methods_list[inst_ind] = methods_list[inst_ind].append(pd.DataFrame(data=new_method), sort=False)

            # incorporate the methods dataframe into the map object
            map_list[inst_ind].append_method_info(methods_list[inst_ind])
            # save these maps to file and then push to the database
            if save_single:
                map_list[inst_ind].write_to_file(map_data_dir, map_type='single', filename=None, db_session=db_session)

    # --- 5. Combine Maps -----------------------------------
    combined_map = combine_maps(map_list, del_mu=del_mu)

    # generate a record of the method and variable values used for interpolation
    new_method = {'meth_name': ("Min-Int-Merge_1",), 'meth_description':
        ["Minimum intensity merge version 1"] * 1,
                  'var_name': ("del_mu",), 'var_description': ("max acceptable mu range",), 'var_val': (del_mu,)}
    combined_map.append_method_info(pd.DataFrame(data=new_method))
    combined_map.append_image_info(image_info)
    combined_map.append_map_info(map_info)

    EasyPlot.PlotMap(combined_map, nfig="Combined map for: " + str(center), title="Minimum Intensity Merge Map\nDate: "
                                                                               + str(center))
    plt.show()

    combined_map.write_to_file(map_data_dir, map_type='synoptic', filename=None, db_session=db_session)
