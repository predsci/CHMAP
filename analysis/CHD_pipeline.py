"""
This pipeline first saves individual image maps to the database
- this is an issue because of storage space
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
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings.app import App
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs
from modules.map_manip import combine_maps
import modules.Plotting as EasyPlot

#### ------------ QUERYING PARAMETERS TO UPDATE ------------- #####

# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 10, 0, 0, 0)
map_freq = 2  # number of hours... rename

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
    # maybe this needs to be made a function?
# methods_template is a combination of Meth_Defs and Var_Defs columns
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

for inst_index, instrument in enumerate(inst_list):
    print("Starting corrections for", instrument, "images.\n")
    # query correct image combos
    lbc_combo_query = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=7),
                                                query_time_max + datetime.timedelta(days=7),
                                                meth_name='LBCC Theoretic', instrument=instrument)
    iit_combo_query = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=7),
                                                query_time_max + datetime.timedelta(days=7), meth_name='IIT',
                                                instrument=instrument)

    # create dataframe for instrument
    hist_inst = query_pd['instrument']
    instrument_pd = query_pd[hist_inst == instrument]

    for index, row in instrument_pd.iterrows():
        print("Processing image number", row.image_id, "for LBC and IIT Corrections.")
        # apply LBC
        original_los, lbcc_image, mu_indices, use_indices = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                 lbc_combo_query, image_row=row,
                                                                                 n_intensity_bins=n_intensity_bins,
                                                                                 R0=R0)
        # apply IIT
        lbcc_image, iit_image, use_indices = iit_funcs.apply_iit(db_session, hdf_data_dir, iit_combo_query,
                                                                 lbcc_image, use_indices, image_row=row, R0=R0)

        # Store Coronal Hole Map with data map? or as separate map-object?
        #chd_list = [None] * len(los_list)
        #for ii in range(len(los_list)):
            # call function to ezseg los_list[ii]
            # chd_list[ii] = ezseg_wrapper(los_list[ii])
         #   pass

        # use fixed map resolution
        map_image = iit_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
        # Alternatively, we could have resolution determined from image
        # map_list[ii] = los_list[ii].interp_to_map(R0=R0)
        # record image info
        map_image.append_image_info(row)

        # generate a record of the method and variable values used for interpolation
        new_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':
            ["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                      'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
        # add to the methods dataframe for this map
        methods_list[index] = methods_list[index].append(pd.DataFrame(data=new_method), sort=False)

        # incorporate the methods dataframe into the map object
        map_image.append_method_info(methods_list[index])
        # save these maps to file and then push to the database
        map_image.write_to_file(map_data_dir, map_type='single', filename=None, db_session=db_session)

### CREATE COMBO MAPS
map_frequency = int((query_time_max - query_time_min).seconds / 3600 / map_freq)
moving_avg_centers = np.array(
    [np.datetime64(str(query_time_min)) + ii * np.timedelta64(map_freq, 'h') for ii in range(map_frequency+1)])


for index, center in enumerate(moving_avg_centers):
    date_time = np.datetime64(center).astype(datetime.datetime)
    map_info, image_info, method_info, map_list = db_funcs.query_euv_maps(db_session, mean_time_range=[date_time - datetime.timedelta(hours=1),
                                                               date_time + datetime.timedelta(hours=1)], n_images=1)
    if len(map_list) == 0:
        continue
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
