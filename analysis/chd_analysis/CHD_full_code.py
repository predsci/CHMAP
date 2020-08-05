"""
outline to create combination EUV maps
- this method doesn't automatically save individual image maps to database, bc storage
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps
6. Save to DB
"""

import os
import time
import pandas as pd
import numpy as np
import datetime

from settings.app import App
import modules.Plotting as EasyPlot
import ezseg.ezsegwrapper as ezsegwrapper
import modules.DB_classes as db_class
import modules.DB_funs as db_funcs
from modules.map_manip import combine_maps
import modules.datatypes as datatypes
import analysis.lbcc_analysis.LBCC_theoretic_funcs as lbcc_funcs
import analysis.iit_analysis.IIT_pipeline_funcs as iit_funcs
import anaylsis.chd_analysis.CHD_pipeline_funcs as chd_funcs


# -------- parameters --------- #
# TIME RANGE FOR QUERYING
query_time_min = datetime.datetime(2011, 4, 12, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 12, 3, 0, 0)
map_freq = 2  # number of hours
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
mu_cutoff = 0.0  # not current used, lower mu cutoff value

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1 = 0.95
thresh2 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

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
methods_list = db_funcs.generate_methdf(query_pd)


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
    lbc_combo = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=180),
                                            query_time_max + datetime.timedelta(days=180),
                                            meth_name='LBCC Theoretic', instrument=instrument)
    iit_combo = db_funcs.query_inst_combo(db_session, query_time_min - datetime.timedelta(days=180),
                                            query_time_max + datetime.timedelta(days=180), meth_name='IIT',
                                            instrument=instrument)
    end = time.time()
    print(end-start, "seconds for combo queries")
    lbc_combo_query.append(lbc_combo)
    iit_combo_query.append(iit_combo)


for date_ind, center in enumerate(moving_avg_centers):
    print("\nStarting corrections for", center, "images:")
    date_time = np.datetime64(center).astype(datetime.datetime)
    # create dataframe for date
    hist_date = query_pd['date_obs']
    date_pd = query_pd[
              (hist_date >= np.datetime64(date_time - datetime.timedelta(hours=map_freq/2))) &
              (hist_date <= np.datetime64(date_time + datetime.timedelta(hours=map_freq/2)))]
    if len(date_pd) == 0:
        print("No Images to Process for this date.")
        continue
    # alpha, x for threshold
    sta_ind = inst_list.index('EUVI-A')
    alpha, x = db_funcs.query_var_val(db_session, meth_name='IIT', date_obs=date_time,
                                      inst_combo_query=iit_combo_query[sta_ind])
    # create map list
    map_list = [datatypes.PsiMap()] * len(inst_list)
    chd_map_list = [datatypes.PsiMap()] * len(inst_list)
    chd_list = [None] * len(inst_list)
    image_info = []
    map_info = []
    for inst_ind, instrument in enumerate(inst_list):
        # query correct image combos
        hist_inst = date_pd['instrument']
        image_pd = date_pd[hist_inst == instrument]
        for image_ind, row in image_pd.iterrows():
            print("Processing image number", row.image_id, "for LBC and IIT Corrections.")
            # apply LBC
            start=time.time()
            original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session, hdf_data_dir,
                                                                                     lbc_combo_query[inst_ind], image_row=row,
                                                                                     n_intensity_bins=n_intensity_bins,
                                                                                     R0=R0)
            # apply IIT
            lbcc_image, iit_image, use_indices, alpha, x  = iit_funcs.apply_iit(db_session, iit_combo_query[inst_ind],
                                                                     lbcc_image, use_indices, original_los, R0=R0)
            end=time.time()
            # print("IPP time:", end-start, "seconds.")

            ### CH DETECTION
            start=time.time()
            image_data = iit_image.iit_data
            use_chd = use_indices.astype(int)
            use_chd = np.where(use_chd == 1, use_chd, -9999)
            nx = iit_image.x.size
            ny = iit_image.y.size
            t1 = thresh1*alpha + x
            t2 = thresh2*alpha + x
            ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, t1, t2, nc, iters)
            chd_list[inst_ind] = np.logical_and(ezseg_output==0, use_chd==1)
            chd_list[inst_ind] = chd_list[inst_ind].astype(int)
            chd_image = datatypes.create_chd_image(original_los, chd_list[inst_ind])
            chd_image.get_coordinates()
            end=time.time()
            # print("CHD TIME:", end-start, 'seconds.')

            # create single maps
            # CHD map
            chd_map_list[inst_ind] = chd_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
            # map of IIT image
            map_list[inst_ind] = iit_image.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=row.image_id)
            # record image and map info
            chd_map_list[inst_ind].append_image_info(row)
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
                EasyPlot.PlotMap(map_list[inst_ind], nfig=10 + inst_ind, title="Map " + instrument)
                EasyPlot.PlotMap(chd_map_list[inst_ind], nfig=100 + inst_ind, title="CHD Map " + instrument,
                                 map_type='CHD')
                map_list[inst_ind].write_to_file(map_data_dir, map_type='single', filename=None, db_session=db_session)

    # --- 5. Combine Maps -----------------------------------
    euv_combined, chd_combined = combine_maps(map_list, chd_map_list, del_mu=del_mu)
    # record merge parameters
    combined_method = {'meth_name': ("Min-Int-Merge_1", "Min-Int-Merge_1"), 'meth_description':
        ["Minimum intensity merge version 1"] * 2,
                       'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value",
                                                                                "max acceptable mu range"),
                       'var_val': (mu_cutoff, del_mu)}
    # generate a record of the method and variable values used for interpolation
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    euv_combined.append_image_info(image_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_image_info(image_info)
    chd_combined.append_map_info(map_info)

    # plot maps
    EasyPlot.PlotMap(euv_combined, nfig="EUV Combined map for: " + str(euv_combined.image_info.date_obs[0]),
                     title="Minimum Intensity Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
    EasyPlot.PlotMap(chd_combined, nfig="CHD Combined map for: " + str(chd_combined.image_info.date_obs[0]),
                     title="CHD Merge Map\nDate: " + str(chd_combined.image_info.date_obs[0]), map_type='CHD')
    EasyPlot.PlotMap(euv_combined, nfig="EUV/CHD Combined map for: " + str(euv_combined.image_info.date_obs[0]),
                     title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
    EasyPlot.PlotMap(chd_combined, nfig="EUV/CHD Combined map for: " + str(chd_combined.image_info.date_obs[0]),
                     title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(chd_combined.image_info.date_obs[0]),
                     map_type='CHD')

    # save EUV and CHD maps to database
    #euv_combined.write_to_file(map_data_dir, map_type='synoptic_euv', filename=None, db_session=db_session)
    #chd_combined.write_to_file(map_data_dir, map_type='synoptic_chd', filename=None, db_session=db_session)

