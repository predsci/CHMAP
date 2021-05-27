"""
Apply LBC to images based on theoretic fit.
Get images and parameters from database to apply correction.
"""

import os
import time
import numpy as np
import datetime
from settings.app import App
from database.db_funs import init_db_conn, query_euv_images, query_var_val, query_inst_combo
import database.db_classes as db_class
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting
import modules.lbcc_utils as lbcc

# define time range to query
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 1, 6, 0, 0)
plot = True  # plot images
n_images_plot = 1  # number of images to plot

# define instruments
inst_list = ['AIA', "EUVI-A", "EUVI-B"]

# define map and binning parameters
R0 = 1.01
n_intensity_bins = 200

# define directory paths
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME

##### INITIALIZE DATABASE CONNECTION #####
# use_db = "sqlite"
# sqlite_path = os.path.join(database_dir, sqlite_filename)
use_db = "mysql-Q"
user = "tervin"
password = ""

##### -------- APPLY LBC FIT ------- ######
# start time
start_time_tot = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)

# method information
meth_name = "LBCC"

##### QUERY IMAGES ######
for inst_index, instrument in enumerate(inst_list):

    query_instrument = [instrument, ]
    image_pd = query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                time_max=lbc_query_time_max, instrument=query_instrument)
    # query correct image combos
    combo_query = query_inst_combo(db_session, lbc_query_time_min - datetime.timedelta(weeks = 2),
                                   lbc_query_time_max + datetime.timedelta(weeks = 2), meth_name, instrument)

    ###### GET LOS IMAGES COORDINATES (DATA) #####
    # apply LBC
    for index in range(n_images_plot):
        row = image_pd.iloc[index]
        print("Processing image number", row.data_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.data_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        original_los = psi_d_types.read_los_image(hdf_path)
        original_los.get_coordinates(R0=R0)
        theoretic_query = query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                        inst_combo_query=combo_query)

        ###### DETERMINE LBC CORRECTION (for valid mu values) ######
        beta1d, y1d, mu_indices, use_indices = lbcc.get_beta_y_theoretic_continuous_1d_indices(theoretic_query,
                                                                                               los_image=original_los)

        ###### APPLY LBC CORRECTION (log10 space) ######
        corrected_lbc_data = np.copy(original_los.data)
        corrected_lbc_data[use_indices] = 10 ** (beta1d * np.log10(original_los.data[use_indices]) + y1d)

        ##### PLOTTING ######
        if plot:
            Plotting.PlotImage(original_los, nfig=100 + inst_index * 10 + index, title="Original LOS Image for " +
                                                                                       instrument)
            Plotting.PlotCorrectedImage(corrected_data=corrected_lbc_data, los_image=original_los,
                                        nfig=200 + inst_index * 10 + index, title="Corrected LBCC Image for " +
                                                                                  instrument)
            Plotting.PlotCorrectedImage(corrected_data=original_los.data - corrected_lbc_data, los_image=original_los,
                                        nfig=300 + inst_index * 10 + index, title="Difference Plot for " + instrument)
# end time
end_time_tot = time.time()
print("LBC has been applied and specified images plotted.")
print("Total elapsed time to apply correction and plot: " + str(round(end_time_tot - start_time_tot, 3))
      + " seconds.")
