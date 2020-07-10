"""
Apply LBC to images based on theoretic fit.
Get images and parameters from database to apply correction.
"""

import os
import time
import numpy as np
import datetime

from settings.app import App
from modules.DB_funs import init_db_conn, query_euv_images, query_var_val, get_method_id
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting
import modules.lbcc_funs as lbcc

# define time range to query
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)
plot = True  # plot images

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
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

##### -------- APPLY LBC FIT ------- ######
# start time
start_time_tot = time.time()

meth_name = "LBCC Theoretic"
db_sesh, meth_id, var_ids = get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                          var_descs=None, create=False)
intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

##### QUERY IMAGES ######
for inst_index, instrument in enumerate(inst_list):

    query_instrument = [instrument, ]
    image_pd = query_euv_images(db_session=db_session, time_min=query_time_min,
                                time_max=query_time_max, instrument=query_instrument)

    ###### GET LOS IMAGES COORDINATES (DATA) #####
    for index, row in image_pd.iterrows():
        print("Processing image number", row.image_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        original_los = psi_d_types.read_los_image(hdf_path)
        original_los.get_coordinates(R0=R0)
        theoretic_query = query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                        instrument=instrument)

        beta, y = lbcc.get_beta_y_theoretic_continuous(theoretic_query, mu_array=original_los.mu)

        ###### APPLY LBC CORRECTION ######
        corrected_los_data = beta * original_los.data + y

        ##### PLOTTING ######
        if plot:
            Plotting.PlotImage(original_los, nfig=100 + inst_index, title="Original LOS Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data=corrected_los_data, los_image=original_los, nfig=200 + inst_index,
                                   title="Corrected LBCC Image for " + instrument)
            Plotting.PlotLBCCImage(lbcc_data=original_los.data - corrected_los_data, los_image=original_los,
                                   nfig=300 + inst_index, title="Difference Plot for " + instrument)
# end time
end_time_tot = time.time()
print("LBC has been applied and specified images plotted.")
print("Total elapsed time to apply correction and plot: " + str(round(end_time_tot - start_time_tot, 3))
      + " seconds.")