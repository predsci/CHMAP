"""
Apply LBC to images based on theoretic fit.
Get images and parameers from database to apply correction.
"""

import os
import numpy as np
import datetime

from settings.app import App
from modules.DB_funs import init_db_conn, query_euv_images, query_lbcc_fit, query_var_val
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting

# define time range to query
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 1, 6, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

# define map and binning parameters
R0 = 1.01
mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# define directory paths
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME

##### INITIALIZE DATABASE CONNECTION #####

use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

##### QUERY IMAGES ######

for inst_index, instrument in enumerate(inst_list):
    # query wants a list
    query_instrument = [instrument, ]
    image_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                                instrument=query_instrument)

###### GET LOS IMAGES COORDINATES (DATA) #####
    for index, row in image_pd.iterrows():
        print("Processing image number", row.image_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        los_temp = psi_d_types.read_los_image(hdf_path)
        los_temp.get_coordinates(R0=R0)

    ###### APPLY LBC CORRECTION #######

    # select image
    selected_image = image_pd.iloc[0]
    # read hdf file to LOS object
    hdf_file = os.path.join(hdf_data_dir, selected_image.fname_hdf)
    original_los = psi_d_types.read_los_image(hdf_file)
    original_los.get_coordinates(R0=R0)

    mu_array = original_los.mu
    beta_query, y_query = query_lbcc_fit(db_session, image = selected_image, meth_name = "LBCC Theoretic")
    beta_array = np.zeros((len(mu_array), len(mu_array)))
    y_array = np.zeros((len(mu_array), len(mu_array)))

    beta_buffer = np.frombuffer(beta_query)
    beta = np.ndarray(shape=(len(mu_array), len(mu_array)), buffer=beta_buffer)
    y_buffer = np.frombuffer(y_query)
    y = np.ndarray(shape=(len(mu_array), len(mu_array)), buffer=y_buffer)
    # apply correction
    corrected_los_data = beta * original_los.data + y

###### PLOT ORIGINAL IMAGE ######
    Plotting.PlotImage(original_los, nfig = 30 + inst_index, title = "Original LOS Image for " + instrument)


##### PLOT CORRECTED IMAGE ######
    Plotting.PlotLBCCImage(lbcc_data = corrected_los_data, los_image = original_los, nfig = 40 + inst_index, title = "Corrected LBCC Image for " + instrument)




