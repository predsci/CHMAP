"""
pipeline for inter-instrument transformation
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from settings.app import App
from modules.DB_funs import init_db_conn
import modules.DB_classes as db_class
import modules.iit_funs as iit
import modules.DB_funs as db_funcs
import modules.datatypes as psi_d_types
import modules.lbcc_funs as lbcc
import analysis.iit_analysis.iit_pipeline_funcs as iit_funcs

##### PARAMETERS TO UPDATE #####

# TIME RANGE FOR LBC CORRECTION AND IMAGE PLOTTING
lbc_query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)
# define instruments
inst_list = ["EUVI-A"]
# declare map and binning parameters
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [-np.pi / 2.4, np.pi / 2.4]

# define database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
create = True  # true if you want to add to database
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

##### PRE STEP: APPLY LBC TO IMAGES ######
# going to have to move the plotting step into here probably
meth_name = "LBCC Theoretic"
mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
mu_bin_centers = (mu_bin_edges[1:] + mu_bin_edges[:-1]) / 2

##### QUERY IMAGES ######
for inst_index, instrument in enumerate(inst_list):

    query_instrument = [instrument, ]
    image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                         time_max=lbc_query_time_max, instrument=query_instrument)
    # TODO: these 2048 numbers may need to be adjusted
    # TODO: should be original_los.mu however that is in the loop - deal with this later
    corrected_lbcc_data = np.ndarray(shape=(len(inst_list), len(image_pd), 2048, 2048))
    corrected_los_data = np.zeros((len(image_pd), 2048, 2048))

    ###### GET LOS IMAGES COORDINATES (DATA) #####
    for index, row in image_pd.iterrows():
        print("Processing image number", row.image_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        original_los = psi_d_types.read_los_image(hdf_path)
        original_los.get_coordinates(R0=R0)
        # query for theoretic LBCC parameters
        theoretic_query = db_funcs.query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'],
                                                 instrument=instrument)

        # calculate beta and y from theoretic fit
        beta, y = lbcc.get_beta_y_theoretic_interp(theoretic_query, mu_array_2d=original_los.mu,
                                                   mu_array_1d=mu_bin_centers)

        ###### APPLY LBC CORRECTION ######
        corrected_data = beta * original_los.data + y
        corrected_los_data[index, :, :] = corrected_data
        intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')

        lbcc_data = psi_d_types.create_lbcc_data(corrected_data, original_los)
        hist_test, use_data = psi_d_types.create_iit_hist(lbcc_data,
                                                          intensity_bin_array=intensity_bin_edges,
                                                          lat_band=lat_band, log10=True)

        norm_hist = np.full(hist_test.shape, 0.)
        row_sums = hist_test.sum(axis=0, keepdims=True)
        # but do not divide by zero
        zero_row_index = np.where(row_sums != 0)
        norm_hist[zero_row_index[0]] = hist_test[zero_row_index[0]] / row_sums[zero_row_index[0]]

plt.plot(intensity_bin_edges[1:], hist_test)

######## ignore
norm_hist = np.full(lbcc_data[0].shape, 0.)
hist_ref = norm_hist[-1,]
hist_ref.astype(np.float32)
hist_fit = lbcc_data[0].flatten()
bin_edges = np.linspace(0, 5, num=200 + 1, dtype='float')
plt.hist(hist_fit, bins=200)
hist_data = np.log10(lbcc_data[0])
sns.distplot(hist_data, bins=200, hist=True, norm_hist=True)
iit_data = iit.optim_iit_linear(hist_ref, hist_fit[-1,], bin_edges, init_pars=np.asarray([1., 0.]))
iit_data = np.log10(hist_fit)
# plot the histogram??
plt.hist(iit_data)

##### STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE ######

##### STEP TWO: CALCULATE TRANSFORMATION COEFFICIENTS ######

##### STEP THREE: APPLY TRANSFORMATION AND PLOT NEW IMAGES ######

###### STEP FOUR: GENERATE NEW HISTOGRAM PLOTS ######
