"""
Apply LBC to images based on theoretic fit.
Get images and parameers from database to apply correction.
"""
import sys
# path to modules and settings folders
sys.path.append('/Users/tamarervin/work/chd')

import os
import numpy as np
import datetime

from settings.app import App
from modules.DB_funs import init_db_conn, query_euv_images, query_lbcc_fit, query_var_val
import modules.DB_classes as db_class
import modules.datatypes as psi_d_types
import modules.Plotting as Plotting
import modules.lbcc_funs as lbcc

plot=False

# define time range to query
query_time_min = datetime.datetime(2011, 4, 1, 0, 0, 0)
query_time_max = datetime.datetime(2011, 4, 1, 3, 0, 0)

# define instruments
inst_list = ['AIA', "EUVI-A", "EUVI-B"]
#inst_list = ['AIA']
meth_name = "LBCC Theoretic"

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
# query wants a list
for inst_index, instrument in enumerate(inst_list):
    # query wants a list
    query_instrument = [instrument, ]
    image_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                                instrument=query_instrument)

    ###### GET LOS IMAGES COORDINATES (DATA) #####
    for index, row in image_pd.iterrows():
        print("Processing image number " + str(row.image_id) + ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        original_los = psi_d_types.read_los_image(hdf_path)
        original_los.get_coordinates(R0=R0)
        print("date obs:", original_los.info['date_string'])
        theoretic_query = query_var_val(db_session, meth_name, date_obs=original_los.info['date_string'], instrument=instrument)
        # get beta and y from theoretic fit
        beta, y = lbcc.get_beta_y_theoretic_continuous_loop(theoretic_query, original_los.mu)
        corrected_los_data = beta * original_los.data + y

        ###### APPLY LBC CORRECTION ######
        Plotting.PlotImage(original_los, nfig=400 + inst_index, title="Original LOS Image for " + instrument)
        Plotting.PlotLBCCImage(lbcc_data=corrected_los_data, los_image=original_los, nfig=500 + inst_index,
                               title="Corrected LBCC Image for " + instrument)

        #Plotting.PlotLBCCImage(lbcc_data=original_los.mu, los_image=original_los, nfig=600 + inst_index,
         #                      title = "Mu Plot for " + instrument)
        #Plotting.PlotLBCCImage(lbcc_data=beta, los_image=original_los, nfig=700 + inst_index,
                             #  title = "Beta Plot for " + instrument)
        Plotting.PlotLBCCImage(lbcc_data=original_los.data - corrected_los_data, los_image=original_los,
                               nfig=800 + inst_index, title = "Difference Plot for " + instrument)

        if plot:

            Plotting.Plot2D_Data(data=original_los.mu, nfig=60+inst_index, xlabel='x (solar radii)',
                                 ylabel='y (solar radii)', title="Mu Plot for " + instrument)
            Plotting.Plot2D_Data(data=beta, nfig=70 + inst_index, xlabel='x (solar radii)',
                                 ylabel='y (solar radii)', title="Beta Plot for " + instrument)
            Plotting.Plot2D_Data(data=original_los.data-corrected_los_data, nfig=80 + inst_index,
                                 xlabel='x (solar radii)', ylabel='y (solar radii)',
                                 title="Difference Plot for " + instrument)


