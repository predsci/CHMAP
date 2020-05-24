"""
construct mu-histogram and push to database for any time period
"""

import os
import datetime
import numpy as np
from settings.app import App
import modules.DB_classes as db_class
from modules.DB_funs import init_db_conn, query_euv_images, add_lbcc_hist
import modules.datatypes as psi_d_types

# PARAMETERS TO UPDATE

# generate plots if true
generate_plots = False

# TIME RANGE
query_time_min = datetime.datetime(2011, 1, 4, 0, 0, 0)
query_time_max = datetime.datetime(2011, 1, 8, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

# define number of bins
n_mu_bins = 18
n_intensity_bins = 200

# declare map and binning parameters
R0 = 1.01
mu_bin_edges = np.array(range(n_mu_bins + 1), dtype="float") * 0.05 + 0.1
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')
log10 = True
lat_band = [-np.pi / 64., np.pi / 64.]

# recover database paths
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)
db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)

# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# loop over instrument
for instrument in inst_list:
    # query wants a list
    query_instrument = [instrument, ]
    query_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max,
                                instrument=query_instrument)

    for index, row in query_pd.iterrows():
        print("Processing image number", row.image_id, ".")
        if row.fname_hdf == "":
            print("Warning: Image # " + str(row.image_id) + " does not have an associated hdf file. Skipping")
            continue
        hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
        los_temp = psi_d_types.read_los_image(hdf_path)
        # add coordinates to los object (is there a way to do this outside the loop?)
        los_temp.get_coordinates(R0=R0)
        # perform 2D histogram on mu and image intensity
        temp_hist = los_temp.mu_hist(image_intensity_bin_edges, mu_bin_edges, lat_band=lat_band, log10=log10)
        hist_lbcc = psi_d_types.create_hist(hdf_path, row.image_id, mu_bin_edges, image_intensity_bin_edges, lat_band, temp_hist)

        # add this histogram and meta data to database
        add_lbcc_hist(hist_lbcc, db_session)

db_session.close()


