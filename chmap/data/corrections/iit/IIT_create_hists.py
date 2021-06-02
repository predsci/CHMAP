
"""
code for creation of IIT histograms and saving to database
"""

import os

# This can be a computationally intensive process.
# To limit number of threads numpy can spawn:
# os.environ["OMP_NUM_THREADS"] = "4"

import time
import datetime
import numpy as np
import pandas as pd

from settings.app import App
from chmap.database.db_funs import init_db_conn
import chmap.database.db_funs as db_funcs
import chmap.database.db_classes as db_class
import utilities.datatypes.datatypes as psi_d_types
import chmap.data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs

###### ------ UPDATEABLE PARAMETERS ------- #######
# TIME RANGE FOR LBC CORRECTION AND HISTOGRAM CREATION
lbc_query_time_min = datetime.datetime(2012, 12, 31, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2013, 1, 7, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
wavelengths = [193, 195]

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

# setup database parameters
create = True  # true if you want to add to database
# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

###### ------- NOTHING TO UPDATE BELOW ------- #######
# start time
start_time = time.time()

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

# determine date of first AIA image
# min_aia_image = db_session.query(db_class.EUV_Images.date_obs)

for instrument in inst_list:
    print("Beginning loop for instrument:", instrument)
    # query EUV images
    query_instrument = [instrument, ]
    image_pd_all = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument,
                                             wavelength=wavelengths)
    # query LBCC histograms
    hist_pd = db_funcs.query_hist(db_session, meth_id=method_id[1], n_intensity_bins=n_intensity_bins,
                                  lat_band=lat_band, time_min=lbc_query_time_min,
                                  time_max=lbc_query_time_max, instrument=query_instrument,
                                  wavelength=wavelengths)

    if hist_pd.shape[0] == 0:
        # use all images in range
        in_index = pd.Series([False] * image_pd_all.shape[0])
    else:
        # compare image results to hist results based on image_id
        in_index = image_pd_all.image_id.isin(hist_pd.image_id)

    # return only images that do not have corresponding histograms
    image_pd = image_pd_all[~in_index]

    # check that images remain that need histograms
    if image_pd.shape[0] == 0:
        print("All " + instrument + " images in timeframe already have associated histograms.")
        continue

    # query correct image combos
    # combo_query = db_funcs.query_inst_combo(db_session, lbc_query_time_min, lbc_query_time_max, meth_name="LBCC",
    #                                         instrument=instrument)

    # apply LBC
    for index, row in image_pd.iterrows():
        print("Calculating IIT histogram at time:", row.date_obs)
        # original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session,
        #                                                                         hdf_data_dir, combo_query,
        #                                                                         image_row=row,
        #                                                                         n_intensity_bins=n_intensity_bins,
        #                                                                         R0=R0)


        original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc_2(
            db_session, hdf_data_dir, image_row=row, n_intensity_bins=n_intensity_bins, R0=R0)
        # check that image load and LBCC application finished successfully
        if original_los is None:
            continue

        # calculate IIT histogram from LBC
        hist = psi_d_types.LBCCImage.iit_hist(lbcc_image, lat_band, log10)

        # create IIT histogram datatype
        iit_hist = psi_d_types.create_iit_hist(lbcc_image, method_id[1], lat_band, hist)

        # add IIT histogram and meta data to database
        db_funcs.add_hist(db_session, iit_hist)

db_session.close()

end_time = time.time()
print("Inter-instrument transformation histograms have been created and saved to the database.")
print(
    "Total elapsed time for histogram creation: " + str(round(end_time - start_time, 3)) + " seconds.")
