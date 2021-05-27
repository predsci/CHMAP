
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from settings.app import App
from database.db_funs import init_db_conn
import database.db_funs as db_funcs
import database.db_classes as db_class
import modules.datatypes as psi_d_types
import data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs

###### ------ UPDATEABLE PARAMETERS ------- #######
# TIME RANGE FOR LBC CORRECTION AND HISTOGRAM CREATION
lbc_query_time_min = datetime.datetime(2011, 6, 1, 0, 0, 0)
lbc_query_time_max = datetime.datetime(2013, 1, 1, 0, 0, 0)

# define instruments
inst_list = ["AIA", "EUVI-A", "EUVI-B"]

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
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
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
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


# create IIT method
meth_name = "IIT"
meth_desc = "IIT Fit Method"
method_id = db_funcs.get_method_id(db_session, meth_name, meth_desc, var_names=None, var_descs=None, create=True)

# for instrument in inst_list:
instrument = inst_list[1]


print("Calculation for instrument:", instrument)
# query EUV images
query_instrument = [instrument, ]
image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                     time_max=lbc_query_time_max, instrument=query_instrument)
# query correct image combos
combo_query = db_funcs.query_inst_combo(db_session, lbc_query_time_min, lbc_query_time_max, meth_name="LBCC",
                                        instrument=instrument)
# apply LBC
# for index, row in image_pd.iterrows():
index = 0
row = image_pd.iloc[index]

print("Calculating IIT histogram at time:", row.date_obs)
original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                                                                                          hdf_data_dir, combo_query,
                                                                                          image_row=row,
                                                                                          n_intensity_bins=n_intensity_bins,
                                                                                          R0=R0)
# calculate IIT histogram from LBC
hist = psi_d_types.LBCCImage.iit_hist(lbcc_image, lat_band, log10)

# create IIT histogram datatype
iit_hist = psi_d_types.create_iit_hist(lbcc_image, method_id[1], lat_band, hist)



# Now query database for histogram to compare
db_hist = db_funcs.query_hist(db_session, iit_hist.meth_id, n_mu_bins=iit_hist.n_mu_bins,
                    n_intensity_bins=iit_hist.n_intensity_bins, lat_band=lat_band,
                    time_min=row.date_obs.to_pydatetime(), time_max=row.date_obs.to_pydatetime(),
                    )

# convert binary to histogram data
mu_bin_edges_db, intensity_bin_edges_db, db_full_hist = psi_d_types.binary_to_hist(hist_binary=db_hist,
                                   n_mu_bins=None, n_intensity_bins=n_intensity_bins)

# plot newly calc'd IIT-hist vs DB-IIT-hist
plt.plot(intensity_bin_edges_db[1:], db_full_hist, 'r', label="From DB")
plt.plot(intensity_bin_edges_db[1:], hist, 'b', label="New Calc")
ax = plt.gca()
ax.legend(loc='upper right', title="")


# Now test database histogram write/read functionality
iit_hist.date_obs = datetime.datetime(1985, 1, 1, 0, 0, 0, 0)

# delete any previous test histograms
# db_session.query(db_class.Histogram).filter(db_class.Histogram.meth_id == iit_hist.meth_id,
#                                             db_class.Histogram.date_obs < datetime.datetime(1990, 1, 1, 0, 0, 0, 0)
#                                             ).delete()
# create bogus image record
image_add = db_class.EUV_Images(date_obs=iit_hist.date_obs, instrument=iit_hist.instrument,
                       wavelength=iit_hist.wavelength, fname_raw="abcd",
                       fname_hdf="abcd", distance=0, cr_lon=0,
                       cr_lat=0, cr_rot=0,
                       time_of_download=datetime.datetime.now())
# Push to DB and record new data_id
db_session.add(image_add)
db_session.flush()
new_data_id = image_add.data_id
db_session.commit()

iit_hist.data_id = new_data_id

# write newly calc'd histogram to DB with date 1985
db_funcs.add_hist(db_session, iit_hist)

# read newly calc'd histogram from DB
new_db_hist = db_funcs.query_hist(db_session, iit_hist.meth_id, n_mu_bins=iit_hist.n_mu_bins,
                    n_intensity_bins=iit_hist.n_intensity_bins, lat_band=lat_band,
                    time_min=iit_hist.date_obs, time_max=iit_hist.date_obs,
                    )

# convert binary to histogram data
new_db_mu_bin_edges, new_db_int_bin_edges, new_db_full_hist = psi_d_types.binary_to_hist(hist_binary=new_db_hist,
                                   n_mu_bins=None, n_intensity_bins=n_intensity_bins)

# plot calc'd hist vs hist returned from DB
plt.plot(intensity_bin_edges_db[1:], new_db_full_hist, 'r', label="From DB")
plt.plot(new_db_int_bin_edges[1:], hist, 'b', label="Original")
ax = plt.gca()
ax.legend(loc='upper right', title="")


# clean-up DB
# remove test histogram record
db_session.query(db_class.Histogram).filter(db_class.Histogram.image_id == new_data_id,
                                            db_class.Histogram.meth_id == iit_hist.meth_id
                                            ).delete()
# remove bogus image record
db_session.query(db_class.EUV_Images).filter(db_class.EUV_Images.data_id == new_data_id
                                             ).delete()

# close connection
db_session.close()
