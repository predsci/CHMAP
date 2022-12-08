
##############################################
# After updating processed images, update all
# effected LLBC values in the database
##############################################

import datetime
import os
import pandas as pd
from sqlalchemy import func
import numpy as np

import chmap.database.db_classes as DBClass
import chmap.database.db_funs as db_funs
from chmap.database.db_funs import init_db_conn_old, update_image_val
import chmap.data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs
import chmap.utilities.utils as psi_utils
import chmap.utilities.datatypes.datatypes as psi_d_types
import chmap.settings.ch_pipeline_pars as pipe_pars

# Time-averaging window full-width
LBCC_window = pipe_pars.LBCC_window  # days for moving average
window_del = datetime.timedelta(days=LBCC_window)
weekday_calc = pipe_pars.LBCC_weekday  # start at 0 for Monday

# TIME WINDOWS FOR IMAGE INCLUSION - do not include all images, just synchronic images on an
# 'image_freq' cadence.
image_freq = pipe_pars.map_freq         # number of hours between window centers
image_del = np.timedelta64(pipe_pars.interval_delta, 'm')   # one-half window width

# MAP AND BINNING PARAMETERS
n_mu_bins = pipe_pars.n_mu_bins
n_intensity_bins = pipe_pars.n_intensity_bins
R0 = pipe_pars.R0
log10 = pipe_pars.log10
lat_band = pipe_pars.LBCC_lat_band

# INSTRUMENTS
inst_list = pipe_pars.inst_list
wavelengths = pipe_pars.IIT_query_wavelengths

#########################
# 1. query latest image
# 2. query latest LBCC
# 3. update LBCCs from latest_LBCC-period/2 to latest_image-period/2
#########################

# processed image directory
hdf_data_dir = "/Volumes/extdata2/CHD_DB/processed_images"
# database location (only for sqlite)
database_dir = None
# give the sqlite file a unique name (only for sqlite)
sqlite_filename = None

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn_old() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, sqlite_path=sqlite_path)
elif use_db in ['mysql-Q', 'mysql-Q_test']:
    # setup database connection to MySQL database on Q
    db_session = init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

# --- Query database for most recent processed image --------------------------------
raw_im_query = db_session.query(func.max(DBClass.Data_Files.date_obs)).join(DBClass.EUV_Images).filter(
    DBClass.Data_Files.fname_hdf != "", DBClass.Data_Files.type == "EUV_Image",
    DBClass.EUV_Images.flag == 0)
max_im_date = pd.read_sql(raw_im_query.statement, db_session.bind)

# --- Query database for most recent LBCC coef --------------------------------------
meth_name = "LBCC"
db_sesh, meth_id, var_ids = db_funs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                  var_descs=None, create=False)
# query var defs for variable id
var_id = db_funs.get_var_id(db_session, meth_id, var_name=None, var_desc=None, create=False)
# query for most recent LBCC coeffs
lbcc_coef_query = db_session.query(func.max(DBClass.Data_Combos.date_mean)).filter(
    DBClass.Data_Combos.meth_id == meth_id)
max_lbcc_coef = pd.read_sql(lbcc_coef_query.statement, db_session.bind)

# --- Query database for all images that need LBCC histograms -----------------------
# generate a list of processed EUV images with flag=0
proc_image_query = db_session.query(DBClass.Data_Files, DBClass.EUV_Images.instrument).join(
    DBClass.EUV_Images).filter(
    DBClass.Data_Files.fname_hdf != "", DBClass.Data_Files.type == "EUV_Image",
    DBClass.EUV_Images.flag == 0
)
# generate a list of LBCC histograms
lbcc_hists_query = db_session.query(DBClass.Histogram).filter(DBClass.Histogram.meth_id == meth_id).subquery()
# do a negative outer join with Histograms table subquery to determine which processed images
# do not yet have an LBCC histogram
non_hist_im_query = proc_image_query.outerjoin(
    lbcc_hists_query, DBClass.Data_Files.data_id == lbcc_hists_query.c.image_id).filter(
    lbcc_hists_query.c.image_id == None)
# send query to MySQL DB and collect results
hist_images = pd.read_sql(non_hist_im_query.statement, db_session.bind)

# --- Set update dates --------------------------------------------------------------
# set dates for updating coefficients
lbcc_coef_start = max_lbcc_coef.loc[0][0].to_pydatetime()
lbcc_coef_end = max_im_date.loc[0][0].to_pydatetime() - window_del/2
# round to nearest day (midnight)
lbcc_coef_start = psi_utils.round_day(lbcc_coef_start)
lbcc_coef_end = psi_utils.round_day(lbcc_coef_end)

# --- Generate new histograms as needed ---------------------------------------------
# lbcc_funcs.save_histograms(db_session, hdf_data_dir, inst_list, lbcc_hist_start, lbcc_hist_end,
#                            n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10,
#                            R0=R0, wavelengths=wavelengths)

# creates mu bin & intensity bin arrays
mu_bin_edges = np.linspace(0.1, 1.0, n_mu_bins + 1, dtype='float')
image_intensity_bin_edges = np.linspace(0, 5, num=n_intensity_bins + 1, dtype='float')
# loop through images and produce intensity histograms
for index, row in hist_images.iterrows():
    print("Processing image number", row.data_id, ".")
    if row.fname_hdf == "":
        print("Warning: Image # " + str(row.data_id) + " does not have an associated hdf file. Skipping")
        continue
    hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
    # attempt to open and read file
    try:
        los_temp = psi_d_types.read_euv_image(hdf_path)
    except:
        print("Something went wrong opening: ", hdf_path, ". Skipping")
        continue
    # add coordinates to los object
    los_temp.get_coordinates(R0=R0)
    # perform 2D histogram on mu and image intensity
    temp_hist = los_temp.mu_hist(image_intensity_bin_edges, mu_bin_edges, lat_band=lat_band, log10=log10)
    hist_lbcc = psi_d_types.create_lbcc_hist(hdf_path, row.data_id, meth_id, mu_bin_edges,
                                             image_intensity_bin_edges, lat_band, temp_hist)

    # add this histogram and meta data to database
    db_funs.add_hist(db_session, hist_lbcc)


# --- Delete existing LBCC coefs in update window -----------------------------------
# combos_query = db_session.query(DBClass.Data_Combos).filter(
#     DBClass.Data_Combos.meth_id == meth_id,
#     DBClass.Data_Combos.date_mean >= lbcc_coef_start
#     )
# combos_del = pd.read_sql(combos_query.statement, db_session.bind)
# vars_query = db_session.query(DBClass.Var_Vals).filter(
#     DBClass.Var_Vals.combo_id.in_(combos_del.combo_id)
# )
# vars_del = vars_query.delete()
# # also delete these combos
# combo_assoc = db_session.query(DBClass.Data_Combo_Assoc).filter(
#     DBClass.Data_Combo_Assoc.combo_id.in_(combos_del.combo_id)
# )
# assoc_del = combo_assoc.delete()
# c_num_del = combos_query.delete()
# db_session.commit()

# --- Generate updated LBCC coefs for update window ---------------------------------
lbcc_funcs.calc_theoretic_fit(db_session, inst_list, lbcc_coef_start,
                              lbcc_coef_end, weekday=weekday_calc, image_freq=image_freq,
                              image_del=image_del, number_of_days=LBCC_window, n_mu_bins=n_mu_bins,
                              n_intensity_bins=n_intensity_bins, lat_band=lat_band, create=True,
                              wavelengths=wavelengths)

db_session.close()
