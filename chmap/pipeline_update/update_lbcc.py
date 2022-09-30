
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

# Time-averaging window full-width
LBCC_window = 180  # days for moving average
window_del = datetime.timedelta(days=LBCC_window)
weekday_calc = 0  # start at 0 for Monday

# TIME WINDOWS FOR IMAGE INCLUSION - do not include all images, just synchronic images on an
# 'image_freq' cadence.
image_freq = 2                          # number of hours between window centers
image_del = np.timedelta64(30, 'm')     # one-half window width

# MAP AND BINNING PARAMETERS
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [- np.pi / 64., np.pi / 64.]

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
wavelengths = [193, 195]

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

# --- Query database for most recent LBCC hist --------------------------------------
lbcc_hist_query = db_session.query(func.max(DBClass.Histogram.date_obs)).filter(
    DBClass.Histogram.meth_id == meth_id)
max_lbcc_hist = pd.read_sql(lbcc_hist_query.statement, db_session.bind)

# --- Set update dates --------------------------------------------------------------
# set dates for updating histograms
lbcc_hist_start = max_lbcc_hist.loc[0][0].to_pydatetime()
lbcc_hist_end = max_im_date.loc[0][0].to_pydatetime()

# set dates for updating coefficients
lbcc_coef_start = max_lbcc_coef.loc[0][0].to_pydatetime()
lbcc_coef_end = max_im_date.loc[0][0].to_pydatetime() - window_del/2
# round to nearest day (midnight)
lbcc_coef_start = psi_utils.round_day(lbcc_coef_start)
lbcc_coef_end = psi_utils.round_day(lbcc_coef_end)

# --- Generate new histograms as needed ---------------------------------------------
lbcc_funcs.save_histograms(db_session, hdf_data_dir, inst_list, lbcc_hist_start, lbcc_hist_end,
                           n_mu_bins=n_mu_bins, n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10,
                           R0=R0, wavelengths=wavelengths)

# --- Delete existing LBCC coefs in update window -----------------------------------
combos_query = db_session.query(DBClass.Data_Combos).filter(
    DBClass.Data_Combos.meth_id == meth_id,
    DBClass.Data_Combos.date_mean >= lbcc_coef_start
    )
combos_del = pd.read_sql(combos_query.statement, db_session.bind)
vars_query = db_session.query(DBClass.Var_Vals).filter(
    DBClass.Var_Vals.combo_id.in_(combos_del.combo_id)
)
vars_del = vars_query.delete()
db_session.commit()

# --- Generate updated LBCC coefs for update window ---------------------------------
lbcc_funcs.calc_theoretic_fit(db_session, inst_list, lbcc_coef_start,
                              lbcc_coef_end, weekday=weekday_calc, image_freq=image_freq,
                              image_del=image_del, number_of_days=LBCC_window, n_mu_bins=n_mu_bins,
                              n_intensity_bins=n_intensity_bins, lat_band=lat_band, create=True,
                              wavelengths=wavelengths)

db_session.close()
