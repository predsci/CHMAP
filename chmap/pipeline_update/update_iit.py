
##############################################
# After updating LBCC coeffs, update all
# effected IIT values in the database
##############################################


import datetime
import numpy as np
import pandas as pd
from sqlalchemy import func

import chmap.database.db_classes as DBClass
import chmap.database.db_funs as db_funs
import chmap.data.corrections.iit.IIT_pipeline_funcs as iit_funcs
import chmap.utilities.utils as psi_utils

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME WINDOW FOR IIT PARAMETER CALCULATION
weekday = 0  # start at 0 for Monday
number_of_days = 180  # days for moving average
window_del = datetime.timedelta(days=number_of_days)
# TIME WINDOWS FOR IMAGE INCLUSION - do not include all images in time range,
# only those that are synchronic at a specific 'image_freq' cadence
image_freq = 2      # number of hours between window centers
image_del = np.timedelta64(30, 'm')  # one-half window width

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
ref_inst = "AIA"  # reference instrument to fit histograms to
wavelengths = [193, 195]

# declare map and binning parameters
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
lat_band = [-np.pi / 2.4, np.pi / 2.4]

# DATABASE FILE-SYSTEM PATHS
raw_data_dir = "/Volumes/extdata2/CHD_DB/raw_images"
hdf_data_dir = "/Volumes/extdata2/CHD_DB/processed_images"

# DATABASE CONNECTION
create = True  # set to False to disallow writes to the database
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
# setup database connection to MySQL database on Q
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user, password=password)

# --- Query update datetimes --------------------------
# 1. Most recent LBCC coef
# 2. Most recent IIT coef

meth_name = "LBCC"
db_sesh, meth_id, var_ids = db_funs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                  var_descs=None, create=False)
# query var defs for variable id
var_id = db_funs.get_var_id(db_session, meth_id, var_name=None, var_desc=None, create=False)
# query for most recent LBCC coeffs
lbcc_coef_query = db_session.query(func.max(DBClass.Data_Combos.date_mean)).filter(
    DBClass.Data_Combos.meth_id == meth_id)
max_lbcc_coef = pd.read_sql(lbcc_coef_query.statement, db_session.bind)

# Query database for most recent LBCC hist
# lbcc_hist_query = db_session.query(func.max(DBClass.Histogram.date_obs)).filter(
#     DBClass.Histogram.meth_id == meth_id)
# max_lbcc_hist = pd.read_sql(lbcc_hist_query.statement, db_session.bind)

meth_name = "IIT"
db_sesh, meth_id, var_ids = db_funs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                  var_descs=None, create=False)
# query var defs for variable id
var_id = db_funs.get_var_id(db_session, meth_id, var_name=None, var_desc=None, create=False)
# query for most recent IIT coeffs
iit_coef_query = db_session.query(func.max(DBClass.Data_Combos.date_mean)).filter(
    DBClass.Data_Combos.meth_id == meth_id)
max_iit_coef = pd.read_sql(iit_coef_query.statement, db_session.bind)

# Query database for most recent IIT hist
iit_hist_query = db_session.query(func.max(DBClass.Histogram.date_obs)).filter(
    DBClass.Histogram.meth_id == meth_id)
max_iit_hist = pd.read_sql(iit_hist_query.statement, db_session.bind)

# --- Set update dates --------------------------------------------------------------
# set dates for updating histograms
iit_hist_start = max_iit_hist.loc[0][0].to_pydatetime()
iit_hist_end = max_lbcc_coef.loc[0][0].to_pydatetime()

# set dates for updating coefficients
iit_coef_start = max_iit_coef.loc[0][0].to_pydatetime()
iit_coef_end = max_lbcc_coef.loc[0][0].to_pydatetime() - window_del/2
# round to nearest day (midnight)
iit_coef_start = psi_utils.round_day(iit_coef_start)
iit_coef_end = psi_utils.round_day(iit_coef_end)


##### ------ INTER INSTRUMENT TRANSFORMATION FUNCTIONS BELOW ------- ########

# --- STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE --------
iit_funcs.create_histograms(db_session, inst_list, iit_hist_start, iit_hist_end, hdf_data_dir,
                            n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10, R0=R0,
                            wavelengths=wavelengths)

# --- Delete existing IIT coefs in update window -----------------------------------
combos_query = db_session.query(DBClass.Data_Combos).filter(
    DBClass.Data_Combos.meth_id == meth_id,
    DBClass.Data_Combos.date_mean >= iit_coef_start
    )
combos_del = pd.read_sql(combos_query.statement, db_session.bind)
vars_query = db_session.query(DBClass.Var_Vals).filter(
    DBClass.Var_Vals.combo_id.in_(combos_del.combo_id)
)
vars_del = vars_query.delete()

# --- CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS AND SAVE TO DATABASE ----
iit_funcs.calc_iit_coefficients(db_session, inst_list, ref_inst, iit_coef_start, iit_coef_end,
                                weekday=weekday, number_of_days=number_of_days, image_freq=image_freq,
                                image_del=image_del, n_intensity_bins=n_intensity_bins,
                                lat_band=lat_band, create=create, wavelengths=wavelengths)



