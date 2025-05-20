
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
import chmap.utilities.datatypes.datatypes as psi_d_types
import chmap.data.corrections.lbcc.LBCC_theoretic_funcs as lbcc_funcs
import chmap.data.corrections.iit.IIT_pipeline_funcs as iit_funcs
import chmap.utilities.utils as psi_utils
import chmap.settings.ch_pipeline_pars as pipe_pars

####### ------ UPDATABLE PARAMETERS ------ #########
# TIME WINDOW FOR IIT PARAMETER CALCULATION
weekday = pipe_pars.IIT_weekday  # start at 0 for Monday
number_of_days = pipe_pars.IIT_number_of_days  # days for moving average
window_del = datetime.timedelta(days=number_of_days)
# TIME WINDOWS FOR IMAGE INCLUSION - do not include all images in time range,
# only those that are synchronic at a specific 'image_freq' cadence
image_freq = pipe_pars.map_freq      # number of hours between window centers
image_del = np.timedelta64(pipe_pars.interval_delta, 'm')   # one-half window width

# INSTRUMENTS
inst_list = pipe_pars.inst_list
ref_inst = pipe_pars.IIT_ref_inst
wavelengths = pipe_pars.IIT_query_wavelengths

# declare map and binning parameters
n_mu_bins = pipe_pars.n_mu_bins
n_intensity_bins = pipe_pars.n_intensity_bins
R0 = pipe_pars.R0
log10 = pipe_pars.log10
lat_band = pipe_pars.IIT_lat_band

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
db_session, meth_id, var_ids = db_funs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
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
db_session, meth_id, var_ids = db_funs.get_method_id(db_session, meth_name, meth_desc=None, var_names=None,
                                                  var_descs=None, create=False)
# query var defs for variable id
var_id = db_funs.get_var_id(db_session, meth_id, var_name=None, var_desc=None, create=False)
# query for most recent IIT coeffs
iit_coef_query = db_session.query(func.max(DBClass.Data_Combos.date_mean)).filter(
    DBClass.Data_Combos.meth_id == meth_id)
max_iit_coef = pd.read_sql(iit_coef_query.statement, db_session.bind)

# --- Query database for all images that need LBCC histograms -----------------------
# generate a list of processed EUV images with flag=0
proc_image_query = db_session.query(DBClass.Data_Files, DBClass.EUV_Images.instrument).join(
    DBClass.EUV_Images).filter(
    DBClass.Data_Files.fname_hdf != "", DBClass.Data_Files.type == "EUV_Image",
    DBClass.EUV_Images.flag == 0
)
# specify a query of all IIT histograms
# iit_hists_query = db_session.query(DBClass.Histogram).filter(DBClass.Histogram.meth_id == meth_id).subquery()
# # do a negative outer join with Histograms table subquery to determine which processed images
# # do not yet have an IIT histogram
# # non_hist_im_query = proc_image_query.outerjoin(
# #     iit_hists_query, DBClass.Data_Files.data_id == iit_hists_query.c.image_id).filter(
# #     iit_hists_query.c.image_id == None)
# non_hist_im_query = proc_image_query.outerjoin(
#     iit_hists_query, DBClass.Data_Files.data_id == iit_hists_query.c.image_id).filter(
#     iit_hists_query.c.image_id == None, DBClass.Data_Files.date_obs <= max_lbcc_coef.loc[0][0].to_pydatetime())
# # send query to MySQL DB and collect results
# iit_images = pd.read_sql(non_hist_im_query.statement, db_session.bind)

# In this case, we only need all AIA images from 2024-05-01
iit_images = db_funs.query_euv_images(db_session, time_min=datetime.datetime(2024, 5, 1, 0, 0, 0),
                                       time_max=datetime.datetime.now(), instrument=["AIA", ])

# --- Set update dates --------------------------------------------------------------
# set dates for updating coefficients
iit_coef_start = datetime.datetime(2024, 2, 1, 0, 0, 0)
iit_coef_end = max_lbcc_coef.loc[0][0].to_pydatetime() - window_del/2
# round to nearest day (midnight)
iit_coef_start = psi_utils.round_day(iit_coef_start)
iit_coef_end = psi_utils.round_day(iit_coef_end)


##### ------ INTER INSTRUMENT TRANSFORMATION FUNCTIONS BELOW ------- ########

# --- STEP ONE: CREATE 1D HISTOGRAMS AND SAVE TO DATABASE --------
# iit_funcs.create_histograms(db_session, inst_list, iit_hist_start, iit_hist_end, hdf_data_dir,
#                             n_intensity_bins=n_intensity_bins, lat_band=lat_band, log10=log10, R0=R0,
#                             wavelengths=wavelengths)

# loop through images
for index, row in iit_images.iterrows():
    print("Calculating IIT histogram at time:", row.date_obs)

    original_los, lbcc_image, mu_indices, use_indices, theoretic_query = lbcc_funcs.apply_lbc_2(
        db_session, hdf_data_dir, image_row=row, n_intensity_bins=n_intensity_bins, R0=R0)
    # check that image load and LBCC application finished successfully
    if original_los is None:
        continue

    # calculate IIT histogram from LBC
    hist = psi_d_types.LBCCImage.iit_hist(lbcc_image, lat_band, log10)

    # create IIT histogram datatype
    iit_hist = psi_d_types.create_iit_hist(lbcc_image, meth_id, lat_band, hist)

    # add IIT histogram and meta data to database
    db_funs.add_hist(db_session, iit_hist, overwrite=True)


# --- Delete existing IIT coefs in update window -----------------------------------
# combos_query = db_session.query(DBClass.Data_Combos).filter(
#     DBClass.Data_Combos.meth_id == meth_id,
#     DBClass.Data_Combos.date_mean >= iit_coef_start
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


# --- CALCULATE INTER-INSTRUMENT TRANSFORMATION COEFFICIENTS AND SAVE TO DATABASE ----
iit_funcs.calc_iit_coefficients(db_session, inst_list, ref_inst, iit_coef_start, iit_coef_end,
                                weekday=weekday, number_of_days=number_of_days, image_freq=image_freq,
                                image_del=image_del, n_intensity_bins=n_intensity_bins,
                                lat_band=lat_band, create=create, wavelengths=wavelengths)

db_session.close()

