
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy

import sunpy
import sunpy.visualization.colormaps as sunpy_cm
import astropy.units as u

import chmap.data.corrections.degradation.AIA as aia_degrad
from chmap.database import db_funs
import chmap.database.db_classes as DBClass
import chmap.utilities.datatypes.datatypes as datatypes
import chmap.data.corrections.apply_lbc_iit as apply_lbc_iit
import chmap.coronal_holes.detection.chd_funcs as chd_funcs
import chmap.maps.util.map_manip as map_manip
import chmap.utilities.file_io.io_helpers as io_helpers
import chmap.utilities.plotting.psi_plotting as psi_plt

# -------------------------------------------------
# --- Plot time-series of AIA degradation factor
# -------------------------------------------------

AIA_wave = 193

start_time = datetime.datetime(2009, 1, 1, 0)
end_time = datetime.datetime.today()

plot_dates = pd.date_range(start_time, end_time, freq="D")
plot_df = pd.DataFrame(dict(date=plot_dates, factor=1., alpha=0., x=0.))

# load AIA degradation information
json_dict = aia_degrad.load_aia_json()
timedepend_dict = aia_degrad.process_aia_timedepend_json(json_dict)

# get correction factor
plot_df.factor = aia_degrad.get_aia_timedepend_factor(timedepend_dict, plot_dates, AIA_wave)

# for index, row in plot_df.iterrows():
#     plot_df.loc[index, 'factor'] = aia_degrad.get_aia_timedepend_factor(timedepend_dict, row.date, AIA_wave)

plt.plot(plot_df.date, plot_df.factor)


# -------------------------------------------
# --- Test OFTpy improvements on AIA image
# -------------------------------------------
hdf_data_dir = "/Volumes/extdata2/CHD_DB/processed_images"
hdf_rel = "2017/07/01/aia_lvl2_20170701T020016_193.h5"
AIA_file = os.path.join(hdf_data_dir, hdf_rel)

image_dir = "/Users/turtle/Dropbox/MyNACD/analysis/aia_degrad/test_images"

# the calibration factor in 2014 (not sure I want to modify the threshold before this time.
cal_factor_max = 0.8668075337953921

# INSTRUMENTS
inst_list = ["AIA"]
# CORRECTION PARAMETERS
n_intensity_bins = 200
R0 = 1.01

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1_0 = 0.95
thresh2_0 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 640
reduce_map_nxcoord = 1600
full_map_nycoord = 2048
full_map_nxcoord = 2048*2
low_res_nycoord = 160
low_res_nxcoord = 400
# del_y = (y_range[1] - y_range[0]) / (map_nycoord - 1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0]) / del_y) + 1).astype(int)
# generate map x,y grids. y grid centered on equator, x referenced from lon=0
full_map_y = np.linspace(y_range[0], y_range[1], full_map_nycoord, dtype='<f4')
full_map_x = np.linspace(x_range[0], x_range[1], full_map_nxcoord, dtype='<f4')
reduce_map_y = np.linspace(y_range[0], y_range[1], reduce_map_nycoord, dtype='<f4')
reduce_map_x = np.linspace(x_range[0], x_range[1], reduce_map_nxcoord, dtype='<f4')
low_res_y = np.linspace(y_range[0], y_range[1], low_res_nycoord, dtype='<f4')
low_res_x = np.linspace(x_range[0], x_range[1], low_res_nxcoord, dtype='<f4')

# --- Database Connection for correction params -----------
# designate which database to connect to
use_db = "mysql-Q"  # 'sqlite'  Use local sqlite file-based db

user = "turtle"  # only needed for remote databases.
password = ""  # See example109 for setting-up an encrypted password.  In this case leave password="", and
cred_dir = "/Users/turtle/Dropbox/GitReps/CHMAP/chmap/settings/"

# Establish connection to database
db_session = db_funs.init_db_conn_old(db_name=use_db, chd_base=DBClass.Base, user=user,
                                      password=password, cred_dir=cred_dir)

# Also look-up Stereo A IIT coefs for the same times
for index, row in plot_df.iterrows():
    cur_date = row.date.to_pydatetime()
    theoretic_query = db_funs.get_correction_pars(db_session, "IIT", date_obs=cur_date,
                                                  instrument='EUVI-A')
    # separate alpha and x
    plot_df.loc[index, 'alpha'] = theoretic_query[0]
    plot_df.loc[index, 'x'] = theoretic_query[1]

plt.figure(0)
plt.plot(plot_df.date, plot_df.x)

plt.figure(1)
plt.plot(plot_df.date, plot_df.alpha)

# ---------------------------------------------------------------------
# Plotting function tailored for this script
# ---------------------------------------------------------------------
def plot_map_custom(map, outfile=None, title=None, pmax=10**3.5, pmin=None, scaling_range=2.5,
                    log=True, cmap=None, dpi=300, unit='DN/s'):

    if pmin is None:
        pmin = pmax/10 ** scaling_range

    if cmap is None:
        cmap = sunpy_cm.color_tables.aia_color_table(193*u.Angstrom)

    cmap.set_bad(color="k")

    map.plot_settings['cmap'] = cmap
    if log:
        map.plot_settings['norm'] = colors.LogNorm(pmin, pmax)
    else:
        map.plot_settings['norm'] = colors.Normalize(pmin, pmax)

    map.plot()

    cbar = plt.colorbar()
    cbar.set_label(f'{unit}')

    ax = plt.gca()

    if title is not None:
        ax.set_title(title)

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi)
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------
# Open Prepped AIA image
# ---------------------------------------------------------------------
test_im = datatypes.read_euv_image(AIA_file)

sol_rad = test_im.map.meta['rsun_obs']
cdelt = test_im.map.meta['cdelt1']
test_im.map.peek()
test_im.map.draw_limb()
ax = plt.gca()
ax.add_patch(plt.Circle((1020.3, 1022.5), R0*sol_rad/cdelt, color='w', fill=False))

# ---------------------------------------------------------------------
# Create a fake dataframe for this image
# ---------------------------------------------------------------------
synch_image = pd.Series(dict(data_id=0, date=test_im.map.date, instrument=test_im.info['instrument'],
                             wavelength=test_im.info['wavelength'], distance=test_im.info['distance'],
                             cr_lon=test_im.info['cr_lon'], cr_lat=test_im.info['cr_lat'],
                             cr_rot=test_im.info['cr_rot'], flag=0, time_of_download=test_im.map.date,
                             fname_raw="", fname_hdf=hdf_rel)
                        ).to_frame().T

# ---------------------------------------------------------------------
# Now apply the IIT and LBC transformations
# ---------------------------------------------------------------------
date_string = test_im.info['date_string']
center = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%f")
date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x = \
    apply_lbc_iit.apply_ipp_2(db_session, center, synch_image, inst_list, hdf_data_dir,
                              n_intensity_bins, R0)

# 2D array of the corrected image data
corrected_lbc_data = copy.deepcopy(iit_list[0].iit_data)

# ---------------------------------------------------------------------
# Now apply the time-dependent calibration factor correction
# ---------------------------------------------------------------------
# Get the time-dependent degradation factor
cal_factor = aia_degrad.get_aia_timedepend_factor(timedepend_dict, center, 193)
print(f'date: {center}, calibration factor: {cal_factor}')

# scale the LBC data
corrected_timedepend_data = corrected_lbc_data/cal_factor

# now replace the iit_list[0].iit_data tag SO IT IS CHANGED NOWWW!!! (because python can't copy f'ing objects properly...)
iit_list[0].iit_data = corrected_timedepend_data

# ---------------------------------------------------------------------
# Now apply the Coronal Hole Detection
# ---------------------------------------------------------------------
print(f'\n### Starting Coronal Hole Detection')

# adjust the thresholds so the thresh1_0 and thresh2_0 that looked good for 2014
# are equivalently applied
mod_fac = np.log10(1.0/cal_factor_max)

thresh1 = thresh1_0 + mod_fac
thresh2 = thresh2_0 + mod_fac

chd_image_list = chd_funcs.chd_2(iit_list, los_list, use_indices, thresh1,
                                 thresh2, ref_alpha, ref_x, nc, iters)

# change the flag value to be -1.0 so they can look at the FITS files more easily
chd_data = copy.copy(chd_image_list[0].data)
inds = np.where(chd_data == chd_image_list[0].no_data_val)
chd_data[inds] = -1.0

# ---------------------------------------------------------------------
# Plot the intensity images for all of the steps
# ---------------------------------------------------------------------
map_prep = test_im.map
map_lbcc = sunpy.map.Map(corrected_lbc_data, test_im.map.meta)
map_cal = sunpy.map.Map(corrected_timedepend_data, test_im.map.meta)

for plot_map, label in zip([map_prep, map_lbcc, map_cal], ['step3_deconvolved', 'step4_lb_correction',
                                                           'step5_timedepend_correction']):
    title = f'AIA 193, {date_string}, {label}'
    outfile = f'{image_dir}/chmap_AIA_193_{date_string}_{label}.png'
    unit = 'DN/s'
    if label is 'step1_raw':
        unit = 'DN'
    plot_map_custom(plot_map, title=title, outfile=outfile, unit=unit)

# ---------------------------------------------------------------------
# Interpolate to CR map
# ---------------------------------------------------------------------
nprocs = 1
tpp = 4
p_pool = None

test_im.iit_data = iit_list[0].iit_data
test_im.sunpy_meta = test_im.map.meta

aia_map = test_im.interp_to_map(R0=R0, map_x=full_map_x, map_y=full_map_y, interp_field="iit_data",
                                nprocs=nprocs, tpp=tpp, p_pool=p_pool, no_data_val=-9999.0,
                                helio_proj=True)

aia_map_reduced = map_manip.downsamp_reg_grid(aia_map, reduce_map_y, reduce_map_x)






# -------------------------------------------------
# Compare locations for AIA and Stereo-B
# -------------------------------------------------
# reproduction of Caplan 2016 - Fig 18 (right)
aia_rel = "2011/02/03/aia_lvl2_20110203T120031_193.h5"
aia_path = os.path.join(hdf_data_dir, aia_rel)
stb_rel = "2011/02/03/euvib_lvl2_20110203T120106_195.h5"
stb_path = os.path.join(hdf_data_dir, stb_rel)

aia_disk = datatypes.read_euv_image(aia_path)
aia_date_string = aia_disk.info['date_string']
center = datetime.datetime.strptime(aia_date_string, "%Y-%m-%dT%H:%M:%S.%f")
synch_image = pd.Series(dict(data_id=0, date=aia_disk.map.date, instrument=aia_disk.info['instrument'],
                             wavelength=aia_disk.info['wavelength'], distance=aia_disk.info['distance'],
                             cr_lon=aia_disk.info['cr_lon'], cr_lat=aia_disk.info['cr_lat'],
                             cr_rot=aia_disk.info['cr_rot'], flag=0, time_of_download=aia_disk.map.date,
                             fname_raw="", fname_hdf=aia_rel)
                        ).to_frame().T
date_pd, aia_list, aia_iit_list, use_indices, methods_list, ref_alpha, ref_x = \
    apply_lbc_iit.apply_ipp_2(db_session, center, synch_image, inst_list, hdf_data_dir,
                              n_intensity_bins, R0)
aia_disk.iit_data = aia_iit_list[0].iit_data
aia_disk.sunpy_meta = aia_disk.map.meta
aia_map = aia_disk.interp_to_map(R0=R0, map_x=full_map_x, map_y=full_map_y, interp_field="iit_data",
                                 nprocs=nprocs, tpp=tpp, p_pool=p_pool, no_data_val=-9999.0,
                                 helio_proj=True)


stb_disk = datatypes.read_euv_image(stb_path)
stb_date_string = stb_disk.info['date_string']
center = datetime.datetime.strptime(stb_date_string, "%Y-%m-%dT%H:%M:%S.%f")
synch_image = pd.Series(dict(data_id=0, date=stb_disk.map.date, instrument=stb_disk.info['instrument'],
                             wavelength=stb_disk.info['wavelength'], distance=stb_disk.info['distance'],
                             cr_lon=stb_disk.info['cr_lon'], cr_lat=stb_disk.info['cr_lat'],
                             cr_rot=stb_disk.info['cr_rot'], flag=0, time_of_download=stb_disk.map.date,
                             fname_raw="", fname_hdf=stb_rel)
                        ).to_frame().T
date_pd, stb_list, stb_iit_list, use_indices, methods_list, ref_alpha, ref_x = \
    apply_lbc_iit.apply_ipp_2(db_session, center, synch_image, ['EUVI-B', ], hdf_data_dir,
                              n_intensity_bins, R0)
stb_disk.iit_data = stb_iit_list[0].iit_data
stb_disk.sunpy_meta = stb_disk.map.meta
stb_map = stb_disk.interp_to_map(R0=R0, map_x=full_map_x, map_y=full_map_y, interp_field="iit_data",
                                 nprocs=nprocs, tpp=tpp, p_pool=p_pool, no_data_val=-9999.0,
                                 helio_proj=True)




