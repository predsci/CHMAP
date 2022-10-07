
####################################################################
# Centralized collection of CH-pipeline parameters
####################################################################

import datetime
import numpy as np

# define map interval cadence and width
map_freq = 2  # number of hours
interval_delta = 30  # number of minutes
del_interval_dt = datetime.timedelta(minutes=interval_delta)
del_interval = np.timedelta64(interval_delta, 'm')

# INSTRUMENTS
inst_list = ["AIA", "EUVI-A", "EUVI-B"]
# CORRECTION PARAMETERS
n_mu_bins = 18
n_intensity_bins = 200
R0 = 1.01
log10 = True
# AIA wavelength to pull degradation factor from
AIA_wave = 193

# DETECTION PARAMETERS
# region-growing threshold parameters
thresh1 = 0.95
thresh2 = 1.35
# consecutive pixel value
nc = 3
# maximum number of iterations
iters = 1000

# MINIMUM MERGE MAPPING PARAMETERS
del_mu = 0.6  # optional between this method and mu_merge_cutoff method (not both)
mu_cutoff = 0.0  # lower mu cutoff value
mu_merge_cutoff = None  # mu cutoff in overlap areas
EUV_CHD_sep = False  # Do separate minimum intensity merges for image and CHD

# MAP PARAMETERS
x_range = [0, 2 * np.pi]
y_range = [-1, 1]
reduce_map_nycoord = 640
reduce_map_nxcoord = 1600
full_map_nycoord = 2048
full_map_nxcoord = 2048*2
low_res_nycoord = 160
low_res_nxcoord = 400

# Time-averaging window full-width (LBCC)
LBCC_window = 180  # days for moving average
LBCC_window_del = datetime.timedelta(days=LBCC_window)
LBCC_weekday = 0  # start at 0 for Monday

LBCC_lat_band = [- np.pi / 64., np.pi / 64.]

# TIME WINDOW FOR IIT PARAMETER CALCULATION
IIT_weekday = 0  # start at 0 for Monday
IIT_number_of_days = 180  # days for moving average
IIT_window_del = datetime.timedelta(days=IIT_number_of_days)

# IIT instruments
IIT_ref_inst = "AIA"  # reference instrument to fit histograms to
IIT_query_wavelengths = [193, 195]
IIT_lat_band = [-np.pi / 2.4, np.pi / 2.4]
