




import os
import datetime
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from settings.app import App
import chmap.database.db_classes as db_class
from chmap.database.db_funs import init_db_conn, query_euv_images
import utilities.datatypes.datatypes as psi_d_types

# --- 1. Select Images -----------------------------------------------------
# In this example we use the 'reference_data' fits files supplied with repo
# manually set the data-file dirs
raw_data_dir = os.path.join(App.APP_HOME, "reference_data", "raw")
hdf_data_dir = os.path.join(App.APP_HOME, "reference_data", "processed")
# manually set the database location
database_dir = os.path.join(App.APP_HOME, "reference_data")
sqlite_filename = "dbtest.db"

# setup database connection
use_db = "sqlite"
sqlite_path = os.path.join(database_dir, sqlite_filename)

db_session = init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)


# query some images
query_time_min = datetime.datetime(2014, 4, 13, 0, 0, 0)
query_time_max = datetime.datetime(2014, 4, 14, 0, 0, 1)
query_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

print(query_pd.instrument)
# use these three images (one from each instrument)
selected_images = query_pd

# declare map and binning parameters
R0 = 1.01
mu_bin_edges = np.array(range(19), dtype="float") * 0.05 + 0.1
image_intensity_bin_edges = np.array(range(61), dtype="float") * .05 + 0.5
log10 = True
# image_intensity_bin_edges = np.array(range(4000), dtype="float")
# image_intensity_bin_edges = np.append(image_intensity_bin_edges, np.inf)
# log10 = False
lat_band = [-np.pi/64., np.pi/64.]

# read hdf file(s) to LOS objects and generate mu histograms
full_hist = np.full((len(mu_bin_edges)-1, len(image_intensity_bin_edges)-1), 0)
for index, row in selected_images.iterrows():
    hdf_path = os.path.join(hdf_data_dir, row.fname_hdf)
    los_temp = psi_d_types.read_los_image(hdf_path)
    # add coordinates to los object
    los_temp.get_coordinates(R0=R0)
    # perform 2D histogram on mu and image intensity
    temp_hist = los_temp.mu_hist(image_intensity_bin_edges, mu_bin_edges, lat_band=lat_band, log10=log10)
    # add this histogram to the log of histograms
    full_hist = np.add(full_hist, temp_hist)

# simple plot of raw histogram
plt.figure(0)
plt.imshow(full_hist, aspect="auto", interpolation='nearest', origin='low',
           extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-2]+1., mu_bin_edges[0], mu_bin_edges[-1]])
plt.xlabel("Pixel intensities")
plt.ylabel("mu")
plt.title("Raw 2D Histogram Data")


# Normalize each mu bin
norm_hist = np.full(full_hist.shape, 0.)
row_sums = full_hist.sum(axis=1, keepdims=True)
# but do not divide by zero
zero_row_index = np.where(row_sums != 0)
norm_hist[zero_row_index[0]] = full_hist[zero_row_index[0]]/row_sums[zero_row_index[0]]


# simple plot of normed histogram
plt.figure(1)
plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='low',
           extent=[image_intensity_bin_edges[0], image_intensity_bin_edges[-1], mu_bin_edges[0], mu_bin_edges[-1]])
plt.xlabel("Pixel intensities")
plt.ylabel("mu")
plt.title("2D Histogram Data Normalized by mu Bin")
