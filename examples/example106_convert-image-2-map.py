"""
Query an image in DB, load it, and convert to a map.
"""

import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sunpy

from settings.app import App
import modules.DB_classes as db_class
from modules.DB_funs import init_db_conn, query_euv_images
import modules.datatypes as psi_d_types
import modules.Plotting as EasyPlot

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
query_time_min = datetime.datetime(2014, 4, 13, 19, 35, 0)
query_time_max = datetime.datetime(2014, 4, 13, 19, 37, 0)
query_pd = query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)

selected_image = query_pd.iloc[0]


# read hdf file to LOS object
hdf_file = os.path.join(hdf_data_dir, selected_image.fname_hdf)
test_los = psi_d_types.read_los_image(hdf_file)

# also read fits file for reference
fits_infile_aia = os.path.join(raw_data_dir, selected_image.fname_raw)

# Load images image using the built-in methods of SunPy
map_aia = sunpy.map.Map(fits_infile_aia)


# map parameters (input)
R0 = 1.01
y_range = [-1,1]
x_range = [0, 2*np.pi]

# map parameters (from image)
cr_lat = test_los.info['cr_lat']
cr_lon = test_los.info['cr_lon']

# determine number of pixels in map y-grid
map_nycoord = sum(abs(test_los.y) < R0)
del_y = (y_range[1] - y_range[0])/(map_nycoord - 1)
map_nxcoord = (np.floor((x_range[1] - x_range[0])/del_y) + 1).astype(int)

# generate map x,y grids. y grid centered on equator, x referenced from lon=0
map_y = np.linspace(y_range[0], y_range[1], map_nycoord.astype(int))
map_x = np.linspace(x_range[0], x_range[1], map_nxcoord.astype(int))

# test LosImage function interp_to_map()
test_map = test_los.interp_to_map(R0=R0, map_x=map_x, map_y=map_y, image_num=selected_image.image_id[0])

# compare test image to test map
EasyPlot.PlotImage(test_los, nfig=9)
EasyPlot.PlotMap(test_map, nfig=10)


# # determine number of pixels in map y-grid
# map_nycoord = sum(abs(test_los.y) < R0)
# del_y = (y_range[1] - y_range[0])/(map_nycoord-1)
# map_nxcoord = (np.floor((x_range[1] - x_range[0])/del_y) + 1).astype(int)
#
# # generate map x,y grids
# map_y = np.linspace(y_range[0], y_range[1], map_nycoord.astype(int))
# map_x = np.linspace(x_range[0], x_range[1], map_nxcoord.astype(int))
#
# # initialize grid to receive interpolation with values of -1
# interp_result = np.full((map_nycoord, map_nxcoord), np.nan)
#
# # generate as row and column vectors
# # arr_x, arr_y = np.meshgrid(map_x, map_y, sparse=True)
# # Or as matrices if that works better
# mat_x, mat_y = np.meshgrid(map_x, map_y)
#
# map_x_vec = mat_x.flatten()
# map_y_vec = mat_y.flatten()
#
# image_x, image_y, image_z, image_theta, image_phi = coord.map_grid_to_image(map_x_vec, map_y_vec, R0=R0, obsv_lon=cr_lon,
#                                                                             obsv_lat=cr_lat)
#
#
# # generate a sample intensity grid for plotting
# sample_grid = mat_x + mat_y
# # normalize to 0 to 1
# sample_grid = (sample_grid - sample_grid.min())/(sample_grid.max() - sample_grid.min())
# # for image plotting mask off the back half of the sphere
# image_x_vec = image_x
# image_y_vec = image_y
# image_z_vec = image_z
# sample_grid_vec = sample_grid.flatten()
# keep_ind = image_z_vec > 0
#
# # generate a line at equator
# map_equa_y = np.array([0, ]*map_nxcoord)
# map_equa_x = map_x
# # transform to image space
# equa_x, equa_y, equa_z, equa_theta, equa_phi = coord.map_grid_to_image(map_equa_x, map_equa_y, R0=R0, obsv_lon=cr_lon,
#                                                                             obsv_lat=cr_lat)
# # remove back-side equator points
# keep_equa = equa_z > 0
# sorted_order = equa_x[keep_equa].argsort()
# plot_equa_x = equa_x[keep_equa][sorted_order]
# plot_equa_y = equa_y[keep_equa][sorted_order]
#
# # create coords to plot observer location
# cr_map_x = cr_lon*np.pi/180
# cr_map_y = np.sin(cr_lat*np.pi/180)
#
# cr_x, cr_y, cr_z, cr_theta, cr_phi = coord.map_grid_to_image(cr_map_x, cr_map_y, R0=R0, obsv_lon=cr_lon, obsv_lat=cr_lat)
#
# # grab sunpy colormap for aia193
# # im_cmap = plt.get_cmap('sdoaia193')
# im_cmap = plt.get_cmap('sohoeit195')
#
# # lets see what it looks like
# plt.figure(0)
# plt.imshow(sample_grid, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
# plt.plot(map_equa_x, map_equa_y, color="black")
# plt.scatter(cr_map_x, cr_map_y, color="red")
# plt.xlabel(r'$\Phi$')
# plt.ylabel(r'$\sin(\zeta)$')
# plt.title("Test Map with observer (red) at CR_Lat: " + '%.2f' % cr_lat + " and CR_Lon: " + '%.2f' % cr_lon)
# plt.show()
#
# # try plotting in image coords
# # plt.scatter(image_x, image_y, c=sample_grid, cmap="viridis", s=10)
# plt.figure(1)
# plt.tricontourf(image_x_vec[keep_ind], image_y_vec[keep_ind], sample_grid_vec[keep_ind], cmap="viridis",
#                 levels=np.array(list(range(20)), dtype=np.float)/19)
# plt.plot(plot_equa_x, plot_equa_y, color="black")
# plt.scatter(cr_x, cr_y, color="red")
# ax1 = plt.gca()
# ax1.set_aspect('equal', adjustable='box')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Corresponding Image")
#

# # Now do linear interpolation between image and transformed map grid (on map-grid points that appear in image)
# test_fn = sp_interp.RegularGridInterpolator((test_los.x, test_los.y), test_los.data)
# eval_pts = np.array([image_y_vec[keep_ind], image_x_vec[keep_ind]]).transpose()
# interp_result_vec = interp_result.flatten(order='C')
#
# # Interp takes about 1.1s
# time_start = time.perf_counter()
# interp_result_vec[keep_ind] = test_fn(eval_pts)
# time_end = time.perf_counter()
# elapsed = time_end - time_start
#
# interp_result = interp_result_vec.reshape(interp_result.shape, order='C')
#
# import matplotlib as mpl
# norm = mpl.colors.LogNorm()
# # plot the initial image
# plt.figure(2)
# plt.imshow(test_los.data, extent=[test_los.x.min(), test_los.x.max(), test_los.y.min(), test_los.y.max()],
#            origin="lower", cmap=im_cmap, aspect="equal", norm=norm)


# # lets plot the resulting map
# plt.figure(3)
# plt.imshow(interp_result, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap=im_cmap,
#            aspect='equal', norm=norm)

# plot fits for reference
plt.figure(4)
map_aia.plot()
map_aia.draw_limb()
plt.colorbar()
plt.show()





# test conversion back to map coords



