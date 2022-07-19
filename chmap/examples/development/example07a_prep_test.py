"""
Here we illustrate manually prepping raw FITS files.
- This illustrates differences between the raw images, prepped images,
  and prepped images using deconvolution.
- The files are prepped and then saved to our custom format (a read is illustrated)
- Plots are saved as .png files for assesment/comparison.
- This also serves as a test of the remote deconvolution commands
  as well as SSW/IDL calls to secchi_prep (for EUVI)
"""
import os
import time

import sunpy.map

from chmap.settings.app import App
from chmap.utilities.plotting import euv_fits_plotting
import chmap.data.corrections.image_prep.prep as prep
from chmap.utilities.idl_connect import idl_helper
from chmap.utilities.datatypes import datatypes

# manually specify filenames
fitsfile_aia = App.APP_HOME + '/reference_data/aia_lev1_euv_12s_20140413T190507_193.fits'
fitsfile_sta = App.APP_HOME + '/reference_data/sta_euvi_20140413T190530_195.fits'
fitsfile_stb = App.APP_HOME + '/reference_data/stb_euvi_20140413T190609_195.fits'

# Write the data to a temporary folder
prep_home_dir = App.TMP_HOME

# write the plots to this folder
plot_dir = App.TMP_HOME

# flag to write the h5 format after prep
write = True


# quick function to produce plots of each permutation
def plot_examples(map_raw, map_nod, map_d, lmin, lmax, rawmin, rawmax, prefix):
    # color map to use for plotting 193 and 195 images
    cmap_name = 'sohoeit195'

    # limb alignment plots
    euv_fits_plotting.plot_alignment(map_raw, log_min=rawmin, log_max=rawmax, cmap_name=cmap_name,
                                     outfile=prefix + '_limb_raw.png')
    euv_fits_plotting.plot_alignment(map_nod, log_min=lmin, log_max=lmax, cmap_name=cmap_name,
                                     outfile=prefix + '_limb_no_deconv.png')
    euv_fits_plotting.plot_alignment(map_d, log_min=lmin, log_max=lmax, cmap_name=cmap_name,
                                     outfile=prefix + '_limb_deconv.png')

    # regular image plots, specify plot range in solar coordinates (PSI/MAS Style)
    xrange = [-1.1, 1.1]
    yrange = [-1.1, 1.1]
    euv_fits_plotting.plot_image_rs(map_raw, log_min=rawmin, log_max=rawmax, cmap_name=cmap_name,
                                    xrange=xrange, yrange=yrange, outfile=prefix + '_image_raw.png')
    euv_fits_plotting.plot_image_rs(map_nod, log_min=lmin, log_max=lmax, cmap_name=cmap_name,
                                    xrange=xrange, yrange=yrange, outfile=prefix + '_image_no_deconv.png')
    euv_fits_plotting.plot_image_rs(map_d, log_min=lmin, log_max=lmax, cmap_name=cmap_name,
                                    xrange=xrange, yrange=yrange, outfile=prefix + '_image_deconv.png')


# ----------------------------------------------------------------------
# Example 1: AIA
# ----------------------------------------------------------------------
# Read in the raw image
map_raw_aia = sunpy.map.Map(fitsfile_aia)

# Prep without deconvolution
subdir, fname, los_aia_nod = prep.prep_euv_image(fitsfile_aia, prep_home_dir, deconvolve=False, write=write)

# Prep WITH deconvolution
subdir, fname, los_aia = prep.prep_euv_image(fitsfile_aia, prep_home_dir, deconvolve=True, write=write)

# save the example plots
prefix = os.path.join(plot_dir, 'plot_aia')
lmin, lmax = 1.00, 3.25
plot_examples(map_raw_aia, los_aia_nod.map, los_aia.map, lmin, lmax, lmin + 0.3, lmax + 0.3, prefix)

# read in the deconvolved, prepped hdf5 file (.h5) as our custom image format
hdf_file = os.path.join(prep_home_dir, subdir, fname)
t1 = time.perf_counter()
los = datatypes.read_euv_image(hdf_file)
t2 = time.perf_counter()
print("time to read .h5 file and create LosImage with a map: ", t2 - t1)

# ----------------------------------------------------------------------
# Example 2: EUVI-A
# ----------------------------------------------------------------------
### NOTE: UNLESS SSW/IDL is setup for your system, the EUVI examples WILL NOT WORK
### ---> we need to figure out a solution for remote calls.

# start up an IDL session (used for SSW/IDL secchi_prep for STEREO A and B)
idl_session = idl_helper.Session()

# Read in the raw image
map_raw_sta = sunpy.map.Map(fitsfile_sta)

# Prep without deconvolution
subdir, fname, los_sta_nod = prep.prep_euv_image(fitsfile_sta, prep_home_dir, deconvolve=False, write=write,
                                                 idl_session=idl_session)

# Prep WITH deconvolution
subdir, fname, los_sta = prep.prep_euv_image(fitsfile_sta, prep_home_dir, deconvolve=True, write=write,
                                             idl_session=idl_session)

# save the example plots
prefix = os.path.join(plot_dir, 'plot_sta')
lmin, lmax = 1.00, 3.25
plot_examples(map_raw_sta, los_sta_nod.map, los_sta.map, lmin, lmax, 2.5, 3.8, prefix)

# close the IDL session
idl_session.end()

# ----------------------------------------------------------------------
# Example 3: EUVI-B
# ----------------------------------------------------------------------
# stbrt up an IDL session (used for SSW/IDL secchi_prep for STEREO A and B)
idl_session = idl_helper.Session()

# Read in the raw image
map_raw_stb = sunpy.map.Map(fitsfile_stb)

# Prep without deconvolution
subdir, fname, los_stb_nod = prep.prep_euv_image(fitsfile_stb, prep_home_dir, deconvolve=False, write=write,
                                                 idl_session=idl_session)

# Prep WITH deconvolution
subdir, fname, los_stb = prep.prep_euv_image(fitsfile_stb, prep_home_dir, deconvolve=True, write=write,
                                             idl_session=idl_session)

# save the example plots
prefix = os.path.join(plot_dir, 'plot_stb')
lmin, lmax = 1.00, 3.25
plot_examples(map_raw_stb, los_stb_nod.map, los_stb.map, lmin, lmax, 2.5, 3.8, prefix)

# close the IDL session
idl_session.end()
