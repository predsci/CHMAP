"""
Example script that illustrates routines that project an LMSAL model snapshot to phi & theta.

Here we build a map mesh and project an sflux snapshot to it to determine Br on the full-sun.
"""

import numpy as np
import os
import time

# workaround for circular import issue between datatypes and coord manip

from settings.app import App
from maps.magnetic.lmsal_utils import project_lmsal_map, get_map_flux_balance, get_polar_fields, plot_map
from maps.util.map_manip import MapMesh
from utilities.file_io.psi_hdf import wrh5_meta

# Reference file parameters
sflux_file = os.path.join(App.APP_HOME, 'reference_data', 'kitrun076_20190617_120400.h5')

# Output folder for the test map
output_dir = App.TMP_HOME

# Build a rectilinear p,t mesh to project onto (currently only works for full-sun maps, see lmsal module)
npts = 360
pmin = 0
pmax = np.pi*2
tmin = 0
tmax = np.pi
pmap = np.linspace(pmin, pmax, npts + 1)
tmap = np.linspace(tmin, tmax, int(npts/2) + 1)
map_mesh = MapMesh(pmap, tmap)

# Decide that your FWHM for projection should be SMOOTH but about 3.5 pixels at the equator and grow at the poles.
ds_fac = 2.0
arclength_fac = 2.0
n_neighbors = 10

dp_center = 2*np.pi/npts
fwhm_equator = ds_fac*np.sqrt(2)*dp_center
fwhm_pole = 4./3.*fwhm_equator
fwhm_fixed = fwhm_equator

print(f'\n### FWHM at the equator: {fwhm_equator}')

# Note if trying ALL styles, the 'adaptive' version requires sklearn
# styles = ['adaptive', 'local', 'fixed', 'smooth']
styles = ['smooth']

for style in styles:
    print(f'\n### Working on projection for style: {style}')
    tstart = time.time()

    # Project the map
    br = project_lmsal_map(sflux_file, map_mesh, style=style, fwhm_equator=fwhm_equator, fwhm_pole=fwhm_pole,
                           fwhm_fixed=fwhm_fixed, ds_fac=ds_fac, arclength_fac=arclength_fac, n_neighbors=n_neighbors)
    print(f'### Time Taken for Projection: {time.time() - tstart:8.5f} seconds')

    # Look at the flux balance
    pos, neg, err = get_map_flux_balance(br, map_mesh)
    print('\n### Flux Information for the projected map:')
    print(f'  Positive Flux: {pos}')
    print(f'  Negative Flux: {neg}')
    print(f'  Fractional Error: {err}')

    # Look at the average polar field above 60
    pole_lat = 60.
    br_north, br_south = get_polar_fields(br, map_mesh, latitude=pole_lat)
    print(f'\n### Average Polar fields above {pole_lat:4.1f} degrees:')
    print(f'  Br North: {br_north}')
    print(f'  Br South: {br_south}')

    width = 10.
    plot_map(br, map_mesh, min=-width, max=width)

    output_file = os.path.join(output_dir, 'br_lmsal_test_map.h5')
    print(f'\n### Writing map to {output_file}')
    wrh5_meta(output_file, map_mesh.p, map_mesh.t, np.array([]), np.transpose(br))
