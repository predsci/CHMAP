
"""
- Load a synchronic EUV/CHD map and a magnetic flux map from the 'reference' directory
- Calculate the total coronal flux
"""


import os
import numpy as np

import utilities.datatypes.datatypes as psi_datatype
from settings.app import App
import maps.util.map_manip as map_manip
import matplotlib.pyplot as plt

# load reference maps
reference_dir = os.path.join(App.APP_HOME, 'reference_data')
ref_chd_map = "synchronic_20110812T180046_MID54953.h5"
ref_flux_map = "mag_flux_20110812T180506_MID126344.h5"
# both a coronal hole detection and a radial magnetic field map
chd_map = psi_datatype.read_psi_map(os.path.join(reference_dir, ref_chd_map))
br_map = psi_datatype.read_psi_map(os.path.join(reference_dir, ref_flux_map))

chd_flux = map_manip.chdmap_br_flux(br_map, chd_map)

# If instead, you want to evaluate for a single coronal hole
chd_array = np.zeros(br_map.data.shape)
# set coronal hole pixels to 1.0
chd_array[30:50, 30:50] = 1.0
plt.imshow(chd_array)

chd_flux2 = map_manip.chdarray_br_flux(br_map, chd_array, chd_no_data_val=-9999.)
