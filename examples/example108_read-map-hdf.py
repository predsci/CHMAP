"""
Open a map-hdf file into a PsiMap object
"""

import os

from settings.app import App
import modules.datatypes_v2 as psi_dt

# explicitly specify a map file
file_path = os.path.join(App.MAP_FILE_HOME, "single", "2014", "04", "13", "single_20140413T193506_MID3.h5")

# open the file
test_map = psi_dt.read_psi_map(file_path)

