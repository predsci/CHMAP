"""
Open a map-hdf file into a PsiMap object
"""

import os

import utilities.datatypes.datatypes as psi_dt
import MySQLdb

# explicitly specify a map file
#file_path = os.path.join(App.MAP_FILE_HOME, "single", "2014", "04", "13", "single_20140413T193506_MID3.h5")
file_path = os.path.join("/Users/tamarervin/work/CHD/reference_data/processed/2014/04/13/aia_lvl2_20140413T183506_193.h5")
# open the file
test_map = psi_dt.read_psi_map(file_path)

db = MySQLdb.connect(host="shadow.predsci.com",port=3306,passwd="ATlz8d40gh^W7Ge6",user="test_beta", db="CHD")
c=db.cursor()