"""
- Query database for raw LMSAL magentic flux maps
- Process to PSI-map objects (regular grid)
- Save as a map in the database
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd

from chmap.settings.app import App
import chmap.database.db_classes as db_class
import chmap.database.db_funs as db_funs
import chmap.maps.util.map_manip as map_manip
from chmap.maps.magnetic.lmsal_utils import project_lmsal_map
import chmap.utilities.datatypes.datatypes as psi_d_types

###### ------ PARAMETERS TO UPDATE -------- ########

# TIME RANGE
query_time_min = datetime.datetime(2012, 1, 7, 0, 0, 0)
query_time_max = datetime.datetime(2021, 1, 1, 0, 0, 0)

# Data_File query (in the form of a list)
file_type = ["magnetic map", ]
file_provider = ["lmsal", ]

# declare map parameters (?)
R0 = 1.01

# recover database paths
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
map_dir = App.MAP_FILE_HOME
# set path to raw mag-flux maps
raw_mag_dir = os.path.join(database_dir, 'raw_maps')

# designate which database to connect to
use_db = "mysql-Q"      # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
                        # 'mysql-Q_test' Use the development database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.

# Note if using the 'adaptive' style, requires sklearn
# style options = ['adaptive', 'local', 'fixed', 'smooth']
style = 'smooth'

# Build a rectilinear p,t mesh to project onto (currently only works for full-sun maps, see lmsal module)
npts_p = 400
pmin = 0
pmax = np.pi*2
pmap = np.linspace(pmin, pmax, npts_p)
# npts_t = 160
# tmin = 0
# tmax = np.pi
# tmap = np.linspace(tmin, tmax, npts_t)
npts_sinlat = 160
slmin = -1.
slmax = 1.
slmap = np.linspace(slmin, slmax, npts_sinlat)
tmap = np.flip(np.pi/2 - np.arcsin(slmap))
map_mesh = map_manip.MapMesh(pmap, tmap)

# Decide that your FWHM for projection should be SMOOTH but about 3.5 pixels at the equator and grow at the poles.
ds_fac = 2.0
arclength_fac = 2.0
n_neighbors = 10

dp_center = 2*np.pi/npts_p
fwhm_equator = ds_fac*np.sqrt(2)*dp_center
fwhm_pole = 4./3.*fwhm_equator
fwhm_fixed = fwhm_equator

# Establish connection to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base,
                                      sqlite_path=sqlite_path)
elif use_db in ('mysql-Q', 'mysql-Q_test'):
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base,
                                      user=user, password=password)

# Define methods for mag-map making
style_options = np.array(['adaptive', 'local', 'fixed', 'smooth'])
style_num = np.where(style_options == style)[0][0]
interp_method = {'meth_name': ["ProjFlux2Map"] * 7, 'meth_description':
                 ["Project irregular flux map to a rectilinear PsiMap"] * 7,
                 'var_name': ("style", "fwhm_equator", "fwhm_pole", "fwhm_fixed",
                              "ds_fac", "arclength_fac", "n_neighbors"),
                 'var_description': ("Projection style #", "FWHM equator", "FWHM pole",
                                     "FWHM fixed", "Local mesh spacing factor",
                                     "Arclength factor", "Number of neighbors"),
                 'var_val': (style_num, fwhm_equator, fwhm_pole, fwhm_fixed,
                             ds_fac, arclength_fac, n_neighbors)}
method_df = pd.DataFrame(interp_method)

# also record a method for map grid size
grid_method = {'meth_name': ("GridSize_sinLat", "GridSize_sinLat"), 'meth_description':
               ["Map number of grid points: phi x sin(lat)"] * 2, 'var_name': ("n_phi", "n_SinLat"),
               'var_description': ("Number of grid points in phi", "Number of grid points in sin(lat)"),
               'var_val': (npts_p, npts_sinlat)}
method_df = method_df.append(pd.DataFrame(grid_method))

# query data files
query_pd = db_funs.query_data_file(db_session=db_session, time_min=query_time_min,
                                   time_max=query_time_max, datatype=file_type,
                                   provider=file_provider)

# loop through files and project to a PsiMap
for file_index, row in query_pd.iterrows():
    sflux_file = os.path.join(raw_mag_dir, row.fname_raw)
    # first check that file exists
    if not os.path.exists(sflux_file):
        # test if raw_mag_dir is accessible
        dir_exists = os.path.isdir(raw_mag_dir)
        if not dir_exists:
            sys.exit("Root directory for raw magnetic flux maps does not exist"
                     "(or is not accessible).")
        # delete the record from the database
        print("LMSAL file does not exist for", row.date_obs, ". Deleting database record.")
        num_del = db_session.query(db_class.Data_Files).filter(
            db_class.Data_Files.data_id == row.data_id).delete()
        db_session.commit()
        # do not attempt to process the file
        continue


    print("Processing flux map for: ", row.date_obs)

    tstart = time.time()
    # Project the map
    br = project_lmsal_map(sflux_file, map_mesh, style=style, fwhm_equator=fwhm_equator,
                           fwhm_pole=fwhm_pole, fwhm_fixed=fwhm_fixed, ds_fac=ds_fac,
                           arclength_fac=arclength_fac, n_neighbors=n_neighbors)
    # print(f'### Time Taken for Projection: {time.time() - tstart:8.5f} seconds')
    #
    # # Look at the flux balance
    # pos, neg, err = get_map_flux_balance(br, map_mesh)
    # print('\n### Flux Information for the projected map:')
    # print(f'  Positive Flux: {pos}')
    # print(f'  Negative Flux: {neg}')
    # print(f'  Fractional Error: {err}')
    #
    # # Look at the average polar field above 60
    # pole_lat = 60.
    # br_north, br_south = get_polar_fields(br, map_mesh, latitude=pole_lat)
    # print(f'\n### Average Polar fields above {pole_lat:4.1f} degrees:')
    # print(f'  Br North: {br_north}')
    # print(f'  Br South: {br_south}')
    #
    #
    # width = 10.
    # plot_map(br, map_mesh, min=-width, max=width)

    # convert matrix to PSImap format
    br_2 = br.transpose()
    br_2 = np.flip(br_2, axis=0)

    # initialize PSImap object
    br_map = psi_d_types.PsiMap(data=br_2, x=pmap, y=slmap)

    # add methods
    br_map.append_method_info(method_df)
    # add data_info
    br_map.append_data_info(row)

    # construct map_info df to record basic map info
    map_info_df = pd.DataFrame(data=[[1, datetime.datetime.now()], ],
                               columns=["n_images", "time_of_compute"])
    br_map.append_map_info(map_info_df)

    # write map to database
    db_session = br_map.write_to_file(map_dir, map_type='flux',
                                      db_session=db_session)
    end_time = time.time()
    print(f' {end_time - tstart:8.5f} seconds elapsed processing flux map.')

db_session.close()

