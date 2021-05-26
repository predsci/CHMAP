"""
Quick example for manually flagging bad images and/or outliers in the database

For now a *negative* number indicates a bad image.

We can use different flag numbers to indicate different things. Current values are:

  -1: Unclassified: Bad image, but no specific label has been determined (yet).
  -2: Missing Data: All or a portion of the sun is missing data. This is NOT FIXABLE.
  -3: AIA pointing error: The CRPIX values are "nan" from the master pointing series. This is POTENTIALLY FIXABLE.

NOTES:
- We may want to use positive value flags for "good" images that have special meaning.
  - Before doing so we will need to change things that look for flag==0 to flag=>0,
    and test (e.g. modules.DB_funs.query_euv_images).

HISTORY:
- This example was adapted by CD from commented code by TE in image_testing.py on 2021/01/04
"""
import pandas as pd
import datetime
from database.db_funs import init_db_conn, update_image_val, query_euv_images
from database.db_classes import Base, EUV_Images
from helpers.misc_helpers import print_full_dataframe

# setup the mysql session
use_db = "mysql-Q"
user = "cdowns"
password = ""
db_session = init_db_conn(db_name=use_db, chd_base=Base, user=user, password=password)

# columns to print when running the script (so i can read it...)
columns = ['image_id', 'date_obs', 'instrument', 'wavelength', 'flag', 'fname_raw', 'fname_hdf']

# OPTIONAL query a specific date to get image_ids of UNFLAGGED IMAGES (right now query_euv_images only looks for flag=0)
# set this to False if you want the script to continue
do_check_ids = False
if do_check_ids:
    period_start = datetime.datetime(2020, 2, 24, 1, 0, 0)
    period_end = datetime.datetime(2020, 2, 24, 3, 0, 0)
    query = query_euv_images(db_session=db_session, time_min=period_start, time_max=period_end,
                             instrument=('AIA', 'EUVI-A', 'EUVI-B'))

    # display ALL the info
    print_full_dataframe(query[columns])
    print()

    # now cause an exception to stop the script
    raise KeyboardInterrupt

# build a list of bad images (can be one or many IDs set manually as done here, or make a list from the query).
bad_images = [126544]

# set the flag that you'll apply to these images (assume that you are flagging similar types)
flag = -3

# Confirm that you actually want to do this
query_full = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.image_id.in_(tuple(bad_images))).statement,
                         db_session.bind)

print(f'\n### Will apply a new flag value of {flag}, to the following images:')
print_full_dataframe(query_full[columns])

if input(f"\n  Do you want to continue? [y/n]") == "y":
    # loop over image ids one by one since this is how update_image_val works
    for image_id in bad_images:
        query_out = pd.read_sql(db_session.query(EUV_Images).filter(EUV_Images.image_id == image_id).statement,
                                db_session.bind)
        print(f'  Flagging image: {query_out[columns].to_string(header=False, index=False)}')
        update_image_val(db_session, query_out, 'flag', flag)

else:
    print('  Aborted!')
