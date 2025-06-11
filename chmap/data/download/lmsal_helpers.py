
import os
import requests
import tempfile
import h5py
import pandas as pd
from astropy.time import Time, TimeDelta


def download_lmsal_index(file_write):

    # specify index url
    down_url = "https://www.lmsal.com/solarsoft/pfss_genxcat/surffield_v2.h5"
    # download url contents
    print("Downloading current LMSAL index from ", down_url, "...\n", sep="")
    out_obj = requests.get(down_url)
    # write to file
    print("Download complete. Saving to file ", file_write)
    file_con = open(file_write, 'wb')
    file_con.write(out_obj.content)
    file_con.close()


def build_lmsal_index_path(filename=None, path=None):

    # Default path. database /tmp folder
    if path is None:
        path = tempfile.mkdtemp()
    # Default filename
    if filename is None:
        filename = "lmsal_index.h5"

    out_path = os.path.join(path, filename)
    return out_path


def read_lmsal_index(lmsal_filename, alter=1):
    """
    Read LMSAL .h5 catalog file and convert TAI seconds to UTC datetime.
    :param lmsal_filename: Path to file as a string
    :param alter: At time of function development (Oct 2020), the links in the catalog file
    do not work out-of-the-box.  alter=1 fixes the links. It is intended that alter=0 will
    be used if the catalog links are fixed, and alter=2 will be used if a new fix is required
    in the future.
    :return:
    """
    # read file
    lmsal_con = h5py.File(lmsal_filename, 'r')
    # unpack data
    lmsal_array = lmsal_con['ssw_pfss_database'][0]
    base_url = lmsal_array[0].decode("utf-8")

    lmsal_float_time = lmsal_array[1]
    lmsal_path = lmsal_array[2].astype(str)
    # lmsal_path = lmsal_path.tolist()

    # convert TAI seconds to a datetime format
    tai_epoch = Time("1958-01-01T00:00:00.000", scale="tai", format="isot")
    tai_datetime = tai_epoch + TimeDelta(lmsal_float_time, format="sec", scale="tai")

    # convert to UTC:
    utc_datetime = tai_datetime.utc
    # convert to datetime class. Because of the database, we generally want to work with the basic datetime class.
    out_datetime = utc_datetime.tt.datetime

    if alter == 1:
        # remove dashes and colons from lmsal_path
        for ii in range(lmsal_path.__len__()):
            lmsal_path[ii] = lmsal_path[ii].replace("-", "")
            lmsal_path[ii] = lmsal_path[ii].replace(":", "")
            lmsal_path[ii] = lmsal_path[ii].replace("T", "_")

        # add 'sfield_' to base_url
        base_url = base_url.replace("pfss_links_v2", "pfss_links_sfield_v2")

    # combine into dataframe (this is really slow)
    # out_frame = pd.DataFrame({'timestamp': utc_datetime, 'sub_path': lmsal_path})

    return base_url, out_datetime, lmsal_path


def query_lmsal_index(min_datetime, max_datetime=None):
    """
    Time-based query of lmsal synchronic magnetic maps.  When max_datetime is None, function looks for exact matche
    to min_datetime. Else, the function will search the interval min_datetime<= x <=max_datetime.
    :param min_datetime: a single datetime
    :param max_datetime: a single datetime
    :return: a pandas dataframe of datetimes and corresponding urls
    """
    # generate path to index file
    lmsal_index_path = build_lmsal_index_path()
    # check that index file exists
    file_exists = os.path.isfile(lmsal_index_path)
    if not file_exists:
        print("Synchronic magnetic map index downloading...\n")
        download_lmsal_index(lmsal_index_path)
    # read index file
    base_url, utc_datetime, lmsal_path = read_lmsal_index(lmsal_index_path)

    if max_datetime is None:
        max_datetime = min_datetime

    match_index = (utc_datetime >= min_datetime) & (utc_datetime <= max_datetime)
    # check for any matches
    if match_index.sum() == 0 and file_exists:
        print("No matches found. Updating synchronic magnetic map index file...")
        # if we used an existing file and the query had no matches, re-download the index
        download_lmsal_index(lmsal_index_path)
        base_url, utc_datetime, lmsal_path = read_lmsal_index(lmsal_index_path)
        # re-evaluate the match index
        match_index = (utc_datetime >= min_datetime) & (utc_datetime <= max_datetime)
        if match_index.sum() == 0:
            print("No matches found in updated file. Check datetime inputs.\n")

    out_dates = utc_datetime[match_index]
    out_url_stub = lmsal_path[match_index]

    out_url = [base_url + s for s in out_url_stub]

    out_df = pd.DataFrame({'datetime': out_dates, 'full_url': out_url})

    return out_df


