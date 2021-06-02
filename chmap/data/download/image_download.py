"""
Functions to do common image downloads
"""

import pandas as pd
import os

import astropy.units as u

from chmap.data.download import drms_helpers, vso_helpers
from chmap.data.download.euv_utils import cluster_meth_1, list_available_images
from chmap.database.db_funs import add_image2session


def synchronic_euv_download(synch_times, raw_data_dir, db_session, download=True, overwrite=False, verbose=True):
    """

    :param synch_times: A pandas DataFrame with columns 'target_time', 'min_time', and 'max_time'. The function will
    look for images to best approximate a synchronic map at target_time, using only images that fall between min_time
    and max_time.
    :param raw_data_dir: Full path to where images are stored.
    :param db_session:
    :param download:
    :param overwrite:
    :param verbose:
    :return: pandas dataframe with results for each download.
                result -1: no image found in time range
                        0: image downloaded
                        1: download error
                        2: image file already exists (no download/overwrite)
                        3: image file already exists, but file overwritten
    """

    # query parameters
    interval_cadence = 2*u.hour
    aia_search_cadence = 12*u.second
    wave_aia = 193
    wave_euvi = 195
    n_inst = 3

    # initialize the jsoc drms helper for aia.lev1_euv_12
    s12 = drms_helpers.S12(verbose=True)

    # initialize the helper class for EUVI
    euvi = vso_helpers.EUVI(verbose=True)

    # initialize a pandas dataframe to store download results
    download_result = pd.DataFrame(columns=["instrument", "wavelength", "target_time", "min_time", "max_time",
                                            "result", "result_desc", "url", "local_path"])

    for index, row in synch_times.iterrows():
        time0 = row['target_time']
        jd0 = time0.jd
        print(time0)

        # add new rows to download results for each instrument (defaults to 'No image in time range')
        new_rows = pd.DataFrame({'instrument': ["AIA", "EUVI-A", "EUVI-B"],
                                 'wavelength': [wave_aia, wave_euvi, wave_euvi],
                                 'target_time': [time0]*n_inst, 'min_time': [row['min_time']]*n_inst,
                                 'max_time': [row['max_time']]*n_inst, 'result': [-1]*n_inst,
                                 'result_desc': ['No image in time range']*n_inst, 'url': [""]*n_inst,
                                 'local_path': [""]*n_inst})

        # query various instrument repos for available images.
        f_list = list_available_images(time_start=row['min_time'], time_end=row['max_time'],
                                       euvi_interval_cadence=interval_cadence, aia_search_cadence=aia_search_cadence,
                                       wave_aia=wave_aia, wave_euvi=wave_euvi)

        # check for empty instrument results or query errors (and remove)
        del_list = []
        for ii in range(len(f_list)):
            if len(f_list[ii]) == 0:
                # mark element for removal
                del_list.append(ii)
                # no images found in time range. do not attempt to download
            elif isinstance(f_list[ii], str) and f_list[ii] == "query_error":
                # mark element for removal
                del_list.append(ii)
                # record 'download error' in new_rows. do not attempt to download
                new_rows.iloc[ii, ['result', 'result_desc']] = [1, 'Image times query failed.']
        # keep f_list elements that are not in del_list
        f_list = [f_list[ii] for ii in range(len(f_list)) if ii not in set(del_list)]

        if f_list.__len__() == 0:
            print("No instrument images in time range.\n")
            # record download results 'No image in time range'
            download_result = download_result.append(new_rows)
            # skip to next iteration of temporal for loop
            continue

        # Now loop over all the image pairs to select the "perfect" group of images.
        imin = cluster_meth_1(f_list=f_list, jd0=jd0)

        # download and enter into database
        for ii in range(0, len(imin)):
            instrument = f_list[ii]['instrument'][0]
            spacecraft = f_list[ii]['spacecraft'][0]
            image_num = imin[ii][0]

            if download:
                if instrument == "AIA":
                    print("Downloading AIA: ", f_list[ii].iloc[image_num].time)
                    subdir, fname, download_flag = s12.download_image_fixed_format(f_list[ii].iloc[image_num],
                                                                    raw_data_dir, update=True,
                                                                    overwrite=overwrite, verbose=verbose)
                    results_index = 0
                elif instrument == "EUVI" and spacecraft == "STEREO_A":
                    print("Downloading EUVI A: ", f_list[ii].iloc[image_num].time)
                    subdir, fname, download_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num],
                                                                    raw_data_dir, compress=True,
                                                                    overwrite=overwrite, verbose=verbose)
                    results_index = 1
                elif instrument == "EUVI" and spacecraft == "STEREO_B":
                    print("Downloading EUVI B: ", f_list[ii].iloc[image_num].time)
                    subdir, fname, download_flag = euvi.download_image_fixed_format(f_list[ii].iloc[image_num],
                                                                    raw_data_dir, compress=True,
                                                                    overwrite=overwrite, verbose=verbose)
                    results_index = 2
                else:
                    print("Instrument ", instrument, " does not yet have a download function.  SKIPPING DOWNLOAD ")
                    continue

                # update download results
                if download_flag == 1:
                    # download failed. do not attempt to add to DB
                    new_rows.iloc[results_index, ['result', 'result_desc', 'url']] = [download_flag, 'Download failed',
                                                                                      f_list[ii].iloc[image_num].url]
                    continue
                elif download_flag == 0:
                    # download successful
                    new_rows.loc[results_index, ['result', 'result_desc', 'url', 'local_path']] = [download_flag,
                                                                'Image downloaded', f_list[ii].iloc[image_num].url,
                                                                os.path.join(raw_data_dir, subdir, fname)]
                elif download_flag == 2:
                    new_rows.loc[results_index, ['result', 'result_desc', 'url', 'local_path']] = [download_flag,
                                                                'Image file already exists. No download/overwrite',
                                                                f_list[ii].iloc[image_num].url,
                                                                os.path.join(raw_data_dir, subdir, fname)]
                elif download_flag == 3:
                    new_rows.loc[results_index, ['result', 'result_desc', 'url', 'local_path']] = [download_flag,
                                                                'Image file already exists. File overwritten',
                                                                f_list[ii].iloc[image_num].url,
                                                                os.path.join(raw_data_dir, subdir, fname)]

                # print("DEBUG: download_flag=", download_flag, ". Path: ", os.path.join(raw_data_dir, subdir, fname))
                # use the downloaded image to extract metadata and write a row to the database (session)
                db_session = add_image2session(data_dir=raw_data_dir, subdir=subdir, fname=fname, db_session=db_session)


        print("\nDownloads complete with all images added to DB session.  \nNow commit session changes to DB.\n")
        # commit the changes to the DB, this also assigns auto-incrementing prime-keys 'data_id'
        db_session.commit()
        # record download outcomes to dataframe
        download_result = download_result.append(new_rows)

    return download_result

