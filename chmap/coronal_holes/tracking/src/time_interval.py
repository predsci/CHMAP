""" Given a start and end time, how many images in the database are there in such interval?

Last Modified: July 19th, 2021 (Opal)"""

import os
import pickle
import chmap.database.db_funs as db_funcs


def get_number_of_frames_in_interval(curr_time, time_window, list_of_timestamps):
    """ Given a start and end time, how many images in the database are there in such interval?
    start time <= end time
    :param curr_time: datetime timestamp
    :param time_window: delta datetime
    :param list_of_timestamps: list of all timestamps in the datebase.
    :return: number of frames (integer)
    """
    # initialize the number of frames.
    ii = 0
    # get start time.
    start_time = curr_time - time_window
    for timestamp in list_of_timestamps:
        if start_time <= timestamp <= curr_time:
            ii += 1
    return ii


def time_distance(time_1, time_2):
    """ Compute the time distance between two timestamps, where time1 > time2.

    :param time_1: datetime timestamp.
    :param time_2: datetime timestamp.
    :return: time difference in hrs.
    """
    difference = time_1 - time_2
    # total seconds (int)
    total_seconds = difference.total_seconds()
    # convert from seconds to hrs.
    return total_seconds / (60 * 60)


def read_prev_run_pkl_results(ordered_time_stamps, prev_run_path):
    """ return a list of frames for matching history of previous run results.

    :param ordered_time_stamps: list of ordered timestamps with previous window pkl files.
    :param prev_run_path: path to previous run pickle files.
    :return: list of frames.
    """
    window_holder = [None] * len(ordered_time_stamps)
    # loop over all the ordered timestamps.
    for ii, timestamp in enumerate(ordered_time_stamps):
        # get pkl file name
        pickle_file = str(timestamp).replace(':', '-') + ".pkl"
        pickle_file = pickle_file.replace(" ", "-")
        # load pickle frame.
        window_holder[ii] = pickle.load(open(os.path.join(prev_run_path, pickle_file), "rb"))
    return window_holder


def get_time_interval_list(db_session, query_start, query_end, map_vars, map_methods):
    """ return a list of timestamps between two dates.

    :param db_session: database session.
    :param query_start: starttime timestamp.
    :param query_end: endtime timestamp.
    :param map_vars: map variables.
    :param map_methods: define map type and grid to query.
    :return: list of timestamps ordered.
    """
    # --- Begin execution ----------------------
    # query maps in time range
    map_info, data_info, method_info, image_assoc = db_funcs.query_euv_maps(
        db_session, mean_time_range=(query_start, query_end), methods=map_methods,
        var_val_range=map_vars)
    return map_info.date_mean