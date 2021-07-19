""" Given a start and end time, how many images in the database are there in such interval?

Last Modified: July 19th, 2021 (Opal)"""


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
    # convert from seconds to days.
    return total_seconds / (60 * 60 * 24)
