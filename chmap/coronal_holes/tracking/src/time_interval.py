""" Given a start and end time, how many images in the database are there in such interval?

Last Modified: July 16th, 2021 (Opal)"""


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
