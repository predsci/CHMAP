
import datetime

def round_day(raw_datetime):
    shifted_datetime = raw_datetime + datetime.timedelta(hours=12)
    concat_datetime = datetime.datetime(*shifted_datetime.timetuple()[:3])
    return concat_datetime


def round_hour(raw_datetime, n_hours):
    # generate a decimal number of hours for day of the timestamp
    hours_dec = raw_datetime.hour + raw_datetime.minute/60 + raw_datetime.second/3600 + \
                raw_datetime.microsecond/(1e6*3600)
    hours_div, hours_rem = divmod(hours_dec, n_hours)
    # determine the rounded number of hours
    if hours_rem >= n_hours/2:
        out_hours = int((hours_div+1)*n_hours)
    else:
        out_hours = int(hours_div*n_hours)
    # if the rounded number of hours is >= 24, add a day. Then generate new timestamp
    if out_hours > 23:
        add_days, out_hours = divmod(out_hours, 24)
        rounded_dt = raw_datetime.replace(day=raw_datetime.day + add_days, hour=out_hours, minute=0,
                                          second=0, microsecond=0)
    else:
        rounded_dt = raw_datetime.replace(hour=out_hours, minute=0, second=0, microsecond=0)

    return rounded_dt




