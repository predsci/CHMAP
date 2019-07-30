
import drms

date0 = '2012-07-12T16:00:00Z'

# convert a DRMS time to an astropy Time object, This can then be converted to tai
from astropy.time import Time
td = drms.to_datetime(date0)
print(td)
ta = Time(td, format='datetime', scale='utc')
print(ta)
tt = Time(ta, format='datetime', scale='tai')
print(tt)


from sunpy.time import parse_time
print(parse_time(date0.split('Z')[0]))

print(ta.tai)
print(parse_time(ta.isot))


# test out time differencing
d0='2012-07-12T11:00:00.000'
d1='2012-07-12T13:00:00.000'
t0=Time(d0,scale='tai')
t1=Time(d1,scale='tai')
delta_t=t1-t0
print()
print('-----------------------')
print(t0)
print(t1)
print(delta_t.sec)

# sunpy time ranges
from sunpy.time import TimeRange
time_range = TimeRange(t0, t1)
print(time_range.center)
tc=Time(time_range.center,scale='utc')
print(tc.mjd)
print(tc.datetime)
print(time_range.seconds)
print(time_range.next())

# example of windowing time ranges at a candence in sunpy (potentially very useful!)
import astropy.units as u
w=time_range.window(600*u.second,60*u.second)
for win in w:
    print(win.center)

"""
illustration of how I think I will use Astropy and Sunpy time objects
- start with a set of dates that spans your range of interest
- convert these to astropy time objects
- generate a sunpy time range that spans the whole interval
- divide up this interval into entries
"""
print('\n---------------------\n')
date_start='2012-07-01T00:00:00.000'
date_end  ='2012-08-01T00:00:00.000'
time_start=Time(date_start,scale='utc')
time_end=Time(date_end,scale='utc')
cadence = 2*u.hour
full_range=TimeRange(date_start, date_end)
time_ranges=full_range.window(cadence,cadence)
print(len(time_ranges))
#for time_range in time_ranges:
#    print(Time(time_range.start).isot)

time_start=Time( '2007-01-01T00:00:00.000', scale='utc')
time_end  =Time( '2018-01-01T00:00:00.000', scale='utc')
full_range=TimeRange(time_start, time_end)
time_ranges=full_range.window(cadence,cadence)
print(len(time_ranges))
print(time_start.datetime)
dt = time_end - time_start

# quick unit conversion example
print(dt.to(u.second))
