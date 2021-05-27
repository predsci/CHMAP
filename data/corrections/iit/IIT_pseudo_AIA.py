

import numpy as np
import pandas as pd
import pickle
import datetime
from sqlalchemy.sql import func
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from settings.app import App
import database.db_funs as db_funs
import database.db_classes as db_class

# load IIT parameters and histogram stats
IIT_pars_file = '/Users/turtle/Dropbox/MyNACD/analysis/iit/IIT_pars-and-hists.pkl'

file = open(IIT_pars_file, 'rb')
iit_dict = pickle.load(file)
file.close()

# INITIALIZE DATABASE CONNECTION
# DATABASE PATHS
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME
raw_data_dir = App.RAW_DATA_HOME
hdf_data_dir = App.PROCESSED_DATA_HOME
image_out_path = "/Users/turtle/Dropbox/MyNACD/analysis/iit/"

# DATABASE PATHS
create = True  # true if save to database
database_dir = App.DATABASE_HOME
sqlite_filename = App.DATABASE_FNAME

# designate which database to connect to
use_db = "mysql-Q"       # 'sqlite'  Use local sqlite file-based db
                        # 'mysql-Q' Use the remote MySQL database on Q
user = "turtle"         # only needed for remote databases.
password = ""           # See example109 for setting-up an encrypted password.  In this case leave password="", and
# init_db_conn() will automatically find and use your saved password. Otherwise, enter your MySQL password here.


# ------------ NO NEED TO UPDATE ANYTHING BELOW  ------------- #

# connect to database
if use_db == 'sqlite':
    # setup database connection to local sqlite file
    sqlite_path = os.path.join(database_dir, sqlite_filename)

    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, sqlite_path=sqlite_path)
elif use_db == 'mysql-Q':
    # setup database connection to MySQL database on Q
    db_session = db_funs.init_db_conn(db_name=use_db, chd_base=db_class.Base, user=user, password=password)


# determine first AIA image
aia_query = db_session.query(func.min(db_class.EUV_Images.date_obs))\
    .filter(db_class.EUV_Images.instrument == "AIA")
aia_min_date = pd.read_sql(aia_query.statement, db_session.bind)
# determine first AIA LBC combo
aia_combo_query = db_session.query(func.min(db_class.Data_Combos.date_mean),
                                   func.max(db_class.Data_Combos.date_mean))\
    .filter(db_class.Data_Combos.instrument == "AIA", db_class.Data_Combos.meth_id == 1)
aia_combo_date = pd.read_sql(aia_combo_query.statement, db_session.bind)

stereoA_min = aia_combo_date.iloc[0, 0].to_pydatetime()
stereoA_max = aia_combo_date.iloc[0, 1].to_pydatetime()

stereoA_trim_min = datetime.datetime(2014, 9, 1, 0, 0, 0)
stereoA_trim_max = datetime.datetime(2015, 12, 1, 0, 0, 0)

stereoA_alpha = iit_dict['IIT_alpha'][:, 1]
stereoA_x = iit_dict['IIT_x'][:, 1]
moving_avg_centers = iit_dict['moving_avg_centers'].astype(datetime.datetime)

# trim pre-AIA dates as well as spotty 2015 data and current dates that do not have IIT yet
trim_index = (moving_avg_centers < stereoA_min) | (
    (moving_avg_centers >= stereoA_trim_min) & (moving_avg_centers <= stereoA_trim_max)\
    | (moving_avg_centers > stereoA_max)
)

alpha_use = stereoA_alpha[~trim_index]
x_use = stereoA_x[~trim_index]
dates_use = moving_avg_centers[~trim_index]
means_use = iit_dict['lbc_hist_mean'][~trim_index, 1]

# establish color palette
n_time = moving_avg_centers.shape[0]
color_dist = np.linspace(0., 1., n_time)
v_cmap = cm.get_cmap('viridis')
color_dist_use = color_dist[~trim_index]

# generate some color tick marks
time_range = moving_avg_centers.max() - moving_avg_centers.min()
tick_years = [datetime.datetime(2008, 1, 1, 0, 0),
              datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2014, 1, 1, 0, 0),
              datetime.datetime(2017, 1, 1, 0, 0), datetime.datetime(2020, 1, 1, 0, 0)]
date_diffs = [date - moving_avg_centers.min() for date in tick_years]
date_props = [days/time_range for days in date_diffs]

# plot scatter alpha vs mean
fig, ax = plt.subplots()

scatter_ax = ax.scatter(means_use, alpha_use, c=v_cmap(color_dist_use), s=15)
plt.ylabel("IIT alpha (Stereo A)")
plt.xlabel("Mean of 6-month processed histogram")
# plt.grid()

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = fig.colorbar(scatter_ax, cax=cax, label='Date', ticks=date_props)
cb.ax.set_yticklabels(["2008", "2011", "2014", "2017", "2020"])

# add linear regression
lin_alpha = stats.linregress(means_use, alpha_use)
# plot
plot_min = min(means_use)
plot_max = max(means_use)
line_x = np.linspace(plot_min, plot_max, num=10)
line_y = line_x * lin_alpha.slope + lin_alpha.intercept
line_obj = mpl.lines.Line2D(line_x, line_y, color="black")
ax.add_line(line_obj)

# use early Stereo-A IIT value to pin intercept
early_mean = means_use[0]
early_alpha = alpha_use[0]
new_alpha_intercept = early_alpha - lin_alpha.slope*early_mean
line_y2 = line_x * lin_alpha.slope + new_alpha_intercept
line_obj2 = mpl.lines.Line2D(line_x, line_y2, color="blue")
ax.add_line(line_obj2)
ax.scatter(early_mean, early_alpha, c="blue")

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="black"),
               mpl.lines.Line2D([0], [0], color="blue", marker="o")]
ax.legend(model_lines, ["Best Fit", "Pinned to Aug-2010"], loc='upper right')
# bbox_to_anchor=(1., 0.65), title="model"

# save and close
plot_fname = image_out_path + 'StereoA_LinFit_alpha.pdf'
plt.savefig(plot_fname)

fig.clf()
plt.close()


# plot scatter x vs mean
fig, ax = plt.subplots()

scatter_ax = ax.scatter(means_use, x_use, c=v_cmap(color_dist_use), s=15)
plt.ylabel("IIT x (Stereo A)")
plt.xlabel("Mean of 6-month processed histogram")
# plt.grid()

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = fig.colorbar(scatter_ax, cax=cax, label='Date', ticks=date_props)
cb.ax.set_yticklabels(["2008", "2011", "2014", "2017", "2020"])

# add linear regression
lin_x = stats.linregress(means_use, x_use)
# plot
plot_min = min(means_use)
plot_max = max(means_use)
line_x = np.linspace(plot_min, plot_max, num=10)
line_y = line_x * lin_x.slope + lin_x.intercept
line_obj = mpl.lines.Line2D(line_x, line_y, color="black")
ax.add_line(line_obj)

# use early Stereo-A IIT value to pin intercept
early_mean = means_use[0]
early_x = x_use[0]
new_x_intercept = early_x - lin_x.slope*early_mean
line_y2 = line_x * lin_x.slope + new_x_intercept
line_obj2 = mpl.lines.Line2D(line_x, line_y2, color="blue")
ax.add_line(line_obj2)
ax.scatter(early_mean, early_x, c="blue")

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="black"),
               mpl.lines.Line2D([0], [0], color="blue", marker="o")]
ax.legend(model_lines, ["Best Fit", "Pinned to Aug-2010"], loc='upper left')
# bbox_to_anchor=(1., 0.65), title="model"

# save and close
plot_fname = image_out_path + 'StereoA_LinFit_x.pdf'
plt.savefig(plot_fname)

plt.close()

# plot alpha for all instruments including approximated EUVI-A
sterA_means = iit_dict['lbc_hist_mean'][:, 1]
pseudo_alpha = sterA_means*lin_alpha.slope + lin_alpha.intercept
pseudo_x = sterA_means*lin_x.slope + lin_x.intercept

pseudo_alpha_shift = sterA_means*lin_alpha.slope + new_alpha_intercept
pseudo_x_shift = sterA_means*lin_x.slope + new_x_intercept

plot_max_index = 700

plt.figure()
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_alpha[0:plot_max_index], c="blue")
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_alpha_shift[0:plot_max_index], "green")
plt.plot(dates_use, alpha_use, c="red")
plt.ylabel("IIT alpha (Stereo A)")
plt.xlabel("Date (weekly)")

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="red"),
               mpl.lines.Line2D([0], [0], color="blue"),
               mpl.lines.Line2D([0], [0], color="green")]
plt.legend(model_lines, ["IIT", "Best Fit", "Pinned to Aug-2010"], loc='upper right')

# save and close
plot_fname = image_out_path + 'StereoA_alpha.pdf'
plt.savefig(plot_fname)

plt.close()

# plot x for all instruments including approximated EUVI-A
plt.figure()
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_x[0:plot_max_index], c="blue")
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_x_shift[0:plot_max_index], "green")
plt.plot(dates_use, x_use, c="red")
plt.ylabel("IIT x (Stereo A)")
plt.xlabel("Date (weekly)")
plt.grid()

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="red"),
               mpl.lines.Line2D([0], [0], color="blue"),
               mpl.lines.Line2D([0], [0], color="green")]
plt.legend(model_lines, ["IIT", "Best Fit", "Pinned to Aug-2010"], loc='upper right')

# save and close
plot_fname = image_out_path + 'StereoA_x.pdf'
plt.savefig(plot_fname)

plt.close()

# lets try solving alpha and x simultaneously to minimize mean and std diff
aia_means = iit_dict['lbc_hist_mean'][:, 0]
aia_std = iit_dict['lbc_hist_std'][:, 0]
sterA_std = iit_dict['lbc_hist_std'][:, 1]
n_dates = len(means_use)
A = np.ndarray((2*n_dates, 2))
b = np.ndarray((2*n_dates))

A[0:n_dates, 0] = sterA_means[~trim_index]
A[n_dates:, 0] = sterA_std[~trim_index]
A[0:n_dates, 1] = 1
A[n_dates:, 1] = 0

b[0:n_dates] = aia_means[~trim_index]
b[n_dates:] = aia_std[~trim_index]

solution = np.linalg.lstsq(A, b, rcond=None)

# same idea, second-order polynomial
A2 = np.ndarray((2*n_dates, 4))
b2 = np.ndarray((2*n_dates))

A2[0:n_dates, 0] = sterA_means[~trim_index]**2
A2[n_dates:, 0] = sterA_std[~trim_index]**2
A2[0:n_dates, 1] = sterA_means[~trim_index]
A2[n_dates:, 1] = sterA_std[~trim_index]
A2[0:n_dates, 2] = sterA_means[~trim_index]
A2[n_dates:, 2] = 0
A2[0:n_dates, 3] = 1
A2[n_dates:, 3] = 0

b2[0:n_dates] = aia_means[~trim_index]
b2[n_dates:] = aia_std[~trim_index]

solution2 = np.linalg.lstsq(A2, b2, rcond=None)
coefs2 = solution2[0]

aia_means2 = coefs2[0]*sterA_means**2 + coefs2[1]*sterA_means + coefs2[2]*sterA_means + coefs2[3]
aia_std2 = coefs2[0]*sterA_std**2 + coefs2[1]*sterA_std


# plot pseudo AIA mean vs actual mean
pseudo_aia_mean = sterA_means*pseudo_alpha + pseudo_x
pseudo_shift_aia_mean = sterA_means*pseudo_alpha_shift + pseudo_x_shift
aia_means[aia_means == 0.] = None

plt.figure()
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_aia_mean[0:plot_max_index], c="blue")
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_shift_aia_mean[0:plot_max_index], "green")
plt.plot(moving_avg_centers[0:plot_max_index], aia_means2[0:plot_max_index], "purple")
plt.plot(moving_avg_centers[0:plot_max_index], aia_means[0:plot_max_index], c="red")
plt.ylabel("AIA mean intensity (log10)")
plt.xlabel("Date (weekly)")
plt.grid()

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="red"),
               mpl.lines.Line2D([0], [0], color="blue"),
               mpl.lines.Line2D([0], [0], color="green"),
               mpl.lines.Line2D([0], [0], color="purple")]
plt.legend(model_lines, ["Post-LBC data", "Best Fit", "Pinned to Aug-2010", "Order-2 Poly"], loc='upper right')

# save and close
plot_fname = image_out_path + 'AIA_pseudo_mean.pdf'
plt.savefig(plot_fname)

plt.close()


# plot pseudo AIA std vs actual std
pseudo_aia_std = sterA_std*pseudo_alpha
pseudo_shift_aia_std = sterA_std*pseudo_alpha_shift
aia_std[aia_std == 0.] = None

plt.figure()
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_aia_std[0:plot_max_index], c="blue")
plt.plot(moving_avg_centers[0:plot_max_index], pseudo_shift_aia_std[0:plot_max_index], "green")
plt.plot(moving_avg_centers[0:plot_max_index], aia_std2[0:plot_max_index], c="purple")
plt.plot(moving_avg_centers[0:plot_max_index], aia_std[0:plot_max_index], c="red")
plt.ylabel("AIA intensity standard deviation (log10)")
plt.xlabel("Date (weekly)")
plt.grid()

# add a legend
model_lines = [mpl.lines.Line2D([0], [0], color="red"),
               mpl.lines.Line2D([0], [0], color="blue"),
               mpl.lines.Line2D([0], [0], color="green"),
               mpl.lines.Line2D([0], [0], color="purple")]
plt.legend(model_lines, ["Post-LBC data", "Best Fit", "Pinned to Aug-2010", "Order-2 Poly"], loc='upper left')

# save and close
plot_fname = image_out_path + 'AIA_pseudo_std.pdf'
plt.savefig(plot_fname)

plt.close()

# save resulting alpha and x time-series to a file
IIT_pars_file2 = '/Users/turtle/Dropbox/MyNACD/analysis/iit/IIT_pseudo-AIA_pars.pkl'

psuedo_AIA_dict = {"moving_avg_centers": moving_avg_centers, "fit_index": ~trim_index,
                   "pseudo_alpha": pseudo_alpha, "pseudo_x": pseudo_x,
                   "pseudo_alpha_shift": pseudo_alpha_shift, "pseudo_x_shift": pseudo_x_shift}

f = open(IIT_pars_file2, "wb")
pickle.dump(psuedo_AIA_dict, f)
f.close()
