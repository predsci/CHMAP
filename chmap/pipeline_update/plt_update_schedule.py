

import matplotlib.pyplot as plt
import datetime

import chmap.settings.ch_pipeline_pars as pipe_pars

most_recent_image = 0
new_available_image = 28

lbcc_window = pipe_pars.LBCC_window_del.days
iit_window = pipe_pars.IIT_window_del.days

lbcc_hist_range = (most_recent_image, new_available_image)
lbcc_coef_range = (most_recent_image - lbcc_window/2, new_available_image - lbcc_window/2)
iit_hist_range = lbcc_coef_range
iit_coef_range = (lbcc_coef_range[0] - iit_window/2, lbcc_coef_range[1] - iit_window/2)

plot_range = (iit_coef_range[0] - 30, new_available_image + 25)
existing_maps = (plot_range[0], most_recent_image)
update_maps = (iit_coef_range[0], new_available_image)

plt_ticks = ("EUV Images", "LBCC hist", "LBCC coef",
             "IIT hist", "IIT coef", "Maps")
plt_y = (6, 5, 4, 3, 2, 1)

# plot updated bars
plt_left = (most_recent_image, lbcc_hist_range[0], lbcc_coef_range[0],
            iit_hist_range[0], iit_coef_range[0], update_maps[0])

plt_width = (new_available_image - most_recent_image,
             lbcc_hist_range[1] - lbcc_hist_range[0],
             lbcc_coef_range[1] - lbcc_coef_range[0],
             iit_hist_range[1] - iit_hist_range[0],
             iit_coef_range[1] - iit_coef_range[0],
             update_maps[1] - update_maps[0])

plt.barh(y=plt_y, width=plt_width, left=plt_left, color="forestgreen", tick_label=plt_ticks)

# plot existing bars
existing_left = [plot_range[0]]*6
existing_width = (most_recent_image - plot_range[0],
                  lbcc_hist_range[0] - plot_range[0],
                  lbcc_coef_range[0] - plot_range[0],
                  iit_hist_range[0] - plot_range[0],
                  iit_coef_range[0] - plot_range[0],
                  most_recent_image - plot_range[0])
plt.barh(y=plt_y, width=existing_width, left=existing_left, fill=False, hatch='/',
         linewidth=1., edgecolor="black")

plt.xlim(plot_range)
plt.xlabel("Days from current image")
plt.title("CHMAP Update Timelines")

# add a legend
colors = {'Updates': 'forestgreen', 'Existing': None}
hatchs = {'Updates': None, 'Existing': '/'}
fills = {'Updates': True, 'Existing': False}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label], hatch=hatchs[label],
                         fill=fills[label]) for label in labels]
plt.legend(handles, labels)

plt.tight_layout()

plt.savefig(fname="/Users/turtle/Dropbox/MyNACD/plots/pipeline_timeline/pipe_timeline.png", dpi=600)
