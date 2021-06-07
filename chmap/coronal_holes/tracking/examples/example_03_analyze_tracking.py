""" Analyze results of a test run saved from example_01_run_tracking.py
or example_02_run_tracking_db.py

This includes:

1. plotting CH features in time.

2. plotting + analyzing the connectivity graph.

Last Modified: May 10th, 2021 (Opal).
"""

import os
import datetime
import numpy as np
import cv2
from chmap.coronal_holes.tracking.tools.plots import set_up_plt_figure
import pickle
import matplotlib.pyplot as plt

Verbose = True
SaveVid = True

# ================================================================================================================
# Step 1: Choose a test case - time interval
# ================================================================================================================
# define map query start and end times
query_start = datetime.datetime(year=2011, month=2, day=17, hour=23, minute=0, second=0)
query_end = datetime.datetime(year=2011, month=10, day=27, hour=1, minute=0, second=0)

# ================================================================================================================
# Step 2: Initialize directory and folder to save results (USER PARAMETERS)
# ================================================================================================================
# --- User Parameters ----------------------
dir_name = "/Users/opalissan/desktop/CHT_RESULTS/"
folder_name = "2011-02-17-2011-10-27/"

# dimensions of the CH Maps
image_dims = [398, 644]

# ================================================================================================================
# Step 3: Read in Graph object.
# ================================================================================================================
graph_file_name = "connectivity_graph.pkl"

# read in graph object.
graph = pickle.load(open(os.path.join(dir_name, folder_name, graph_file_name), "rb"))

# print statistics about the graph size.
if Verbose:
    print("number of edges:", graph.G.number_of_edges())
    print("number of nodes:", graph.G.number_of_nodes())

# ================================================================================================================
# Step 3: Access a specific coronal hole and plots its attribute with respect to time.
# ================================================================================================================

# id of interest.
target_id = 4

# initialize the data
area = []
pca_tilt = []
sig_tilt = []
timestamp = []
frame_num = []


for node in graph.G.nodes:
    if graph.G.nodes[node]["id"] == target_id:
        if graph.G.nodes[node]["frame_num"] not in frame_num:
            # add area to the list.
            area.append(graph.G.nodes[node]["area"])
            # add timestamp as x axis values.
            timestamp.append(str(graph.G.nodes[node]["frame_timestamp"])[:10])
            # save the frame number appearance.
            frame_num.append(graph.G.nodes[node]["frame_num"])
            # open the pkl file with the same frame timestamp.
            frame_file_name = os.path.join(dir_name, folder_name, str(graph.G.nodes[node]["frame_timestamp"]) + ".pkl")
            frame = pickle.load(open(frame_file_name, "rb"))
            for ch in frame.contour_list:
                if ch.count == graph.G.nodes[node]["count"] and ch.id == graph.G.nodes[node]["id"]:
                    pca_tilt.append(ch.pca_tilt)
                    sig_tilt.append(ch.sig_tilt)
        else:
            area[frame_num.index(graph.G.nodes[node]["frame_num"])] += graph.G.nodes[node]["area"]

# plot the area results.
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(timestamp, area)
ax.set_xticks(ticks=timestamp[::5])
ax.set_xticklabels(labels=timestamp[::5], fontsize=8)
ax.tick_params(axis='x', labelrotation=45)
ax.set_ylabel("Area")
ax.set_title("Area of Coronal Hole #" + str(target_id))

# plot the tilt results.
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 7))
ax[0].scatter(timestamp, pca_tilt, c="r")
ax[0].set_xticks(ticks=timestamp[::5])
ax[0].set_xticklabels(labels=timestamp[::5], fontsize=8)
ax[0].tick_params(axis='x', labelrotation=45)
ax[0].set_ylabel("Tilt Angle (Deg)")

ax[1].scatter(timestamp, sig_tilt, c="g")
ax[1].set_xticks(ticks=timestamp[::5])
ax[1].set_xticklabels(labels=timestamp[::5], fontsize=8)
ax[1].tick_params(axis='x', labelrotation=45)
ax[1].set_ylabel("$\lambda_{1}/\lambda_{2}$ Significance of Tilt")
fig.suptitle("Tilt of Coronal Hole #" + str(target_id))
plt.tight_layout()
plt.show()


# ================================================================================================================
# Step 3: Create Video of a single Coronal Hole.
# ================================================================================================================
if SaveVid:
    # id of interest.
    target_id = 4

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dir_name + folder_name + "single_mov_ch" + str(target_id) + ".mov",
                            fourcc, 1, (640, 480))

    for node in graph.G.nodes:
        if graph.G.nodes[node]["id"] == target_id:
            # open the pkl file with the same frame timestamp.
            frame_file_name = os.path.join(dir_name, folder_name, str(graph.G.nodes[node]["frame_timestamp"]) + ".pkl")
            frame = pickle.load(open(frame_file_name, "rb"))
            for ch in frame.contour_list:
                if ch.count == graph.G.nodes[node]["count"] and ch.id == graph.G.nodes[node]["id"]:
                    image = np.zeros((image_dims[0], image_dims[1], 3), dtype=np.int32)
                    image[ch.contour_pixels_theta, ch.contour_pixels_phi] = ch.color
                    set_up_plt_figure(image=image, n_p=image_dims[1], n_t=image_dims[0],
                                      title=str(graph.G.nodes[node]["frame_timestamp"]),
                                      filename=os.path.join(dir_name, folder_name, "example01.png"))

                    img = cv2.imread(os.path.join(dir_name, folder_name, "example01.png"))
                    video.write(img)

    cv2.destroyAllWindows()
    video.release()


