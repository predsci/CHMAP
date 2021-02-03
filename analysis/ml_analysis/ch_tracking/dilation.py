""" How to apply morphological operators using opencv.
Here, erosion and dilation will help join disparate elements in an image. """
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.ch_db import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.projection import map_new_polar_projection, map_back_to_long_lat

# def apply_latitude_weighted_erosion()

if __name__ == "__main__":
    # read in a random image.
    image = pickle.load(file=open("example_vid/frame1.pkl", "rb"))

    # image dimensions.
    n_t, n_p = np.shape(image)

    # image division.
    nd = 10
    dt = int(np.shape(image)[0] / nd)

    # initialize new image.
    new_image = np.zeros((n_t, n_p), dtype=np.uint8) * 255
    nimage = np.zeros((n_t, n_p), dtype=np.uint8) * 255

    # kernel
    kernel = np.ones((3, 3), dtype=np.uint8)

    # erode the whole image to remove small detections.
    erimage = cv2.dilate(image, kernel, iterations=1)

    nimage[0:2, :] = np.min(image[0:2, :]) * np.ones(n_p)
    nimage[-3:-1, :] = np.min(image[-3:-1, :]) * np.ones(n_p)
    nimage[2:-3, :] = cv2.dilate(image[2:-3, :], kernel, iterations=1)

    for ii in range(nd):
        # compute number of iterations based on the latitude.
        dist = abs(int(nd / 2) - ii)
        # save erosion strip.
        new_image[dt * ii:dt * (ii + 1), :] = cv2.erode(nimage[dt * ii:dt * (ii + 1), :], kernel, iterations=dist)

    ret, thresh = cv2.threshold(new_image, CoronalHoleDB.BinaryThreshold, 255, 0)
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    idx = 20
    rbg = np.ones((n_t, n_p, 3), dtype=np.uint8) * 255
    for c in contours:
        cv2.drawContours(rbg, [c], contourIdx=0, color=np.random.randint(low=0, high=255, size=(3,)).tolist(),
                         thickness=cv2.FILLED)
        idx += 6

    # map back to each pixel in original image.
    # ret, thresh = cv2.threshold(image, CoronalHoleDB.BinaryThreshold, 255, 0)
    fimage = np.ones((n_t, n_p, 3), dtype=np.uint8) * 255
    for ii in range(n_t):
        for jj in range(n_p):
            if image[ii, jj] < CoronalHoleDB.BinaryThreshold:
                fimage[ii, jj, :] = rbg[ii, jj, :]

    extent = [0, 2 * np.pi, 0, np.pi]
    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(rbg, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Dilated classification')

    # image 1.
    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(erimage, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Uniform Erosion to Exclude False Detections')

    # image 2.
    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(image, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Input Image')

    # image 3.
    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(new_image, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Latitude Weighted Dilation')

    # image 4.
    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(fimage, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Final Image')

    plt.show()
