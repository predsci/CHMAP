"""Plot a single coronal hole and its features.

Author: Opal Issan, March 15th, 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_coronal_hole(ch_list, n_t, n_p, title, filename):
    """

    Parameters
    ----------
    ch_list: list of Contour object
    n_t: theta dimensions
    n_p: phi dimensions
    title: title of the plot
    filename: save file directory

    Returns
    -------
        None, saves plot to directory.
    """
    # initialize the image matrix.
    final_image = np.ones((n_t, n_p, 3), dtype=np.uint8) * 255

    for ch in ch_list:
        # save the coronal hole set of pixels.
        final_image[ch.contour_pixels_theta, ch.contour_pixels_phi, :] = ch.color

        # plot the contours center.
        cv2.circle(img=final_image, center=(ch.pixel_centroid[1], ch.pixel_centroid[0]),
                   radius=3, color=(0, 0, 0), thickness=-1)

        # check if its has multiple bounding boxes.
        ii = 0
        while ii < len(ch.straight_box) / 4:
            # plot bounding box c.straight box returns top left x, y, w, h.
            cv2.rectangle(img=final_image, pt1=(ch.straight_box[4 * ii + 0], ch.straight_box[4 * ii + 1]),
                          pt2=(ch.straight_box[4 * ii + 0] + ch.straight_box[4 * ii + 2], ch.straight_box[4 * ii + 1] +
                               ch.straight_box[4 * ii + 3]),
                          color=(0, 255, 0), thickness=2)
            ii += 1

        # draw rotated box.
        cv2.drawContours(final_image, [ch.rot_box_corners], 0, (0, 0, 255), 2)

    # plot using matplotlib
    set_up_plt_figure(image=final_image, n_p=n_p, n_t=n_t, title=title, filename=filename)


def set_up_plt_figure(image, n_p, n_t, title, filename):
    """Set up proper axis labels and ticks, include title and save figure.

    Parameters
    ----------
    image: numpy array the image plotted.
    n_t: theta dimensions
    n_p: phi dimensions
    title: title of the plot
    filename: save file directory

    Returns
    -------
        N/A
    """
    # plot using matplotlib.imshow function.
    plt.imshow(image)

    # pixel coordinates + set ticks.
    p_pixel = np.linspace(0, n_p, 5)
    t_pixel = np.linspace(0, n_t, 5)

    plt.xticks(p_pixel, ["0", "$90$", "$180$", "$270$", "$360$"])
    plt.yticks(t_pixel, ["1", "$\dfrac{1}{2}$", "$0$", "-$\dfrac{1}{2}$", "-$1$"])

    # axis label.
    plt.xlabel("Longitude (Deg.)")
    plt.ylabel("Sin(Lat.)")

    # title of the image includes its frame number.
    plt.title(title)

    # save figure in filename.
    if filename is not False:
        plt.savefig(filename)