"""Plot a list of coronal holes and its features
(centroid, straight and rotated bounding box, unique color, and unique ID) on one frame.

Last Modified: April 13th, 2021 (Opal)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_coronal_hole(ch_list, n_t, n_p, title, filename=False, plot_rect=True, plot_circle=True,
                      circle_radius=50, thickness_circle=1, thickness_rect=2, fontscale=0.3):
    """

    Parameters
    ----------
    ch_list: list of Contour object
    n_t: theta dimensions
    n_p: phi dimensions
    title: title of the plot
    filename: save file directory
    plot_rect: if True then we plot the bounding box.
    plot_circle: if True then we plot the circle relative to area size.
    circle_radius: default is 50. depends on the image dimensions.
    thickness_rect: default is 2. depends on the image dimensions.
    thickness_circle: default is 2. depends on the image dimensions.
    fontscale: default is 0.3. depends on the image dimensions.

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

        if plot_circle:
            # plot circle based on area.
            cv2.circle(img=final_image, center=(ch.pixel_centroid[1], ch.pixel_centroid[0]),
                       radius=int(ch.area*circle_radius), color=(255, 0, 0), thickness=thickness_circle)

        # check if its has multiple bounding boxes.
        if plot_rect:
            ii = 0
            while ii < len(ch.straight_box) / 4:
                # plot bounding box c.straight box returns top left x, y, w, h.
                cv2.rectangle(img=final_image, pt1=(ch.straight_box[4 * ii + 0], ch.straight_box[4 * ii + 1]),
                              pt2=(ch.straight_box[4 * ii + 0] + ch.straight_box[4 * ii + 2], ch.straight_box[4 * ii + 1] +
                                   ch.straight_box[4 * ii + 3]),
                              color=(0, 255, 0), thickness=thickness_rect)

                ii += 1

            # draw rotated box.
            if ii > 1:
                cv2.drawContours(final_image, [ch.rot_box_corners[:4, :]], 0, (0, 0, 255), thickness_rect)
                cv2.drawContours(final_image, [ch.rot_box_corners[4:, :]], 0, (0, 0, 255), thickness_rect)
            else:
                cv2.drawContours(final_image, [ch.rot_box_corners], 0, (0, 0, 255), thickness_rect)

        # plot the contour's ID number.
        cv2.putText(img=final_image, text=str(ch.id),
                    org=tuple(np.add((ch.pixel_centroid[1], ch.pixel_centroid[0]), (-15, 15))),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontscale, color=(0, 0, 0), thickness=1)

    # plot using matplotlib
    set_up_plt_figure(image=final_image, n_p=n_p, n_t=n_t, title=title, filename=filename)


def set_up_plt_figure(image, n_p, n_t, title, filename, cmap=None):
    """Set up proper axis labels and ticks, include title and save figure.

    Parameters
    ----------
    image: numpy array the image plotted.
    n_t: theta dimensions
    n_p: phi dimensions
    title: title of the plot
    cmap: color map
    filename: save file directory

    Returns
    -------
        N/A
    """
    fig, ax = plt.subplots()
    # plot using matplotlib.imshow function.
    ax.imshow(image, cmap=cmap, aspect=n_p/(2*n_t))

    # pixel coordinates + set ticks.
    p_pixel = np.linspace(0, n_p, 5)
    t_pixel = np.linspace(0, n_t, 5)

    ax.set_xticks(p_pixel)
    ax.set_xticklabels(["0", "$90$", "$180$", "$270$", "$360$"])
    ax.set_yticks(t_pixel)
    ax.set_yticklabels(["1", "$\dfrac{1}{2}$", "$0$", "-$\dfrac{1}{2}$", "-$1$"])

    # axis label.
    ax.set_xlabel("Longitude (Deg.)")
    ax.set_ylabel("Sin(Lat.)")

    # title of the image includes its frame number.
    ax.set_title(title)

    # save figure in filename.
    if filename is not False:
        plt.savefig(filename)

    plt.close()