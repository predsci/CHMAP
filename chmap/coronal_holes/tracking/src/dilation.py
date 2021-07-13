"""Apply Dilation on CH Maps.

Last Modified: May 6th, 2021 (Opal)
"""
import cv2
import numpy as np
from chmap.coronal_holes.tracking.src.contour import Contour


def get_kernel_width(t, gamma, n_p):
    """The dilation kernel width based on latitude.

    Parameters
    ----------
    t: float
        theta latitude in [0, pi]

    gamma: int
        constant param of kernel width at the equator.

    n_p: int
        number of pixels in longitude.

    Returns
    -------
        kernel width: int
    """
    # piecewise function.
    alpha = np.arcsin(gamma / n_p)
    # due to symmetry.
    beta = np.pi - alpha
    # loop over each interval.
    if alpha < t < beta:
        return int(gamma / np.sin(t))
    elif 0 <= t <= alpha:
        return n_p
    elif beta <= t <= np.pi:
        return n_p
    else:
        raise Exception("latitude value is invalid.")


def latitude_weighted_dilation(grey_scale_image, theta, gamma, n_p):
    """Latitude weighted dilation on EUV Images.
    TODO: optimize.

    Parameters
    ----------
    theta:
        (numpy array) theta coordinate numpy.linspace(0, pi, n_t)

    gamma:
        (int) dilation hyper parameter.

    n_p:
        (int) number of phi (longitude) pixels.

    grey_scale_image:
            (numpy array) grey scaled image or binary image

    Returns
    -------
        dilated_image:
            (numpy array)
    """
    # create copy of greyscaled_image
    dilated_image = np.zeros(grey_scale_image.shape, dtype=np.uint8)

    # latitude weighted dilation.
    for ii in range(len(theta)):
        # build the flat structuring element.
        width = get_kernel_width(t=theta[ii], gamma=gamma, n_p=n_p)
        kernel = np.ones(width, dtype=np.uint8)
        # save dilated strip.
        dilated_image[ii, :] = np.reshape(cv2.dilate(grey_scale_image[ii, :], kernel, iterations=1), n_p)
    return dilated_image


def uniform_dilation_in_latitude(image, beta=8):
    """Uniform dilation in latitude - kernel height is the size of "beta" (int).

    Parameters
    ----------
    beta: (int) structuring element height.
            Default is 8 pixels (this should change depending on the size of the input image.
    image: (numpy array) greyscale image.

    Returns
    -------
        (numpy array) dilated image.
    """
    # initialize the kernel size.
    kernel = np.ones(beta, dtype=np.uint8)
    # dilate in latitude.
    return cv2.dilate(image, kernel=kernel.T, iterations=1)


def generate_ch_color():
    """generate a random color

    Returns
    -------
    list of 3 integers between 0 and 255.
    """
    return np.random.randint(low=0, high=255, size=(3,)).tolist()


def plot_dilated_contours(contours, Mesh):
    """Draw filled contours of dilated greyscale input image.

    Parameters
    ----------
    contours: opencv contours.
    Mesh: MapMesh object.

    Returns
    -------
    rbg: image where each contour has a unique color
    color_list: list of unique contour colors.
    """
    # initialize RBG image.
    rbg = np.zeros((Mesh.n_t, Mesh.n_p, 3), dtype=np.uint8)

    # initialize contour color list.
    color_list = np.zeros((len(contours), 3))

    # draw contours on rbg.
    for ii, contour in enumerate(contours):
        color_list[ii] = generate_ch_color()
        cv2.drawContours(image=rbg, contours=[contour], contourIdx=0, color=color_list[ii],
                         thickness=cv2.FILLED)
    return rbg, color_list.astype(int)


def find_contours(image, thresh, Mesh):
    """Find contours contours of a greyscale dilated image.

    Parameters
    ----------
    image:
        (numpy array) gray scaled image.

    thresh:
        (float) binary threshold for contours.

    Mesh:
        MapMesh object.

    Returns
    -------
        rbg image
        list of unique colors.
    """
    # create binary threshold.
    ret, thresh = cv2.threshold(image, thresh, 1, 0)
    # find contours using opencv function.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    # draw contours.
    return plot_dilated_contours(contours=contours, Mesh=Mesh)


def get_list_of_contours_from_rbg(rbg_image, color_list, Mesh, frame_num=0, frame_timestamp=None):
    """Save all the image pixel coordinates that are assigned to each coronal hole
    (along with other coronal hole features- such as area, centroid, etc...).

    Parameters
    ----------
    Mesh: MeshMap() object with image coordinates.
    rbg_image: rbg lon-lat classified coronal hole image.
    color_list: list of contour unique colors.
    frame_timestamp: frame time stamp, default is None.
    frame_num: frame id number, default is 0.

    Returns
    -------
    coronal_hole_list : coronal hole list of Contour object.
    """
    # initialize list of contours.
    coronal_hole_list = []

    # loop over each contour saved.
    for color in color_list:
        # save pixel locations.
        mask = np.all(rbg_image == color, axis=-1)
        # find image pixel coordinates.
        contour_pixel = np.asarray(np.where(mask))
        # save contour in a list if its not zero.
        coronal_hole_list.append(Contour(contour_pixels=contour_pixel,
                                         frame_num=frame_num,
                                         frame_timestamp=frame_timestamp,
                                         Mesh=Mesh))
    return coronal_hole_list
