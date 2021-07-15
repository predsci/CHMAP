import numpy as np
import matplotlib.pyplot as plt
import pickle


def map_new_polar_projection(gray_image):
    """ A function to rotate a grayscaled image and project.
     The projection steps:
     1. transform to cartesian coordinates.
     2. rotate about the x axis by angle=pi/2:
            * rotation matrix = [1      0      0 ]   [1  0  0]
                                [0 cos(a) -sin(a)] = [0  0 -1]
                                [0 sin(a)  cos(a)]   [0  1  0]

    3. map back to spherical coordinates. - return image in new projection.

    :parameter gray_image = image matrix (n_t x n_p) dimensions.
    Gray scaled, meaning its elements are between 0 and 255. """
    # extract the dimensions of the grayscaled image.
    n_t, n_p = np.shape(gray_image)

    # create 1d arrays for spherical coordinates.
    theta = np.linspace(np.pi, 0, n_t)
    phi = np.linspace(0, 2 * np.pi, n_p)

    # spacing in theta and phi.
    delta_t = theta[1] - theta[0]
    delta_p = phi[1] - phi[0]

    # compute theta and phi grids.
    theta_grid = np.arccos(np.outer(np.sin(theta), np.sin(phi)))
    phi_grid = np.arctan2(np.outer(-np.cos(theta), np.ones(n_p)), np.outer(np.sin(theta), np.cos(phi)))

    # Change phi range from [-pi,pi] to [0,2pi]
    neg_phi = phi_grid < 0
    phi_grid[neg_phi] = phi_grid[neg_phi] + 2 * np.pi

    # initialize new image.
    image = np.zeros((n_t, n_p))

    # assign the new index.
    for ii in range(0, n_t):
        for jj in range(0, n_p):
            image[ii, jj] = gray_image[int(np.abs(theta_grid[ii, jj]) / delta_t), int(phi_grid[ii, jj] / delta_p)]
    return image


def map_back_to_long_lat(gray_image):
    """ A function to rotate a grayscaled image and project.
        The projection steps:
     1. transform to cartesian coordinates.
     2. rotate about the x axis by angle=-pi/2:
            * rotation matrix = [1      0      0 ]   [1  0  0]
                                [0 cos(a) -sin(a)] = [0  0  1]
                                [0 sin(a)  cos(a)]   [0 -1  0]

    3. map back to spherical coordinates. - return image in new projection.

    :parameter gray_image = image matrix (n_t x n_p) dimensions.
    Gray scaled, meaning its elements are between 0 and 255. """
    # extract the dimensions of the grayscaled image.
    n_t, n_p = np.shape(gray_image)

    # create 1d arrays for spherical coordinates.
    theta = np.linspace(np.pi, 0, n_t)
    phi = np.linspace(0, 2 * np.pi, n_p)

    # spacing in theta and phi.
    delta_t = theta[1] - theta[0]
    delta_p = phi[1] - phi[0]

    # compute theta and phi grids.
    theta_grid = np.arccos(np.outer(-np.sin(theta), np.sin(phi)))
    phi_grid = np.arctan2(np.outer(np.cos(theta), np.ones(n_p)), np.outer(np.sin(theta), np.cos(phi)))

    # Change phi range from [-pi,pi] to [0,2pi]
    neg_phi = phi_grid < 0
    phi_grid[neg_phi] = phi_grid[neg_phi] + 2 * np.pi

    # initialize new image.
    image = np.zeros((n_t, n_p))

    # assign the new index.
    for ii in range(0, n_t):
        for jj in range(0, n_p):
            image[ii, jj] = gray_image[int(np.abs(theta_grid[ii, jj]) / delta_t), int(phi_grid[ii, jj] / delta_p)]
    return image


if __name__ == '__main__':
    # load image from pickle file.
    image = pickle.load(file=open("example_vid/frame1.pkl", "rb"))

    n_t, n_p = np.shape(image)
    extent = [0, 2 * np.pi, 0, np.pi]

    singularity_lat_lon = np.zeros((n_t, n_p))

    t = np.linspace(np.pi, 0, n_t)
    p = np.linspace(0, 2*np.pi, n_p)

    for ii in range(0, n_t):
        if t[ii] > np.pi*3/4:
            singularity_lat_lon[ii, :] = 1
        elif t[ii] < np.pi/4:
            singularity_lat_lon[ii, :] = 1

    fig = plt.figure()
    ax = plt.axes()
    plt.imshow(image)

    # pixel coordinates + set ticks.
    p_pixel = np.linspace(0, n_p, 5)
    t_pixel = np.linspace(0, n_t, 5)

    plt.xticks(p_pixel, ["0", "$90$", "$180$", "$270$", "$360$"])
    plt.yticks(t_pixel, ["1", "$\dfrac{1}{2}$", "$0$", "-$\dfrac{1}{2}$", "-$1$"])

    # axis label.
    plt.xlabel("Longitude (Deg.)")
    plt.ylabel("Sin(Lat.)")

    ax.set_title('Original Image')

    singularity_polar = np.zeros((n_t, n_p))

    ind=255

    for ii in range(n_t):
        for jj in range(n_p):
            if np.sin(t[ii])*np.sin(p[jj]) > 1/2:
                singularity_polar[ii, jj] = (ind-ii-jj)
            elif np.sin(t[ii])*np.sin(p[jj]) < -1/2:
                singularity_polar[ii, jj] = (ind-ii-jj)


    singularity_polar =255* singularity_polar/ np.min(singularity_polar)

    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(singularity_polar, extent=extent, cmap='hsv')
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Polar projection distortion region in lat-lon projection ')


    fig = plt.figure()
    ax = plt.axes()
    pos = ax.imshow(map_new_polar_projection(gray_image=singularity_polar), extent=extent, cmap='hsv')
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Polar projection distortion region in polar projection ')
    plt.show()

