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
    ax.imshow(singularity_lat_lon, extent=extent)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\Theta$")
    ax.set_title('Lat-lon distortion region ')


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

    # new_image = map_new_polar_projection(gray_image=image)
    # new_new_image = map_back_to_long_lat(gray_image=new_image)
    #
    # # extent=[0, 2 * np.pi, 0, np.pi]
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.imshow(image)
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('Original Map ')
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.imshow(new_image)
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('Rotate $\pi$/2')
    # plt.show()
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.imshow(new_new_image)
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('Rotate -$\pi$/2')
    # plt.show()
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # im = ax.imshow(new_new_image - image)
    # fig.colorbar(im, orientation='horizontal')
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('diff')
    # plt.show()

    # # create 1d arrays for spherical coordinates.
    # t = np.linspace(0, np.pi, np.shape(image)[0])
    # p = np.linspace(0., 2 * np.pi, np.shape(image)[1])

    # # spacing in theta and phi.
    # dt = t[1] - t[0]
    # dp = p[1] - p[0]
    #
    # # initialize new coordinates.
    # new_t = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_p = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_image = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
    #
    # for i in range(0, len(t)):
    #     for j in range(0, len(p)):
    #         new_t[i, j] = np.arccos(np.sin(t[i]) * np.sin(p[j]))
    #         new_p[i, j] = np.arctan2(-np.cos(t[i]), np.sin(t[i]) * np.cos(p[j]))
    #
    #         # arctan2 output is between -pi and pi --> move to 0 to 2*pi
    #         if new_p[i, j] <= 0:
    #             new_p[i, j] += 2 * np.pi
    #
    #         new_image[i, j, :] = image[-int(new_t[i, j] / dt), int(new_p[i, j] / dp), :]
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.imshow(image, extent=[0, 2 * np.pi, 0, np.pi])
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('Projected Map CH classification')
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.imshow(new_image.astype('uint8'), extent=[0, 2 * np.pi, 0, np.pi])
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\Theta$")
    # ax.set_title('Original Map')
    # plt.show()
    #
    # pickle.dump(new_image, file=open("example_vid/new_projection.pkl", "wb"))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # # get map 3D cartesian coords
    # x = np.outer(np.sin(t), np.cos(p))
    # y = np.outer(np.sin(t), np.sin(p))
    # z = np.outer(np.cos(t), np.ones(len(p)))
    # ax.scatter(x, y, z, c=image, cmap='Greys', linewidth=0.5)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    # ax.set_title("cartesian coordinates")
    # plt.show()

    # # rotate theta (about x-axis. observer theta goes to +z)
    # rotation_matrix = rotation_matrix(alpha=3*np.pi/2)
    #
    # new_x = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_y = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_z = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    #
    # for i in range(0, len(t)):
    #     for j in range(0, len(p)):
    #         vec = np.array([x[i, j], y[i, j], z[i, j]])
    #         res = np.dot(rotation_matrix, vec)
    #         new_x[i, j], new_y[i, j], new_z[i, j] = res[0], res[1], res[2]
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(new_x, new_y, new_z, c=new_image, cmap='Greys', linewidth=0.5)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    # ax.set_title("Post Rotation")
    # plt.show()
    # rotation_matrix1 = rotation_matrix(alpha=np.pi / 2)
    #
    # new_x = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_y = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    # new_z = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    #
    # for i in range(0, len(t)):
    #     for j in range(0, len(p)):
    #         vec = np.array([x[i, j], y[i, j], z[i, j]])
    #         res = np.dot(rotation_matrix1, vec)
    #         new_x[i, j], new_y[i, j], new_z[i, j] = res[0], res[1], res[2]
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(new_x, new_y, new_z, c=image, cmap='Greys', linewidth=0.5)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    # ax.set_title("Post Rotation")
    #
    # rotation_matrix2 = rotation_matrix(-np.pi / 2)
    #
    # for i in range(0, len(t)):
    #     for j in range(0, len(p)):
    #         vec = np.array([new_x[i, j], new_y[i, j], new_z[i, j]])
    #         res = np.dot(rotation_matrix2, vec)
    #         new_x[i, j], new_y[i, j], new_z[i, j] = res[0], res[1], res[2]
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(new_x, new_y, new_z, c=image, cmap='Greys', linewidth=0.5)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    # ax.set_title("Post Post Rotation")
    # plt.show()
    #
    # # map back...
    # r = np.sqrt(np.square(new_x) + np.square(new_y) + np.square(new_z))
    # theta = np.arctan2(np.sqrt(np.square(new_x) + np.square(new_y)), new_z)
    # phi = np.arctan2(new_y, new_x)

    # @staticmethod
    # def map_new_polar_projection(gray_image):
    #     """ A function to rotate a grayscaled image and project.
    #      The projection steps:
    #      1. transform to cartesian coordinates.
    #      2. rotate about the x axis by angle=pi/2:
    #             * rotation matrix = [1      0      0 ]   [1  0  0]
    #                                 [0 cos(a) -sin(a)] = [0  0 -1]
    #                                 [0 sin(a)  cos(a)]   [0  1  0]
    #
    #     3. map back to spherical coordinates. - return image in new projection.
    #
    #     :parameter gray_image = image matrix (n_t x n_p) dimensions.
    #     Gray scaled, meaning its elements are between 0 and 255. """
    #     # extract the dimensions of the grayscaled image.
    #     n_t, n_p = np.shape(gray_image)
    #
    #     # create 1d arrays for spherical coordinates.
    #     theta = np.linspace(np.pi, 0, n_t)
    #     phi = np.linspace(0, 2 * np.pi, n_p)
    #
    #     # spacing in theta and phi.
    #     delta_t = theta[1] - theta[0]
    #     delta_p = phi[1] - phi[0]
    #
    #     # compute theta and phi grids.
    #     theta_grid = np.arccos(np.outer(np.sin(theta), np.sin(phi)))
    #     phi_grid = np.arctan2(np.outer(-np.cos(theta), np.ones(n_p)), np.outer(np.sin(theta), np.cos(phi)))
    #
    #     # Change phi range from [-pi,pi] to [0,2pi]
    #     neg_phi = phi_grid < 0
    #     phi_grid[neg_phi] = phi_grid[neg_phi] + 2 * np.pi
    #
    #     # initialize new image.
    #     image = np.zeros((n_t, n_p))
    #
    #     # assign the new index.
    #     for ii in range(0, n_t):
    #         for jj in range(0, n_p):
    #             image[ii, jj] = gray_image[int(np.abs(theta_grid[ii, jj]) / delta_t), int(phi_grid[ii, jj] / delta_p)]
    #     return image.astype(np.uint8)
    #
    # @staticmethod
    # def map_back_to_long_lat_rbg(input_image):
    #     """ A function to rotate a grayscaled image and project.
    #         The projection steps:
    #      1. transform to cartesian coordinates.
    #      2. rotate about the x axis by angle=-pi/2:
    #             * rotation matrix = [1      0      0 ]   [1  0  0]
    #                                 [0 cos(a) -sin(a)] = [0  0  1]
    #                                 [0 sin(a)  cos(a)]   [0 -1  0]
    #
    #     3. map back to spherical coordinates. - return image in new projection.
    #
    #     :parameter input_image = image matrix (n_t x n_p) dimensions.
    #     Gray scaled, meaning its elements are between 0 and 255. """
    #     # extract the dimensions of the grayscaled image.
    #     n_t, n_p, n_c = np.shape(input_image)
    #
    #     # create 1d arrays for spherical coordinates.
    #     theta = np.linspace(np.pi, 0, n_t)
    #     phi = np.linspace(0, 2 * np.pi, n_p)
    #
    #     # spacing in theta and phi.
    #     delta_t = theta[1] - theta[0]
    #     delta_p = phi[1] - phi[0]
    #
    #     # compute theta and phi grids.
    #     theta_grid = np.arccos(np.outer(-np.sin(theta), np.sin(phi)))
    #     phi_grid = np.arctan2(np.outer(np.cos(theta), np.ones(n_p)), np.outer(np.sin(theta), np.cos(phi)))
    #
    #     # Change phi range from [-pi,pi] to [0,2pi]
    #     neg_phi = phi_grid < 0
    #     phi_grid[neg_phi] = phi_grid[neg_phi] + 2 * np.pi
    #
    #     # initialize new image.
    #     image = np.zeros((n_t, n_p, n_c))
    #
    #     # assign the new index.
    #     for ii in range(0, n_t):
    #         for jj in range(0, n_p):
    #             image[ii, jj, :] = input_image[int(np.abs(theta_grid[ii, jj]) / delta_t),
    #                                int(phi_grid[ii, jj] / delta_p), :]
    #
    #     return image.astype(np.uint8)
