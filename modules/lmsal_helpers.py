"""
Module for working with LMSAL flux transport model snapshots.

The main purpose is to project the Lagrangian Fluxes to a Br map on a phi/theta grid.

Basic Overview:
- The flux elements are projected using 2D Gaussian function of arc distance to each pixel.
- The length scale (FWHM/sigma) is chosen based on the projection strategy.

Inputs
- Raw LMSAL SFT data
- Map mesh to project to (rectilinear grid in phi and theta, can be non-uniform).

Algorithm
- 1) Read the LMSAL data
- 2) Build a mesh for projection (see the MapMesh class)
- 3) Prep the lMSAL data (remove zeros, determine cartesian locations of flux elements, scale the fluxes to end up with Br in Gauss)
- 4) Pick a projection strategy to determine the Gaussian sigma for each flux element, currently these are four options.
- 5) For each flux element determine the cell indexes of pixels within N sigmas of the flux elements. This means we don’t need to compute the Gaussian at every pixel for EVERY flux element (achieve linear vs. cube scaling)
- 6) Loop over each flux element, compute Gaussian*dA, use this to normalize integral of Gaussian*dA to be the flux of the flux element. Add this to the map to get Br.
- 7) Save the map or compute auxilliary things like flux imbalance and average polar fields

References:
- LMSAL model website: https://www.lmsal.com/forecast/index.html
- Original paper: Schrijver & DeRosa 2003: https://ui.adsabs.harvard.edu/abs/2003SoPh..212..165S/abstract

ToDo: Get projection to work for a submap that doesn't cover the whole sun. Need to account for phi pixel spans
      for distant fluxes that are giving no contribution to the gaussian --> NaN overflow.
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

from modules.coord_manip import get_arclength, s2c
from modules.map_manip import MapMesh

# Flux Conversion factor to Gauss. The units of the flux elements are in units of 10^18 Mx
# so they are converted through the area division
fn_conv = 1e18/6.96e10**2


def project_lmsal_map(filename, mesh: MapMesh, style='adaptive', n_neighbors=10, fwhm_fixed=0.04,
                      ds_fac=2.0, arclength_fac=2.0, n_trunc=4.0, fwhm_pole=0.08, fwhm_equator=0.06,
                      balance_flux=False):
    """Project a LMSAL flux transport model snapshot to a rectilinear grid.

    This converts the "Lagrangian" model (a list of fluxes and positions) to a lon/lat map
    of the radial magnetic field (Br). The projection is done using a 2D Guassian using the
    arclength of the flux element to nearby pixels, truncated at `n_trunc` sigmas (default 4).

    The destination mesh may be a subset of the full-sun and contain non-uniform spacing.

    The `style` keyword determines how the projection is done. The following options are implemented:
    - 'adaptive': Use the local mesh spacing AND the nth nearest neighbor to determine the fwhm.
    - 'mesh': Use the local mesh spacing to determine the fwhm.
    - 'fixed': Use a uniform value for fwhm for all flux elements, set by `fwhm_fixed`.
    - 'smooth': Use a smoothly varying cosine squared profile in theta to set the fwhm. The pole/equator values
        are set by fwhm_pole and fwhm_equator.

    Parameters
    ----------
    filename : str
        Name of the .h5 file containing an LMSAL model snapshot.
    mesh : MapMesh
        A MapMesh class containing the information of the mesh you will be projecting to.
    style : str
        A string indicating the different options for projection. See above for the options.
    n_neighbors : int
        The number of neighbors used to get determine the distance sigma for style=='adaptive'.
    fwhm_fixed : float
        A uniform value of the Guassian fwhm, used for style == 'fixed', in radians (default 0.04).
    ds_fac : float
        Factor to multiply the local mesh spacing for the fwhm computation (default 2.0).
    arclength_fac : float
        Factor to multiply the arclength distance for the fwhm computation (default 2.0).
    n_trunc : float
        Number of sigmas to go before truncating the Gaussian (default 4.0)
    fwhm_pole : float
        The fwhm at the pole for style=='smooth'.
    fwhm_equator : float
        The fwhm at the equator for style=='smooth'.
    balance_flux : bool
        Optionally balance the raw fluxes to be equal positive and negative BEFORE projection (default False).

    Returns
    -------
    br : numpy.ndarray
        A 2D array containing the Br value at each pixel in phi, theta ordering.

    """

    # -----------------------------------------------------------------
    # Part 1: SFLUX I/O + Prep
    # -----------------------------------------------------------------
    # Read the LMSAL file
    sflux = read_sflux_file(filename)

    # Check for and remove any zero flux elements (they don't do anything for the map)
    sflux = sflux_remove_zeros(sflux)

    # Optional subsetting for developing the algorithm
    nsub = 1
    if nsub > 1:
        sflux = subset_sflux(sflux, n=nsub)

    # Get the spherical and cartesian coordinates of all points, convert to float64
    r = np.ones_like(sflux['phis'], dtype=np.float64)
    t = np.asarray(sflux['thetas'], dtype=np.float64)
    p = np.asarray(sflux['phis'], dtype=np.float64)
    x, y, z = s2c(r, t, p)

    # Optionally balance the fluxes
    if balance_flux:
        sflux, raw_flux_error = fluxbalance_sflux(sflux)

    # Scale the fluxes so that you get Br after projection
    fluxs = sflux['fluxs']*fn_conv

    # -----------------------------------------------------------------
    # Part 2: Determine the FWHM of the guassian
    # -----------------------------------------------------------------
    # Determine the local mesh spacing (diagonal) at the locations of the flux elements
    dp = mesh.interp_p2dp(p)
    dt = mesh.interp_t2dt(t)
    ds = np.sqrt(dt**2 + np.sin(t)*dp**2)

    # Now determine the FWHM of the guassian for each flux element
    if style == 'fixed':
        fwhm = np.zeros_like(fluxs) + fwhm_fixed

    elif style == 'local':
        fwhm = ds*ds_fac

    # Compute the arclength distance of the nth nearest neighbor of a flux element.
    # Use a fwhm that is the maximum of arclength or ds
    elif style == 'adaptive':
        chord_length = get_nth_distance(x, y, z, n_neighbors)
        arclength = get_arclength(chord_length)
        length1 = ds*ds_fac
        length2 = arclength*arclength_fac
        fwhm = np.where(length1 >= length2, length1, length2)

    # Or use a smoothly varying function of latitude (cosine squared).
    elif style == 'smooth':
        fwhm = (fwhm_pole - fwhm_equator)*np.cos(t)**2 + fwhm_equator

    # -----------------------------------------------------------------
    # Part 3: Project the flux elements to the sphere
    # -----------------------------------------------------------------
    # Set the sigma (a multiple of the FWHM)
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))

    # number of sigmas that you'll cover in the gaussian before truncating it
    max_length = n_trunc*sigma

    # Determine the bounding phi and theta range of the mesh for each flux point.
    # These are the points on the mesh that will be needed for the Gaussian Mapping
    # - Theta range is simple
    t_left = np.where(t - max_length > 0.0, t - max_length, 0.0)
    t_right = np.where(t + max_length < np.pi, t + max_length, np.pi)

    # - Phi range needs to be scaled by max change in lengthscale (1/min(sin(t))
    st_min = np.minimum(np.sin(t_left), np.sin(t_right))

    # - get one over the max length scale, trap for values whose inverse would be larger than 2pi (the whole map)
    tmp = np.where(st_min/max_length > 1/(2*np.pi), st_min/max_length, 1/(2*np.pi))
    phi_length = 1/tmp
    p_left = np.where(p - phi_length > 0.0, p - phi_length, 0.0)
    p_right = np.where(p + phi_length < 2*np.pi, p + phi_length, 2*np.pi)

    # - for boxes that cross the periodic boundaries, just include the whole range
    # - this is "lazier" but enables a less complex &/or if-statement free loop over fluxes
    p_left = np.where(p_right == 2*np.pi, 0.0, p_left)
    p_right = np.where(p_left == 0, 2*np.pi, p_right)

    # - now convert these to cell indexes
    it_left = np.intc(np.floor(mesh.interp_t2index(t_left)))
    it_right = np.intc(np.floor(mesh.interp_t2index(t_right)))
    ip_left = np.intc(np.floor(mesh.interp_p2index(p_left)))
    ip_right = np.intc(np.floor(mesh.interp_p2index(p_right)))

    # generate a blank br map
    br = np.zeros_like(mesh.da)

    # Loop over fluxes and use Gaussians to project them to the br map
    for iflux in range(0, len(fluxs)):
        # Get the coordinates of the current flux element, divide by radius in case of floating point conversion issue
        x0 = x[iflux]
        y0 = y[iflux]
        z0 = z[iflux]

        # get the sub window of points on the regular map grid (note python : indexing needs the +1).....
        i0 = ip_left[iflux]
        i1 = ip_right[iflux] + 1
        j0 = it_left[iflux]
        j1 = it_right[iflux] + 1

        # skip any elements that don't project onto the grid (happens if the map is a sub-window of the full sun)
        # This slows down the loop --> fix later when sub-windows are working
        # if i0 == i1 - 1 or j0 == j1 - 1:
        #    continue

        xsub = mesh.x2d[i0:i1, j0:j1]
        ysub = mesh.y2d[i0:i1, j0:j1]
        zsub = mesh.z2d[i0:i1, j0:j1]
        asub = mesh.da[i0:i1, j0:j1]

        # get the angular distance of each point on the map subset to the flux element
        dot = x0*xsub + y0*ysub + z0*zsub
        angle = np.arccos(dot)

        # compute the gaussian function G = exp(-0.5*angle^2/sig^2) times the area
        sig0 = sigma[iflux]
        exp_fac = 0.5/sig0**2
        g = np.exp(-exp_fac*angle**2)

        # compute the normalization so that the integral of G*A = Flux --> C=int(G*dA)/Flux
        flux0 = fluxs[iflux]
        norm = flux0/np.sum(g*asub)

        # add this contribution to the map
        br[i0:i1, j0:j1] = br[i0:i1, j0:j1] + norm*g

    # Return the map
    return br


def fluxbalance_sflux(sflux_dict):
    """Balance the the raw fluxes of the sflux model to be equally negative and positive.

    Parameters
    ----------
    sflux : dict
        An sflux_dict in the format returned by read_sflux_file

    Returns
    -------
    sflux : dict
        An sflux_dict, but now the fluxs array will be float64 and fluxbalanced.
    flux_error : float
        The fractional flux imbalance of the original data.

    """
    # Check the flux imbalance
    fluxs = np.float64(sflux_dict['fluxs'])
    locs_pos = np.nonzero(fluxs >= 0)
    locs_neg = np.nonzero(fluxs < 0)
    positive_flux = np.sum(fluxs[locs_pos])
    negative_flux = np.sum(fluxs[locs_neg])
    flux_error = (positive_flux + negative_flux)/(positive_flux + np.abs(negative_flux))*2

    # Correct the flux imbalance using a multiplicative factor
    ratio = np.sqrt(abs(positive_flux/negative_flux))
    fluxs[locs_pos] = fluxs[locs_pos]/ratio
    fluxs[locs_neg] = fluxs[locs_neg]*ratio

    # put the fluxes back into the dict
    sflux_dict['fluxs'] = fluxs

    return sflux_dict, flux_error


def get_map_flux_balance(br, mesh: MapMesh):
    """Compute the positive flux, negative flux, and fractional flux imbalance of a Br map.

    Parameters
    ----------
    br : numpy.ndarray
        A 2D numpy array of radial magnetic field (Br) values.
    mesh : MapMesh
        A MapMesh class containing all the mesh information for the rectilinear p, t grid.

    Returns
    -------
    positive_flux : float
        Positive flux integrated over the unit sphere (where Br >= 0).
    negative_flux : float
        Negative flux integrated over the unit sphere (where Br < 0).
    flux_error : float
        Fractional flux imbalance.

    """
    # Create an array of Br times the pixel area
    brda = br*mesh.da

    # use optimized array functions to count the flux for positive and negative seperately
    locs_pos = np.nonzero(brda >= 0)
    positive_flux = np.sum(brda[locs_pos])
    brda[locs_pos] = 0
    negative_flux = np.sum(brda)

    # get the flux error
    flux_error = (positive_flux + negative_flux)/(positive_flux + abs(negative_flux))*2

    return positive_flux, negative_flux, flux_error


def get_polar_fields(br, mesh: MapMesh, latitude=60.):
    """Compute the average Br values above a certain degrees latitude.

    Parameters
    ----------
    br : numpy.ndarray
        A 2D numpy array of radial magnetic field (Br) values.
    mesh : MapMesh
        A MapMesh class containing all the mesh information for the rectilinear p, t grid.
    latitude : float
        Latitude above which to average Br in degrees (default 60: i.e within 30 degrees of the pole).

    Returns
    -------
    br_north : float
        Average Br in the north.
    br_south : float
        Average Br in the south.

    """
    # Create an array of Br times the pixel area
    brda = br*mesh.da

    # Get the limits in radians
    deg_to_rad = np.pi/180.
    t_north = (90. - latitude)*deg_to_rad
    t_south = (90. + latitude)*deg_to_rad

    # Get the northern average
    j0 = 0
    j1 = int(np.floor(mesh.interp_t2index(t_north)))
    flux = np.sum(brda[:, j0:j1])
    area = np.sum(mesh.da[:, j0:j1])
    br_north = flux/area

    # Get the southern average
    j0 = int(np.floor(mesh.interp_t2index(t_south)))
    j1 = len(mesh.t) + 1
    flux = np.sum(brda[:, j0:j1])
    area = np.sum(mesh.da[:, j0:j1])
    br_south = flux/area

    return br_north, br_south


def get_nth_distance(x, y, z, n_neighbors=20):
    """Get the distance of the nth neighbor for a collection of points.

    This method requires scikit-learn

    Parameters
    ----------
    x : numpy.ndarray
        1D vector of Cartesian x positions for each point.
    y : numpy.ndarray
        1D vector of Cartesian y positions for each point.
    z : numpy.ndarray
        1D vector of Cartesian z positions for each point.
    n_neighbors : int
        Integer specifying how many neighbors to calculate and the nth distance to return (default 20).

    Returns
    -------
    distance : numpy.ndarray
        Vector of nth neighbor distances for each point

    """
    # This method requires scikit-learn, check for it first
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError as err:
        # Error handling
        print(f'### ERROR! The "adaptive" style of projection requires the scikit-learn package!')
        raise

    # Make a 3D copy of the data for the NearestNeighbors Algorithm
    x3d = np.array([x, y, z]).T.copy()

    # Compute the n nearest neighbors in cartesian space
    for algorithm in ['auto']:  # , 'ball_tree', 'kd_tree', 'brute']:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(x3d)
        distances, indices = nbrs.kneighbors(x3d)

    # get the arc distance of the nth neighbor
    distance = np.ones_like(x)
    for i, pairs in enumerate(distances):
        distance[i] = np.max(pairs[1:])

    return distance


def read_sflux_file(filename):
    """
    Read a LMSAL surface flux 'sfield' snaphot in HDF5 format.
    Return a dictionary with the 7 attributes
    """
    # read the h5 file
    print(f'\n### Reading sflux file: {filename}')
    sflux_container = h5py.File(filename, 'r')

    # everything is placed in the evolving_model_snapshot tag.
    # The first dataset is a one element list, but inside is a 7 element numpy.void containter
    sflux_dataset = sflux_container['evolving_model_snapshot'][0]

    # convert to dict type to make it less confusing (to me at least).
    sflux_dict = {}

    for key in sflux_dataset.dtype.names:
        sflux_dict[key.lower()] = sflux_dataset[key]

    # convert the now tag to a standard string
    sflux_dict['now'] = sflux_dict['now'].decode("utf-8")

    return sflux_dict


def subset_sflux(sflux_dict, n=10):
    """
    Take every nth element from the sflux model for testing
    """
    for key in ['phis', 'thetas', 'fluxs']:
        sflux_dict[key] = sflux_dict[key][::n]

    sflux_dict['nflux'] = len(sflux_dict['fluxs'])

    return sflux_dict


def sflux_remove_zeros(sflux_dict):
    """
    Remove flux elements that have zero flux from the sflux dictionary
    """
    inds = np.nonzero(sflux_dict['fluxs'] != 0)[0]

    # if the number of non-zeros is smaller, trim the arrays
    if len(inds) != len(sflux_dict['fluxs']):
        for key in ['phis', 'thetas', 'fluxs']:
            sflux_dict[key] = sflux_dict[key][inds]

        sflux_dict['nflux'] = len(sflux_dict['fluxs'])

    return sflux_dict


def plot_sflux_positions(sflux, signed=False, radius=None, xrange=(0, 2*np.pi), yrange=(np.pi, 0), filename=None):
    """Plot the positions of flux elements from an LMSAL FLux Transport Model snapshot.

    Parameters
    ----------
    sflux : dict
        An sflux_dict in the format returned by read_sflux_file
    signed : bool
        Color the dots by the sign of the flux element.
    radius : numpy.ndarray
        A vector whose value scales the relative size of each flux element in the plot (e.g. field or distance).
        It's length must match the length number of elements in the sflux model.
    xrange : tuple
        2 element tuple with the min and max longitude (radians).
    yrange :
        2 element tuple with the max, min of the co-latitude (radians).
    filename : str
        Optional full path of a .png file to save the plot to.
    """
    nx = 2
    ny = 2

    fig = plt.figure(figsize=(15, 10))
    fig.set_tight_layout(True)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')
    ax.set_autoscale_on(False)

    plt.tick_params('both', which='minor')
    ax.set_xlabel(f'Phi [Radians]')
    ax.set_ylabel(f'Theta [Radians]')

    x = sflux['phis']
    y = sflux['thetas']

    if radius is not None:
        smin = 0.01
        smax = 10.0
        sym_size = scale_numbers(radius**2, smin, smax)
    else:
        sym_size = 0.5

    if signed:
        color = scale_numbers(np.sign(sflux['fluxs']), 100, 155, fmin=-1, fmax=1)
        cmap = plt.get_cmap('seismic')
    else:
        color = None
        cmap = None

    this_plot = ax.scatter(x, y, s=sym_size, c=color, cmap=cmap)
    ax.grid(which='both', axis='both', linestyle='-', linewidth=0.5)

    ax.set_title('sflux flux elements: ' + sflux['now'])

    # x range and label
    if xrange is not None:
        ax.set_xlim(xrange[0], xrange[1])

    # y range and label
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])

    # ax.axis([0,1,0,1])

    if filename is not None:
        plt.savefig(fname=filename, dpi=150)
    else:
        plt.show()


def scale_numbers(f, smin, smax, fmin=None, fmax=None):
    """
    scale the values of a vector f to be within smin and smax
    """
    if fmin is None:
        fmin = np.min(f)
    if fmax is None:
        fmax = np.max(f)

    # limit to average if they are all the same value
    if fmin == fmax:
        s = np.zeros_like(f) + 0.5*(smax + smin)
    else:
        s = (smax - smin)*(f - fmin)/(fmax - fmin) + smin

    return s


def plot_map(map: np.ndarray, map_mesh: MapMesh, xrange=None, yrange=None,
             min=None, max=None, cmap_name='seismic', order='pt', title=None, filename=None):
    """
    Quick helper function for plotting a magnetic map. This is a simple thing that can/should
    be re-factored out for a more general subroutine in a different module.
    """
    fig = plt.figure(figsize=(15, 10))
    fig.set_tight_layout(True)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')
    ax.set_autoscale_on(False)

    plt.tick_params('both', which='minor')
    ax.set_xlabel(f'Phi [Radians]')
    ax.set_ylabel(f'Theta [Radians]')

    if min is None:
        min = np.min(map)

    if max is None:
        max = np.max(map)

    if xrange is None:
        xrange = [np.min(map_mesh.p), np.max(map_mesh.p)]

    if yrange is None:
        yrange = [np.max(map_mesh.t), np.min(map_mesh.t)]

    cmap = plt.get_cmap(cmap_name)

    if order == 'pt':
        map2plot = np.transpose(map)
    elif order == 'tp':
        map2plot = map

    x = map_mesh.p
    y = map_mesh.t

    this_plot = ax.pcolormesh(x, y, map2plot, shading='nearest', cmap=cmap, vmin=min, vmax=max)

    if title is not None:
        ax.set_title(title)

    # x range and label
    if xrange is not None:
        ax.set_xlim(xrange[0], xrange[1])

    # y range and label
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])

    if filename is not None:
        plt.savefig(fname=filename, dpi=150)
    else:
        plt.show()
