"""
Functions for Evaluation of Inter-Instrument Transformation Coefficients
"""


import numpy as np
import scipy.optimize as optim
import modules.lbcc_funs as lbcc_funcs


def optim_iit_linear(hist_ref, hist_fit, bin_edges, init_pars=np.asarray([1., 0.])):
    """
    Given a reference histogram hist_ref, find best linear coefficients to match hist_fit.
    Optimization performed using Nelder-Mead method.
    :param hist_ref: list of reference histogram values
    :param hist_fit: list of histogram values for histogram to be transformed
    :param bin_edges: intensity bin edges
    :param init_pars: values of [Alpha, x] to initialize the Nelder-Mead process
    :return: minimized
    """

    optim_out = optim.minimize(lbcc_funcs.get_hist_sse, init_pars, args=(hist_fit, hist_ref, bin_edges),
                               method="Nelder-Mead")

    if optim_out.status != 0:
        print("Warning: Nelder-Mead optimization of IIT coefficients failed with status ", optim_out.status)

    return optim_out


def iit_hist(lbcc_data, los_image, intensity_bin_edges, lat_band=[-np.pi / 2.4, np.pi / 2.4], log10=True):
    """
    function to calculate 1D histogram for IIT
    """

    # first reduce to points greater than intensity-min and in lat-band
    intensity_min = min(intensity_bin_edges)
    intensity_max = max(intensity_bin_edges)
    lat_band_index = np.logical_and(los_image.lat <= max(lat_band), los_image.lat >= min(lat_band))
    #intensity_index = np.logical_and(np.log10(lbcc_data) >= intensity_min, np.log10(lbcc_data) <= intensity_max)
    #use_index = np.logical_and(intensity_index, lat_band_index)

    use_data = lbcc_data[lat_band_index]
    if log10:
        use_data[use_data < 0.] = 0.
        use_data = np.log10(use_data)

    # generate intensity histogram
    hist_out, bin_edges = np.histogram(use_data, bins=intensity_bin_edges)

    return hist_out, use_data
