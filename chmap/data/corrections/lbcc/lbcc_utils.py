"""
Functions for evaluating the Limb Brightening Correction Coefficients
"""

import numpy as np
import datetime
import scipy.interpolate as interp
import scipy.optimize as optim
import scipy.integrate as integrate


def LinTrans_1Dhist(hist, bins, a, b):
    """
    Given a histogram H(bins), evaluate an approximation of H(a*bins + b).
    Note: it would be more accurate to apply the transformation to the data
    and re-evaluate the histogram, but when that is computationally prohibitive,
    this function provides a reasonable approximation.
    This is, essentially, the brute-force, discrete integration of a histogram
    from one set of bins to a different set of bins.
    Assumptions:
      - data is uniformly distributed within each bin
    :param hist: list with original histogram values (normalized)
    :param bins: list of bin edges (ascending)
    :param a: linear scaling
    :param b: linear shift
    :return: list with approximated histogram of transform (evaluated at original bins)
    """

    # double check that bins are sorted
    if not np.all(np.diff(bins) > 0):
        raise ("from LinTrans_1Dhist(): bin edges must be a strictly ascending list or 1D array.")

    # histogram is accurately transformed by transforming bin edges
    new_bins = bins * a + b
    # transform back to original bins by integration
    new_hist = hist_integration(hist, new_bins, bins)

    return new_hist


def hist_integration(hist, old_bins, new_bins):
    """
    Given a histogram 'hist' with bin edges defined as 'old_bins', integrate into 'new_bins'
    and return the resulting new_hist
    :param hist: list with original histogram values (normalized)
    :param old_bins: list of original bin edges (ascending)
    :param new_bins: list of new bin edges (ascending)
    :return: list of histogram values for linearly transformed hist integrated over
    old_bins. Note: the returned list will not sum to 1 if the linearly transformed bins
    exceed the limits of the old_bins.  Re-normalization was removed to improve how this
    function interacts with the optimization process.
    """
    old_max = old_bins[-1]
    old_min = old_bins[0]
    new_hist = np.full((len(new_bins) - 1,), 0., dtype='float')
    for ii in range(len(new_bins) - 1):
        l_edge = new_bins[ii]
        r_edge = new_bins[ii + 1]
        if l_edge >= old_max or r_edge < old_min:
            # This bin falls outside the transformed range. Assign it a 0
            # do nothing, the value was initialized to 0
            continue
        else:
            overlap_index = np.where(np.logical_and(old_bins[:-1] < r_edge, old_bins[1:] >= l_edge))
            # loop through each new bin that overlaps the evaluation bin
            for bin_num in overlap_index[0]:
                # determine what portion of the new bin intersects the evaluation bin
                l_overlap = max(l_edge, old_bins[bin_num])
                r_overlap = min(r_edge, old_bins[bin_num + 1])
                portion = (r_overlap - l_overlap) / (old_bins[bin_num + 1] - old_bins[bin_num])
                # add the appropriate portion to the evaluation bin
                new_hist[ii] = new_hist[ii] + portion * hist[bin_num]

    # re-normalize the new histogram (should only be necessary when 'bins' range does not
    # fully encompass the non-zero values of hist)
    # new_hist = new_hist/new_hist.sum()

    return new_hist


def LinTrans_1DHist_Interp(hist, bins, a, b):
    """
    This is how the linear transformation was originally applied.  This method works so long as
    the bin size is uniform and 'a' does not deviate much from 1.0.  The more 'a' deviates from
    1.0 and the more un-smooth 'hist' is, the worse this approximation becomes.
    :param hist: a list of histogram values
    :param bins: list of bin edges
    :param a: linear scaling
    :param b: linear shift
    :return: list of histogram values for linearly transformed hist integrated over original bins.
    """
    # double check that bins are sorted
    if not np.all(np.diff(bins) > 0):
        raise ("from LinTrans_1Dhist(): bin edges must be a strictly ascending list or 1D array.")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    new_centers = bin_centers * a + b

    interp_fun = interp.interp1d(new_centers, hist, bounds_error=False, fill_value=0, assume_sorted=True)
    new_hist = interp_fun(bin_centers)

    # new_hist = new_hist/new_hist.sum()

    return new_hist


def get_hist_sse(x, hist, hist_ref, bin_edges, trans_method=1):
    """
    This function applies a linear transformation 'beta*bin_edges + y' to hist and
    re-evaluates in the original bins.  Then compare to hist_ref and calc sum of
    squared errors.
    Inputs are compatible with scipy.optimize.minimize()
    :param x: a list of linear coefficients [beta, y]
    :param hist: 1D numpy array of normalized histogram values
    :param hist_ref: 1D numpy array of normalized reference-histogram values
    :param bin_edges: histogram bin edges as 1D list/array
    :param trans_method: transformation method 1 - Discrete integration
                                               2 - Approximation by interpolation
    :return: scalar sum of squared errors
    """

    beta = x[0]
    y = x[1]

    # calc the linear transformation of hist
    if trans_method == 1:
        trans_hist = LinTrans_1Dhist(hist, bin_edges, beta, y)
    elif trans_method == 2:
        trans_hist = LinTrans_1DHist_Interp(hist, bin_edges, beta, y)

    # calc sum of squared errors
    sse = sum(np.square(trans_hist - hist_ref))

    return sse


def optim_lbcc_linear(hist_ref, hist_fit, bin_edges, init_pars=np.asarray([1., 0.])):
    """
    Given a reference histogram hist_ref, find best linear coefficients to match hist_fit.
    Optimization performed using Nelder-Mead method.
    :param hist_ref: list of reference histogram values
    :param hist_fit: list of histogram values for histogram to be transformed
    :param bin_edges: intensity bin edges
    :param init_pars: values of [Beta, y] to initialize the Nelder-Mead process
    :return: minimized
    """

    optim_out = optim.minimize(get_hist_sse, init_pars, args=(hist_fit, hist_ref, bin_edges), method="Nelder-Mead")

    if optim_out.status != 0:
        # Nelder-Mead was unsuccessful
        # implement plan B?
        print("Warning: Nelder-Mead optimization of LBCC coefficients failed with status ", optim_out.status)

    return optim_out


def get_functional_sse(x, hist_ref, hist_mat, mu_vec, int_bin_edges, model, trans_method=1):
    """
    This function applies a linear transformation 'beta*bin_edges + y' to hist and
    re-evaluates in the original bins.  Then compare to hist_ref and calc sum of
    squared errors.  Objective function has been augmented to guide the solution
    back to reasonable Beta, y values. This is because exotic values for Beta and/or
    y will shift the histogram so much that it returns zero value to the original
    bins. This situation is problematic for any optimizer.
    Inputs are compatible with scipy.optimize.minimize()
    :param x: a list of functional coefficients [a1, a2, a3, b1, b2, b3]. Number of expected
    coefficients varies based on 'model' input.
    :param hist_ref: numpy vector of reference histogram values
    :param hist_mat: 2D (mu, intensity) numpy array of normalized reference-histogram values
    :param mu_vec: mu bin center values corresponding to first dimension of hist_mat
    :param int_bin_edges: histogram intensity bin edges as 1D list/array
    :param model:   1 - cubic polynomials for beta and y (6 params)
                    2 - power law for Beta; log for y (3 params)
                    3 - formulation based on theoretical LBCCs (6 params)
    :param trans_method: transformation method 1 - Discrete integration
                                               2 - Approximation by interpolation (Method used in paper)
    :return: scalar sum of squared errors
    """
    sum_sse_over_mu = 0.
    # loop through mu values
    for index, mu in enumerate(mu_vec):
        # based on model and mu, evaluate beta and y values
        if model == 0:
            beta = x[0]
            y = x[1]
        elif model == 1:
            beta, y = get_beta_y_cubic(x, mu)
        elif model == 2:
            beta, y = get_beta_y_power_log(x, mu)
        elif model == 3:
            beta, y = get_beta_y_theoretic_based(x, mu)
        else:
            raise ("Error: in get_functional_sse(), model=" + str(model) + " is not a valid selection.")

        sse = 0
        # check for limb-dimming parameters
        # if beta < 0.8:
        #     # impose artificial constraint
        #     sse += 0.8-beta
        # if y > 0.2:
        #     sse += y-0.2
        # Impose penalties for beta/y outside 0 <= beta*I + y <= 5 for I in [2,2.5]
        # Essentially, the 2 to 2.5 range of the original histogram must fall in the
        # range 0 to 5 after being transformed. Do not allow extreme shifts
        if beta < -y / 2.:
            sse += abs(beta + y / 2.)
        elif beta > (5 - y) / 2.5:
            sse += abs((5 - y) / 2.5 - beta)
        else:
            # calc the linear transformation of hist
            if trans_method == 1:
                trans_hist = LinTrans_1Dhist(hist_mat[index,], int_bin_edges, beta, y)
            elif trans_method == 2:
                trans_hist = LinTrans_1DHist_Interp(hist_mat[index,], int_bin_edges, beta, y)

            # calc sum of squared errors
            sse += sum(np.square(trans_hist - hist_ref))
        # add this mu-bin to the total
        sum_sse_over_mu += sse

    return sum_sse_over_mu


def optim_lbcc_functional_forms(hist_ref, hist_mat, mu_vec, int_bin_edges, init_pars=None, model=1, jac_opt=None):
    """
    Code to provide initial contitions and/or optimizer options for any of the three LBCC
    functional forms.
    Code stub?  If we want a function like this, adapt the code from analysis/Beta-y_functionals_analysis_frompkl.py
    to this form.
    :param hist_ref:
    :param hist_mat:
    :param mu_vec:
    :param int_bin_edges:
    :param init_pars:
    :param model:   1 - cubic polynomials for beta and y (6 params)
                    2 - power law for Beta; log for y (3 params)
                    3 - formulation based on theoretical LBCCs (6 params)
    :return:
    """

    if model == 1:
        # first check for the appropriate number of initial parameters
        if init_pars is not None and len(init_pars) != 6:
            raise ("Error: optim_lbcc_functional_forms(model=1) requires len(init_params)==6")
        # set initial conditions and optim method
        if init_pars is None:
            init_pars = np.array([-1., 1.1, -0.4, 4.7, -5., 2.1])
        method = "SLSQP"

        # cubic constraints for first derivative (beta' non-positive, y' non-negative for mu in [0,1]
        lb = np.zeros(4)
        ub = np.zeros(4)
        lb[[0, 1]] = -np.inf
        ub[[2, 3]] = np.inf
        A = np.zeros((4, 6))
        A[0, 0] = 1
        A[1, 0:3] = [1, 2, 3]
        A[2, 3] = 1
        A[3, 3:6] = [1, 2, 3]

        lin_constraint = optim.LinearConstraint(A, lb, ub)

        optim_out = optim.minimize(get_functional_sse, init_pars,
                                   args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
                                   method=method, jac="2-point", constraints=lin_constraint)

    elif model == 2:
        # first check for the appropriate number of initial parameters
        if init_pars is not None and len(init_pars) != 3:
            raise ("Error: optim_lbcc_functional_forms(model=2) requires len(init_params)==3")
        # set initial conditions and optim method
        if init_pars is None:
            init_pars = np.array([.93, -0.13, 0.6])
        method = "BFGS"
        gtol = 1e-4

        optim_out = optim.minimize(get_functional_sse, init_pars,
                                   args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
                                   method=method, jac='2-point', options={'gtol': gtol})

    elif model == 3:
        # first check for the appropriate number of initial parameters
        if init_pars is not None and len(init_pars) != 6:
            raise ("Error: optim_lbcc_functional_forms(model=3) requires len(init_params)==6")
        # set initial conditions and optim method
        if init_pars is None:
            init_pars = np.array([-0.05, -0.3, -.01, 0.4, -1., 6.])
        method = "BFGS"

        optim_out = optim.minimize(get_functional_sse, init_pars,
                                   args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
                                   method=method)

    # init_pars = np.asarray([1., 0.])
    # optim_out = optim.minimize(get_functional_sse, init_pars, args=(hist_ref, hist_mat, mu_vec, int_bin_edges, model),
    #                            method=method, jac=jac_opt)
    mu_vec, int_bin_edges, model = 1
    if optim_out.status != 0:
        # Nelder-Mead was unsuccessful
        # implement plan B?
        print("Warning: Nelder-Mead optimization of LBCC coefficients failed with status ", optim_out.status)

    return optim_out


def eval_lbcc_4reduced_bins(norm_hist, image_intensity_bin_edges, int_bin_n, mu_bin_centers, cur_results, half_steps):
    """
    This is more of an analysis code.  It automates fitting LBCCs for consecutively larger
    intensity bins.  Each step of the loop reduces the number of intensity bins by half.
    See analysis/Intensity-bin_analysis_looped.py for a use-case.
    :param norm_hist: normalized histogram values
    :param image_intensity_bin_edges: Intensity bin edges for norm_hist.
    :param int_bin_n: Number of intensity bins used. Index for second dimension of cur_results.
    :param mu_bin_centers: mu bin centers corresponding to first dim of cur_results
    :param cur_results: 3D array indexed by mu; number of intensity bins; [Beta, y, SSE]
    :param half_steps: number of times to half the number of bins. Code is written a
    little weird: step=1 is actually the 0th step. Bins are actually halved
    (half_steps-1) times.
    :return:
    """
    results = cur_results

    for step in range(1, half_steps + 1):
        if step == 1:
            hist_ref = norm_hist[-1,]
            new_bins = image_intensity_bin_edges
        else:
            new_bins = image_intensity_bin_edges[::2 ** (step - 1)]
            hist_ref = hist_integration(norm_hist[-1,], image_intensity_bin_edges, new_bins)

        n_bins = len(new_bins) - 1
        n_bin_index = np.where(np.equal(n_bins, int_bin_n))
        if len(n_bin_index) == 0:
            raise ("Number of bins does not match index.")
        ref_peak_index = np.argmax(hist_ref)
        ref_peak_val = hist_ref[ref_peak_index]

        for ii in range(mu_bin_centers.__len__() - 1):
            if step == 1:
                hist_fit = norm_hist[ii,]
            else:
                hist_fit = hist_integration(norm_hist[ii,], image_intensity_bin_edges, new_bins)

            # estimate correction coefs that match fit_peak to ref_peak
            fit_peak_index = np.argmax(hist_fit)
            fit_peak_val = hist_fit[fit_peak_index]
            beta_est = fit_peak_val / ref_peak_val
            y_est = new_bins[ref_peak_index] - beta_est * new_bins[fit_peak_index]
            init_pars = np.asarray([beta_est, y_est])

            # optimize correction coefs
            optim_result = optim_lbcc_linear(hist_ref, hist_fit, new_bins, init_pars)
            # record results
            results[ii, n_bin_index, 0] = optim_result.x[0]
            results[ii, n_bin_index, 1] = optim_result.x[1]
            results[ii, n_bin_index, 2] = optim_result.fun

    return results


def get_analytic_limb_brightening_curve(t=1.5e6, g_type=1, R0=1.0, mu_vec=None):
    """
    Code converted from Ron's Matlab library.  Numerically evaluate the theoretic
    function for limb brightening as a function of temperature.  See
    analysis/Generate_theoretical_LBCCs.py for use-case.
    :param t: temperature (Kelvin)
    :param g_type: density function
    :param R0: solar radii
    :param mu_vec: mu values to evaluate at
    :return: vectors of observed brightness and corresponding mu values
    """
    # T_theory = 1.5e6
    t_mas = t / 2.807067e7  # Convert T to MAS units.

    # Constants(MAS units)
    g0 = 0.823 / (R0 ** 2)
    mu_w = 0.6
    kb = 1
    mp = 1
    mu_limb = np.sqrt(1 - 1 / (R0 ** 2))
    lamb = (kb * t_mas) / (g0 * mu_w * mp)

    # Hard-coded parameters:
    x_max = 10.  # Max x for integral
    dx = 0.0001

    if mu_vec is None:
        n_mu = 50  # Number of mu points in curve (size of array).
        mu_vec = np.linspace(mu_limb, 1, num=n_mu)  # Equal-spaced mu values to mu=1.

    # define x-grid to integrate over
    x_vec = np.arange(0., x_max, dx)

    # create meshgrid
    mu_mat, x_mat = np.meshgrid(mu_vec, x_vec)

    # define observer radius 'r' at each grid point
    r = np.sqrt(np.power(x_mat, 2) + 2. * x_mat * mu_mat * R0 + R0 ** 2)

    # Select density function based on gravity assumption:
    if g_type == 0:
        n_of_r = np.exp(-(r - R0) / lamb)
    elif g_type == 1:
        n_of_r = np.exp(2 * (R0 * (R0 - r)) / (r * lamb))
    else:
        n_of_r = 0. * r

    # Compute intensity integral:
    obsv_int = integrate.simps(n_of_r, axis=0) * dx

    return obsv_int, mu_vec


def get_beta_y_cubic(x, mu):
    """
    Evaluation of cubic polynomial form of LBCCs.
    Solutions are pinned at Beta(mu=1)=1 and y(mu=1)=0.
    :param x: parameter values as list/list-like
    :param mu: value of mu
    :return: Beta and y
    """
    beta = 1. - (x[0] + x[1] + x[2]) + x[0] * mu + x[1] * mu ** 2 + x[2] * mu ** 3
    y = -(x[3] + x[4] + x[5]) + x[3] * mu + x[4] * mu ** 2 + x[5] * mu ** 3

    return beta, y


def get_beta_y_power_log(x, mu):
    """
    Evaluate simple 3-parameter log and power-law form.
    Solutions are pinned at Beta(mu=1)=1 and y(mu=1)=0.
    :param x: parameter values as list/list-like
    :param mu: value of mu
    :return: Beta and y
    """
    beta = 1. - x[0] + x[0] * mu ** x[1]
    y = x[2] * np.log10(mu)

    return beta, y


def get_beta_y_theoretic_based(x, mu):
    """
    Theory-based LBCC functional form evaluation.
    Solutions are pinned at Beta(mu=1)=1 and y(mu=1)=0.
    :param x: parameter values as list/list-like
    :param mu: value of mu
    :return: Beta and y
    """

    f1 = -x[0] + x[0] * mu + x[1] * np.log10(mu)
    f0 = -x[2] + x[2] * mu + x[3] * np.log10(mu)
    n = x[4]
    log_alpha = x[5]
    beta = n / (f1 + n)
    y = (f1 * log_alpha / n - f0) * beta

    return beta, y


def get_beta_y_theoretic_continuous_1d_indices(x, los_image, mu_limit=0.1):
    """
    Determine beta and y values as 1D arrays for a given mu array. Return them along
      with their corresponding index locations for mu.

    @param x: parameter values as list/list-like.
    @param los_image: original los image for use of calculating correct indices to use
    @param mu_limit: lower limit on mu
    @return: beta1d: 1d array of valid beta values.
    @return: y1d: 1d array of valid y values.
    @return: mu_indexes: numpy.where output that will select "good" values from the mu_array.
    """
    # definitions
    mu_array = los_image.mu
    data = los_image.data

    # get mu index locations where mu is valid.
    mu_indices = np.where(mu_array > 0)
    # use_indices = np.logical_and(mu_array > 0, data > 0)  # added
    # JT - avoid potential problems with log10 (post LBC values are sometimes below 0)
    use_indices = np.logical_and(mu_array > 0., data > 2.)

    # get a 1D array of the valid mu values
    # mu1d = mu_array[mu_indices]
    mu1d = mu_array[use_indices]

    # threshold mu to be equal to mu_limit when it is below mu_limit
    mu1d = np.where(mu1d < mu_limit, mu_limit, mu1d)

    # calculate beta and y as 1d arrays
    beta1d, y1d = get_beta_y_theoretic_based(x, mu1d)

    return beta1d, y1d, mu_indices, use_indices


def cadence_choose(all_times, window_centers, window_del, method='default'):
    """
    Choose only one observation/datetime per selection window.

    Of the available observation datetimes 'all_times', choose a single observation
    time in each selection window.  Selection windows are defined by multiple
    window_centers and a one-half window width defined by window_del.
    :param all_times: pandas.Series (dtype: datetime64)
                      The available observation times.
    :param window_centers: numpy.ndarray (dtype='datetime64')
                           A one-dimensional array of selection window center times.
    :param window_del: numpy.timedelta64
                       Scalar that defines one half of the selection window width.
    :param method: string {'default', 'sorted'}
                   Choose 'sorted' when all_times is sorted in ascending order. This
                   will result in a shorter computation time.
    :return: list (boolean)
             A logical list that indicates which elements of 'all_times' are the
             best match for the specified selection windows.
    """
    n_times = len(all_times)
    keep_ind = [False, ] * n_times
    # assume window_centers is sorted-ascending
    for center_time in window_centers:
        window_min = center_time - window_del
        window_max = center_time + window_del
        # determine if there are any new candidates
        if (all_times.iloc[0] > window_max) or (all_times.iloc[-1] < window_min):
            # there are no observation times in this window, skip
            continue
        if method == "default":
            match_cands = (all_times <= window_max) & (all_times >= window_min) & (
                np.logical_not(keep_ind))
            if not any(match_cands):
                continue
            else:
                cand_ind = np.where(match_cands)[0]
        elif method == "sorted":
            lower_ind = all_times.searchsorted(window_min, 'left')
            upper_ind = all_times.searchsorted(window_max, 'left')
            if upper_ind == lower_ind:
                continue
            else:
                cand_ind = np.arange(lower_ind, upper_ind, step=1)

        # of the selected candidates, choose the closest one
        if len(cand_ind) > 1:
            cand_diff = np.abs(center_time - all_times[match_cands])
            cand_select = cand_ind[np.argmin(cand_diff)]
            keep_ind[cand_select] = True
        else:
            keep_ind[cand_ind[0]] = True

    return keep_ind


def moving_averages(time_min, time_max, weekday, days=None):
    # find day of the week of start time
    day_start = time_min.weekday()
    if day_start != weekday:
        time_min = time_min + datetime.timedelta(days=7 - day_start)
    # calculate number of weeks
    number_of_weeks = int((time_max - time_min).days / 7)
    # returns array of moving averages center dates, based off start date and number of weeks
    moving_avg_centers = np.array(
        [np.datetime64(str(time_min)) + ii * np.timedelta64(1, 'W') for ii in range(number_of_weeks+1)])
    # returns moving width based of number of days
    if days is not None:
        moving_width = np.timedelta64(days, 'D')
    else:
        moving_width = None

    return moving_avg_centers, moving_width


#### DOESN'T WORK, NOT USED ####
def get_beta_y_theoretic_continuous_1d(x, mu_array):
    """
    this method isn't currently working
    only gets rid of the outer portion of the stuff
    """
    # flatten mu into 1D array
    mu1d = mu_array.flatten()
    inds = np.logical_and(mu1d <= 1, mu1d > 0)
    # calculate beta and y
    beta1d = np.zeros(mu1d.shape)
    y1d = np.zeros(mu1d.shape)
    beta1d[inds], y1d[inds] = get_beta_y_theoretic_based(x, mu1d[inds])
    # create 2d beta& y arrays
    beta = beta1d.reshape(mu_array.shape)
    y = y1d.reshape(mu_array.shape)

    return beta, y


#### SHOULD NOT BE USED - WORKS THOUGH ####
def get_beta_y_theoretic_interp(x, mu_array_2d, mu_array_1d):
    """
    calculate beta and y using the theoretic fit
    for a 2D array of mu values
    @param x: theoretic fit parameters
    @param mu_array_2d: 2D array of mu values
    @param mu_array_1d: 1D array of mu bin centers
    @return:
    """

    # calculate beta and y for mu bin centers array
    beta_y_mu = np.zeros((len(mu_array_1d), 3))
    for mu_index, mu in enumerate(mu_array_1d):
        beta_y_mu[mu_index, 0] = mu
        beta_y_mu[mu_index, 1:] = get_beta_y_theoretic_based(x, mu)
    mu = beta_y_mu[0]
    beta_mu = beta_y_mu[1]
    y_mu = beta_y_mu[2]

    # interpolate to work for 2D mu array
    beta_interp = interp.interp1d(x=mu, y=beta_mu, fill_value=0, bounds_error=False)
    y_interp = interp.interp1d(x=mu, y=y_mu, fill_value=0, bounds_error=False)

    # calculate beta and y array
    beta = beta_interp(mu_array_2d)
    y = y_interp(mu_array_2d)

    return beta, y


# SOMETHING WRONG WITH THE BETA/Y CALCULATION FROM GET_BETA_Y_THEORETIC_BASED ####
def get_beta_y_theoretic_continuous_old(x, mu_array):
    # determine where mu is -9999
    mu = np.where(mu_array > 0, mu_array, 0.0001)

    # calculate beta and y
    beta, y = get_beta_y_theoretic_based(x, mu)
    beta = np.where(mu_array > 0, beta, 0)
    y = np.where(mu_array > 0, y, 0)

    return beta, y