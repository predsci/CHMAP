"""
Functions to plot EUV images and maps
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def PlotImage(los_image, nfig=None, mask_rad=1.5, title=None):
    """Super simple plotting routine for LosImage objects.
    imshow() should be replaced with pcolormesh() for generalizing to non-uniform rectilinear grids
    OR use Ron's Plot2D from PSI's 'tools'
    """
    # set color palette and normalization (improve by using Ron's colormap setup)
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(los_image.data))
    # norm = mpl.colors.LogNorm()
    im_cmap = plt.get_cmap('sohoeit195')

    # mask-off pixels outside of mask_rad
    # mesh_x, mesh_y = np.meshgrid(los_image.x, los_image.y)
    # mesh_rad = np.sqrt(mesh_x**2 + mesh_y**2)
    # plot_arr = los_image.data
    # plot_arr[mesh_rad > mask_rad] = 0.001

    # remove extremely small values from data so that log color scale treats them as black
    # rather than white
    plot_arr = los_image.data
    plot_arr[plot_arr < .001] = .001

    # plot the initial image
    if nfig is None:
        cur_figs = plt.get_fignums()
        if not nfig:
            nfig = 0
        else:
            nfig = cur_figs.max() + 1

    plt.figure(nfig)

    plt.imshow(plot_arr, extent=[los_image.x.min(), los_image.x.max(), los_image.y.min(), los_image.y.max()],
               origin="lower", cmap=im_cmap, aspect="equal", norm=norm)
    plt.xlabel("x (solar radii)")
    plt.ylabel("y (solar radii)")
    if title is not None:
        plt.title(title)

    # may want a save-to-file option at some point
    return None


def PlotCorrectedImage(corrected_data, los_image, nfig=None, title=None):
    # set color map
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(los_image.data))
    # norm = mpl.colors.LogNorm()
    im_cmap = plt.get_cmap('sohoeit195')

    plot_arr = corrected_data
    plot_arr[plot_arr < .001] = .001

    # plot the initial image
    if nfig is None:
        cur_figs = plt.get_fignums()
        if not nfig:
            nfig = 0
        else:
            nfig = cur_figs.max() + 1

    plt.figure(nfig)

    plt.imshow(plot_arr, extent=[los_image.x.min(), los_image.x.max(), los_image.y.min(), los_image.y.max()],
               origin="lower", cmap=im_cmap, aspect="equal", norm=norm)
    plt.xlabel("x (solar radii)")
    plt.ylabel("y (solar radii)")
    if title is not None:
        plt.title(title)

    return None


def PlotMap(map_plot, nfig=None, title=None, map_type=None):
    """
    Super simple plotting routine for PsiMap objects.
    imshow() should be replaced with pcolormesh() for generalizing to non-uniform rectilinear grids
    OR use Ron's Plot2D from PSI's 'tools'
    """
    # set color palette and normalization (improve by using Ron's colormap setup)
    if map_type == "CHD":
        norm = mpl.colors.LogNorm(vmin=0.01, vmax=np.nanmax(map_plot.data))
        im_cmap = plt.get_cmap('twilight_shifted')
    else:
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(map_plot.data))
        im_cmap = plt.get_cmap('sohoeit195')

    # plot the initial image
    if nfig is None:
        cur_figs = plt.get_fignums()
        if not nfig:
            nfig = 0
        else:
            nfig = cur_figs.max() + 1

    # convert map x-extents to degrees
    x_range = [180 * map_plot.x.min() / np.pi, 180 * map_plot.x.max() / np.pi]

    # setup xticks
    xticks = np.arange(x_range[0], x_range[1] + 1, 30)

    plt.figure(nfig)
    plt.imshow(map_plot.data, extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()],
               origin="lower", cmap=im_cmap, aspect=90.0, norm=norm)
    plt.xlabel("Carrington Longitude")
    plt.ylabel("Sine Latitude")
    plt.xticks(xticks)

    if title is not None:
        plt.title(title)

    plt.show(block=False)
    return None


def Plot2D_Data(data, nfig=None, xlabel=None, ylabel=None, title=None):
    # data = data /data.max()
    # normalize by log10
    norm = mpl.colors.LogNorm(vmin=0.001, vmax=data.max(), clip=True)

    plot_arr = data
    plot_arr[plot_arr < .001] = .001

    if nfig is None:
        cur_figs = plt.get_fignums()
        if not nfig:
            nfig = 0
        else:
            nfig = cur_figs.max() + 1

    plt.figure(nfig)

    plt.imshow(plot_arr, extent=[np.amin(data[0:]), np.amax(data[0:]), np.amin(data[1:]), np.amax(data[1:])],
               origin="lower", cmap='Greys', aspect="equal", norm=norm)
    plt.colorbar()

    # plot title and axes labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    return None


def Plot_LBCC_Hists(plot_hist, date_obs, instrument, intensity_bin_edges, mu_bin_edges, figure, plot_index):
    """
    function to plot 2D histograms used in LBCC calculation
    @param plot_hist:
    @param date_obs:
    @param instrument:
    @param intensity_bin_edges:
    @param mu_bin_edges:
    @param figure:
    @param plot_index:
    @return:
    """
    # simple plot of raw histogram
    plt.figure(figure + instrument + " " + str(100 + plot_index))
    plt.imshow(plot_hist, aspect="auto", interpolation='nearest', origin='low',
               extent=[intensity_bin_edges[0], intensity_bin_edges[-2] + 1., mu_bin_edges[0],
                       mu_bin_edges[-1]])
    plt.xlabel("Pixel intensities")
    plt.ylabel("mu")
    plt.title("Raw 2D Histogram Data: \n" + "Instrument: " + instrument + " \n " + str(date_obs))

    # # Normalize each mu bin
    norm_hist = np.full(plot_hist.shape, 0.)
    row_sums = plot_hist.sum(axis=1, keepdims=True)
    # but do not divide by zero
    zero_row_index = np.where(row_sums != 0)
    norm_hist[zero_row_index[0]] = plot_hist[zero_row_index[0]] / row_sums[zero_row_index[0]]

    # # simple plot of normed histogram
    plt.figure(figure + instrument + " " + str(200 + plot_index))
    plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='low',
               extent=[intensity_bin_edges[0], intensity_bin_edges[-1], mu_bin_edges[0],
                       mu_bin_edges[-1]])
    plt.xlabel("Pixel intensities")
    plt.ylabel("mu")
    plt.title(
        "2D Histogram Data Normalized by mu Bin: \n" + "Instrument: " + instrument + " \n " + str(date_obs))
    return None


def Plot1d_Hist(norm_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list, figure,
                xlabel, ylabel, title):
    """
    plot 1D IIT Histogram
    @return:
    """
    # plot original
    plt.figure(figure)
    plt.plot(intensity_bin_edges[:-1], norm_hist, color=color_list[inst_index],
             linestyle=linestyle_list[inst_index], label=instrument)
    plt.xlim(0, np.max(intensity_bin_edges))
    plt.ylim(0, 0.050)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.legend()

    return None