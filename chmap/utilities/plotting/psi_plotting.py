"""
Functions to plot EUV images and maps
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import numpy as np
from matplotlib.lines import Line2D
from astropy.visualization import HistEqStretch, ImageNormalize
import cv2

import chmap.utilities.datatypes.datatypes as psi_dt


def PlotImage(los_image, nfig=None, mask_rad=1.5, title=None):
    """Super simple plotting routine for LosImage objects.
    imshow() should be replaced with pcolormesh() for generalizing to non-uniform rectilinear grids
    OR use Ron's Plot2D from PSI's 'tools'
    """

    # mask-off pixels outside of mask_rad
    # mesh_x, mesh_y = np.meshgrid(los_image.x, los_image.y)
    # mesh_rad = np.sqrt(mesh_x**2 + mesh_y**2)
    # plot_arr = los_image.data
    # plot_arr[mesh_rad > mask_rad] = 0.001

    # remove extremely small values from data so that log color scale treats them as black
    # rather than white
    if type(los_image) is psi_dt.IITImage:
        plot_arr = los_image.iit_data
    elif type(los_image) is psi_dt.LBCCImage:
        plot_arr = los_image.lbcc_data
    else:
        plot_arr = los_image.data

    plot_arr[plot_arr < .001] = .001

    # set color palette and normalization (improve by using Ron's colormap setup)
    norm_max = max(1.01, np.nanmax(plot_arr))

    if los_image.info["instrument"] == "AIA":
        cmap_str = "sdoaia" + str(los_image.info["wavelength"])
        norm = mpl.colors.LogNorm(vmin=10., vmax=norm_max)
        # cmap_str = 'sohoeit195'
    elif los_image.info["instrument"] in ("EUVI-A", "EUVI-B"):
        cmap_str = "sohoeit" + str(los_image.info["wavelength"])
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=norm_max)
    else:
        cmap_str = 'sohoeit195'
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=norm_max)
    im_cmap = plt.get_cmap(cmap_str)

    # set x and y ticks
    xticks = (-1., 0.,  1.)
    yticks = (-1., 0.,  1.)

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
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(alpha=0.6, linestyle='dashed', lw=0.5)
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


def PlotMap(map_plot, nfig=None, title=None, map_type=None, save_map=False,
            save_dir='maps/synoptic/'):
    """
    Super simple plotting routine for PsiMap objects.
    imshow() should be replaced with pcolormesh() for generalizing to non-uniform rectilinear grids
    OR use Ron's Plot2D from PSI's 'tools'
    """
    # set color palette and normalization (improve by using Ron's colormap setup)
    if map_type == "CHD":
        # norm = mpl.colors.LogNorm(vmin=0.01, vmax=np.nanmax(map_plot.data))
        im_cmap = plt.get_cmap('Greys')
        norm = mpl.colors.LogNorm(vmin=0.01, vmax=1)
        plot_mat = map_plot.chd.astype('float32')
    else:
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.nanmax(map_plot.data))
        im_cmap = plt.get_cmap('sohoeit195')
        plot_mat = map_plot.data

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
    if map_type == 'Contour':
        x_extent = np.linspace(x_range[0], x_range[1], len(map_plot.x))
        plt.contour(x_extent, map_plot.y, map_plot.chd, origin="lower", colors='r',
                    extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()], linewidths=0.3)
    elif map_type == 'AR_Contour':
        x_extent = np.linspace(x_range[0], x_range[1], len(map_plot.x))
        plt.contour(x_extent, map_plot.y, map_plot.chd, origin="lower", colors='b',
                    extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()], linewidths=0.3)
    else:
        plt.imshow(plot_mat, extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()],
                   origin="lower", cmap=im_cmap, aspect=90.0, norm=norm)
    plt.xlabel("Carrington Longitude")
    plt.ylabel("Sine Latitude")
    plt.xticks(xticks)

    if title is not None:
        plt.title(title)

    plt.show(block=False)
    if save_map:
        plt.savefig(save_dir + str(map_plot.data_info.date_obs[0]))

    return None


def map_movie_frame(map_plot, int_range, save_path='maps/synoptic/',
                    nfig=None, title=None, map_type=None, dpi=300,
                    dark=True, quality=False, no_data=False):
    """
    Plot and save a PsiMap to png file as a frame for a video.

    :param map_plot: utilities.datatypes.datatypes.PsiMap
                     The map to plot.
    :param int_range: list (float) [min intensity, max intensity]
                      Minimum and maximum values considered in the plotting
                      colorscale. Values beyond this range will saturate the
                      colorscale.
    :param save_path: str
                      Full path (w/ filename) where PNG should be saved. Should
                      end in '.png'.
    :param nfig: int (optional)
                 This may be specified to avoid affecting existing figures.
    :param title: str
                  Title to appear above plot.
    :param map_type: str
                     EUV images are default. Other types of images will be supported
                     in the future.
    :param dpi: int
                Dots-per-inch image quality of PNG.
    :param dark: bool
                Flag to use a dark background for the plot (good for movies).
    :param quality: bool
                Flag to plot the data as a colored mu quality map instead.
                If the 'CHD' map_type is also specified, it will mask the quality
                map by the CH detection (above a threshold set by quality_map_plot_helper).
    :param no_data: bool
                Flag to plot the no-data regions as a gray strip (using masking)
    :return: None
             This routine writes a file to save_path, but has no explicit output.
    """
    # set color palette and normalization (improve by using Ron's colormap setup)
    if map_type == "CHD":
        im_cmap = copy.copy(plt.get_cmap('Greys'))
        norm = mpl.colors.Normalize(vmin=int_range[0], vmax=int_range[1])
        plot_mat = map_plot.chd.astype('float32')
    else:
        norm = mpl.colors.LogNorm(vmin=int_range[0], vmax=int_range[1])
        im_cmap = copy.copy(plt.get_cmap('sohoeit195'))
        plot_mat = map_plot.data

    # mask for no data
    if no_data is True:
        plot_mat = np.ma.masked_where(map_plot.data == map_plot.no_data_val, plot_mat)
        im_cmap.set_bad('gray')

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
    xticks = np.arange(x_range[0], x_range[1] + 0.1, 60)

    # setup yticks
    yticks = np.arange(map_plot.y[0], map_plot.y[-1] + 0.01, 0.5)

    if dark:
        plt.style.use('dark_background')
        plt.rcParams.update({
            # "lines.color": "white",
            # "patch.edgecolor": "white",
            "text.color": "darkgray",
            # "axes.facecolor": "white",
            "axes.edgecolor": "darkgray",
            "axes.labelcolor": "darkgray",
            "xtick.color": "darkgray",
            "ytick.color": "darkgray",
            "grid.color": "dimgrey",
            # "figure.facecolor": "black",
            # "figure.edgecolor": "black",
            # "savefig.facecolor": "black",
            # "savefig.edgecolor": "black"
        })
    else:
        plt.rcParams.update({
            "grid.color": "dimgrey",
        })

    # plt.style.use('grayscale')
    # plt.figure(nfig, figsize=[6.4, 3.5], tight_layout=True)
    plt.figure(nfig, figsize=[6.4, 3.5])
    plt.subplots_adjust(left=0.10, bottom=0.11, right=0.97, top=0.95, wspace=0., hspace=0.)
    plt.xlabel("Carrington Longitude")
    plt.ylabel("Sine Latitude")
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(alpha=0.6, linestyle='dashed', lw=0.5)
    # plt.grid(alpha=0.6, linestyle='dotted', lw=1.0)

    if title is not None:
        plt.title(title)

    # Plot the map
    plt.imshow(plot_mat, extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()],
               origin="lower", cmap=im_cmap, aspect=90.0, norm=norm)

    # if its a quality map, call imshow again for each instrument to create overlays
    if quality is True:
        # setup the quality map arrays/colors/norm
        mask_chd = map_type == 'CHD'
        inst_list, mu_dict, color_dict = quality_plot_helper(map_plot, mask_chd=mask_chd)
        norm = mpl.colors.Normalize(vmin=int_range[0], vmax=int_range[1])
        custom_lines = []
        # loop over each instrument, plot the quality map. custom_lines are for the legend.
        for inst_ind, inst in enumerate(inst_list):
            plot_mat = mu_dict[inst]
            im_cmap = plt.get_cmap(color_dict[inst])
            plt.imshow(plot_mat, extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()],
                       origin="lower", cmap=im_cmap, aspect=90.0, norm=norm)
            custom_lines.append(Line2D([0], [0], color=im_cmap(0.7), lw=2))
        # draw the legend (fine tune bbox_to_anchor through trial and error).
        bbox_to_anchor = (-0.085, -0.185)
        plt.legend(custom_lines, inst_list, ncol=3, loc='lower left', fontsize='small',
                   columnspacing=1.0, handlelength=1.60, frameon=False,
                   bbox_to_anchor=bbox_to_anchor)

    # plt.show(block=False)
    plt.savefig(save_path, dpi=dpi)

    plt.close()

    return None


def euv_map_movie(map_info, png_dir, movie_path, map_dir, int_range, fps, dpi=None):
    """
    Take a list of EUV synchronic maps and generate a video.

    The map_info dataframe references one map per row. Each referenced map
    is plotted and saved to png in png_dir. Each png is added to an .mp4
    as a single frame.
    :param map_info: pandas.DataFrame
                     Generally this will be the output of modules.query_euv_maps()
    :param png_dir: str
                    Location of a directory that png files should be written to.
                    These are an intermediate product and may be deleted after execution.
    :param movie_path: str
                       Full path where output video file should be written
                       (not including filename).
    :param map_dir: str
                    Root path to PsiMap database filesystem. This will be combined
                    with relative paths in map_info to locate input maps.
    :param int_range: list (float) [min intensity, max intensity]
                      Minimum and maximum values considered in the plotting
                      colorscale. Values beyond this range will saturate the
                      colorscale.
    :param fps: int
                Frames per second
    :param dpi: int
                Dots per inch for pngs.  If set to None, an automatic estimation
                will be done to preserve the map resolution.
    :return: None
             This routine writes PNGs to png_dir and an MP4 to movie_dir, but has
             no explicit output.
    """
    # num images
    num_maps = map_info.shape[0]
    # generate first frame to establish framesize
    map_index = 0
    row = map_info.loc[map_index]
    # open map object
    map_path = os.path.join(map_dir, row.fname)
    euv_map = psi_dt.read_psi_map(map_path)
    # generate title (timestamp)
    title = row.date_mean.strftime("%Y/%m/%d, %H:%M:%S")
    # generate filename
    frame_filename = "Frame" + str(map_index).zfill(5) + ".png"
    frame_path = os.path.join(png_dir, frame_filename)
    # if no dpi specified, estimate dpi to preserve map resolution
    if dpi is None:
        map_shape = euv_map.data.shape
        width_dpi = map_shape[1]/5.57
        height_dpi = map_shape[0]/2.78
        dpi = np.ceil(max(width_dpi, height_dpi))
    # generate frame image
    map_movie_frame(euv_map, int_range, save_path=frame_path,
                    title=title, dpi=dpi)
    # read image to cv2 format
    cv_im = cv2.imread(frame_path)
    # extract frame size
    frameSize = list(cv_im.shape[0:2])
    frameSize.reverse()
    frameSize = tuple(frameSize)
    # initiate cv2 movie object
    out = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          fps, frameSize)
    # add first image to video
    out.write(cv_im)

    # loop through map files
    for map_index, row in map_info.iloc[1:].iterrows():
        print("Processing map", map_index+1, "of", num_maps)
        # open map object
        map_path = os.path.join(map_dir, row.fname)
        euv_map = psi_dt.read_psi_map(map_path)
        # generate title (timestamp)
        title = row.date_mean.strftime("%Y/%m/%d, %H:%M:%S")
        # generate filename
        frame_filename = "Frame" + str(map_index).zfill(5) + ".png"
        frame_path = os.path.join(png_dir, frame_filename)
        # generate frame image
        map_movie_frame(euv_map, int_range, save_path=frame_path,
                    title=title, dpi=dpi)
        # read image to cv2 format
        cv_im = cv2.imread(frame_path)
        # add image to movie object
        out.write(cv_im)

    # finish and close file
    out.release()
    out = None
    cv2.destroyAllWindows()

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
    plt.imshow(plot_hist, aspect="auto", interpolation='nearest', origin='lower',
               extent=[intensity_bin_edges[0], intensity_bin_edges[-2] + 1., mu_bin_edges[0],
                       mu_bin_edges[-1]])
    plt.xlabel("Pixel log10 intensity")
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
    plt.imshow(norm_hist, aspect="auto", interpolation='nearest', origin='lower',
               extent=[intensity_bin_edges[0], intensity_bin_edges[-1], mu_bin_edges[0],
                       mu_bin_edges[-1]])
    plt.xlabel("Pixel log10 intensity")
    plt.ylabel("mu")
    plt.title(
        "2D Histogram Data Normalized by mu Bin: \n" + "Instrument: " + instrument + " \n " + str(date_obs))
    return None


def Plot1d_Hist(norm_hist, instrument, inst_index, intensity_bin_edges, color_list, linestyle_list, figure,
                xlabel, ylabel, title, save=None):
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
    if save is not None:
        plt.savefig('maps/iit/' + save)
    return None


def quality_plot_helper(map, mask_chd=False):
    """
    Setup the plot arrays for the quality map so that it can be more easily
    dropped into any of our plotting subroutines.

    The data_info pd/dictionary tag is used to determine which pixels in the
    origin image are from each instrument.

    Parameters
    ----------
    map : PsiMap
        PsiMap class structure with the 2D data arrays.
    mask_chd : bool
        Flag to mask the output arrays with the CH map.

    Returns
    -------
    inst_list : list
        List of the instrument names found in the image.
    mu_dict : dict
        Contains the masked numpy arrays of the mu map for each instrument, indexed
        by the instrument names in inst_list.
    color_dict : dict
        Contains the color map for each instrument image, indexed by inst_list.
    """
    # local names for the 2D data arrays
    data_info = map.data_info
    origin_image = map.origin_image
    mu = map.mu

    # get list of origin images
    origins = np.unique(origin_image)

    # create array of strings that is the same shape as euv/chd origin_image
    name_array = np.empty(origin_image.shape, dtype=str)

    # fill the array of strings with instrument names
    for id in origins:
        data_index = np.where(data_info['data_id'] == id)
        instrument = data_info['instrument'].iloc[data_index[0]]
        if len(instrument) != 0:
            name_array = np.where(origin_image != id, name_array, instrument.iloc[0])

    # list of unique instruments in the array
    inst_list = list(data_info.instrument)

    # generate the dictionaries that will be used for plotting (UPDATE THIS FOR NEW INSTRUMENTS)
    color_dict = {'AIA': 'Reds', 'EUVI-A': 'Blues', 'EUVI-B': 'Greens'}
    mu_dict = {}

    # get the masked mu arrays for each instrument
    for inst_ind, inst in enumerate(inst_list):

        # create an array with ones only where its the desired instrument
        use_image = np.zeros(name_array.shape)
        use_image = np.where(name_array == inst, 1.0, use_image)

        # mu values less than 0 become eps value
        eps = 1e-4
        mu_values = mu
        mu_values = np.where(mu_values > 0, mu_values, eps)
        # add mu weighting to data
        plot_data = use_image * mu_values

        # mask the image by the no data value
        no_data_mask = np.where(map.data == map.no_data_val, 0.0, 1.0)
        plot_data = plot_data * no_data_mask

        # if you are doing a coronal hole quality map, mask by 0
        if mask_chd:
            chd_mask = np.where(map.chd >= 0.5, 1.0, 0.0)
            plot_data = plot_data * chd_mask

        # mask invalid values
        plot_data = np.ma.array(plot_data)
        plot_data_masked = np.ma.masked_where(plot_data <= 0, plot_data)

        # add them to the mu dictionary
        mu_dict[inst] = plot_data_masked

    return inst_list, mu_dict, color_dict


def PlotQualityMap(map_plot, origin_image, inst_list, color_list, nfig=None, title=None, map_type=None):
    plot = [None] * len(inst_list)
    for inst_ind, inst in enumerate(inst_list):

        # create usable data array
        use_image = np.zeros(origin_image.shape)
        use_image = np.where(origin_image != inst, use_image, 1.0)
        use_image = np.where(origin_image == inst, use_image, 0)

        # set color palette and normalization
        color_map = color_list[inst_ind]
        im_cmap = plt.get_cmap(color_map)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        # plot the initial image
        if nfig is None:
            cur_figs = plt.get_fignums()
            if not nfig:
                nfig = 0
            else:
                nfig = cur_figs.max() + 1

        # convert map x-extents to degrees
        x_range = [180 * map_plot.x.min() / np.pi, 180 * map_plot.x.max() / np.pi]

        # setup x-axis tickmarks
        xticks = np.arange(x_range[0], x_range[1] + 1, 30)

        # remove non-CH data
        if map_type == 'CHD':
            use_image = np.where(map_plot.data != 0, use_image, 0)

        # mu values less than 0 become 0
        mu_values = map_plot.mu
        mu_values = np.where(mu_values > 0, mu_values, 0.01)
        # add mu weighting to data
        plot_data = use_image * mu_values

        # mask invalid values
        plot_data = np.ma.array(plot_data)
        plot_data_masked = np.ma.masked_where(plot_data <= 0, plot_data)
        plt.figure(nfig)
        plot[inst_ind] = plt.imshow(plot_data_masked,
                                    extent=[x_range[0], x_range[1], map_plot.y.min(), map_plot.y.max()],
                                    origin="lower", cmap=im_cmap, aspect=90.0, norm=norm, interpolation='nearest')
        plt.xlabel("Carrington Longitude")
        plt.ylabel("Sine Latitude")
        plt.xticks(xticks)
        # title plot
        if title is not None:
            plt.title(title)

    cmaps = [plot[ii].cmap for ii in range(len(inst_list))]
    custom_lines = [Line2D([0], [0], color=cmaps[ii](1.), lw=4) for ii in range(len(inst_list))]

    plt.legend(custom_lines, inst_list)
    plt.show(block=False)
    return None
