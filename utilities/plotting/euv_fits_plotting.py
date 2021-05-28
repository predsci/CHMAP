"""
Methods to simplify making routine / standard plots.
"""
import numpy as np
import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord

# simply importing sunpy.cm will register the sunpy colortables with matplotlib
import sunpy.cm

# determine the default matplotlib backend when the module is loaded
mpl_backend_default = matplotlib.get_backend()

# specify the backend for non-interactive plots
mpl_backend_non_interactive = 'agg'


def plot_image_rs(map_in, xrange=[-1.4, 1.4], yrange=[-1.4, 1.4], log_min=0.5, log_max=3.5,
                  cmap_name=None, outfile=None, dpi=100, save_interactive=False):
    """
    Quick method to plot a map by specifying the x and y range in SOLAR coordinates.

    - xrange and yrange are two elements lists or tuples that specify the solar coords in Rs
      - e.g. xrange=[-1.3, -1.3], yrange=[-1.3, 1.3]

    - if a output file is specified, it will switch to a non-interactive backend and save the file
      without showing the plot (unless save_interactive=True).

    - cmap_name (optional) is a string that specifies a sunpy or matplotlib colormap
    """
    # I don't want to modify the input map at all --> copy the map object just in case
    map = copy.deepcopy(map_in)

    # info from the map
    rs_obs = map.rsun_obs

    # get the coordinate positions of the x and y ranges
    x0 = xrange[0]*rs_obs.value*u.arcsec
    x1 = xrange[1]*rs_obs.value*u.arcsec
    y0 = yrange[0]*rs_obs.value*u.arcsec
    y1 = yrange[1]*rs_obs.value*u.arcsec
    bot_left = SkyCoord(x0, y0, frame=map.coordinate_frame)
    top_right = SkyCoord(x1, y1, frame=map.coordinate_frame)

    # experiment with different styles of plotting the x and y window
    # using "limits" lets you plot outside of the image window, which can be important
    # for aligning images.
    plot_method = 'limits'

    if plot_method == 'submap':
        map = map.submap(bot_left, top_right)

    # setup the optional colormap
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        map.plot_settings['cmap'] = cmap

    # Set the map plot min/max
    pmin = 10.0**(log_min)
    pmax = 10.0**(log_max)
    map.plot_settings['norm'] = colors.LogNorm(pmin, pmax)

    # Change the colormap so undefined values don't show up white
    map.plot_settings['cmap'].set_bad(color='black')

    # if saving a file, don't use the interactive backend
    if outfile is not None and not save_interactive:
        matplotlib.use(mpl_backend_non_interactive)

    # setup the figure
    fig = plt.figure(figsize=(10, 9))

    # Manually specify the axis (vs. getting through map.plot) this way you have more control
    axis = WCSAxes(fig, [0.1, 0.025, 0.95, 0.95], wcs=map.wcs)
    fig.add_axes(axis)  # note that the axes have to be explicitly added to the figure

    # plot the image
    map.plot(axes=axis)

    # example for adjusting the tick spacing (see astropy examples for WCSAxes)
    custom_ticks = False
    if custom_ticks:
        spacing = 500.*u.arcsec
        axis.coords[0].set_ticks(spacing=spacing)
        axis.coords[1].set_ticks(spacing=spacing)

    # if plot is NOT a submap, compute the pixel positions and change the matplotlib limits
    if plot_method == 'limits':
        pp_bot_left = map.world_to_pixel(bot_left)
        pp_top_right = map.world_to_pixel(top_right)
        axis.set_xlim(left=pp_bot_left.x.value, right=pp_top_right.x.value)
        axis.set_ylim(bottom=pp_bot_left.y.value, top=pp_top_right.y.value)

    # plot the colorbar
    plt.colorbar()

    # save the plot (optional)
    if outfile is not None:
        print("Saving image plot to: " + outfile)
        fig.savefig(outfile, dpi=dpi)
        # revert to the default MPL backend
        if not save_interactive:
            plt.close()
            matplotlib.use(mpl_backend_default)
        else:
            plt.show()
    else:
        plt.show()


def plot_image_rs_full(map_in, xrange=[-1.4, 1.4], yrange=[-1.4, 1.4], log_min=0.5, log_max=3.5,
                       cmap_name=None, outfile=None, dpi=100, save_interactive=False):
    """
    Quick method to plot a map by specifying the x and y range in SOLAR coordinates.
    - Unlike plot_image_rs, here the image fills the entire frame, with no outside annotations
      like axes labels or colorbars

    - xrange and yrange are two elements lists or tuples that specify the solar coords in Rs
      - e.g. xrange=[-1.3, -1.3], yrange=[-1.3, 1.3]

    - if a output file is specified, it will switch to a non-interactive backend and save the file
      without showing the plot (unless save_interactive=True).

    - cmap_name (optional) is a string that specifies a sunpy or matplotlib colormap

    ToDo:
      - put white annotations in the corners that describe the image (time, inst, clon, b0)
      - overplot some solar grid lines (e.g. the lat=0 line and/or various clons)
    """
    # I don't want to modify the input map at all --> copy the map object just in case
    map = copy.deepcopy(map_in)

    # info from the map
    rs_obs = map.rsun_obs

    # get the coordinate positions of the x and y ranges
    x0 = xrange[0]*rs_obs.value*u.arcsec
    x1 = xrange[1]*rs_obs.value*u.arcsec
    y0 = yrange[0]*rs_obs.value*u.arcsec
    y1 = yrange[1]*rs_obs.value*u.arcsec
    bot_left = SkyCoord(x0, y0, frame=map.coordinate_frame)
    top_right = SkyCoord(x1, y1, frame=map.coordinate_frame)

    # experiment with different styles of plotting the x and y window
    # using "limits" lets you plot outside of the image window, which can be important
    # for aligning images.
    plot_method = 'limits'

    if plot_method == 'submap':
        map = map.submap(bot_left, top_right)

    # setup the optional colormap
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        map.plot_settings['cmap'] = cmap

    # Set the map plot min/max
    pmin = 10.0**(log_min)
    pmax = 10.0**(log_max)
    map.plot_settings['norm'] = colors.LogNorm(pmin, pmax)

    # Change the colormap so undefined values don't show up white
    map.plot_settings['cmap'].set_bad(color='black')

    # if saving a file, don't use the interactive backend
    if outfile is not None and not save_interactive:
        matplotlib.use(mpl_backend_non_interactive)

    # setup the figure
    fig = plt.figure(figsize=(9, 9))

    # Manually specify the axis (vs. getting through map.plot) this way you have more control
    axis = WCSAxes(fig, [0.0, 0.0, 1.0, 1.0], wcs=map.wcs)
    fig.add_axes(axis)  # note that the axes have to be explicitly added to the figure

    # plot the image
    map.plot(axes=axis)

    # example for adjusting the tick spacing (see astropy examples for WCSAxes)
    custom_ticks = True
    if custom_ticks:
        spacing = map.rsun_obs
        axis.coords[0].set_ticks(spacing=spacing)
        axis.coords[1].set_ticks(spacing=spacing)

    # if plot is NOT a submap, compute the pixel positions and change the matplotlib limits
    if plot_method == 'limits':
        pp_bot_left = map.world_to_pixel(bot_left)
        pp_top_right = map.world_to_pixel(top_right)
        axis.set_xlim(left=pp_bot_left.x.value, right=pp_top_right.x.value)
        axis.set_ylim(bottom=pp_bot_left.y.value, top=pp_top_right.y.value)

    # save the plot (optional)
    if outfile is not None:
        print("Saving image plot to: " + outfile)
        fig.savefig(outfile, dpi=dpi)
        # revert to the default MPL backend
        if not save_interactive:
            plt.close()
            matplotlib.use(mpl_backend_default)
        else:
            plt.show()
    else:
        plt.show()


def plot_alignment(map_in, log_min=0.5, log_max=3.5, cmap_name=None, outfile=None,
                   dpi=100, save_interactive=False):
    """
      Quick subroutine to take a sunpy map and plot a view of each limb to asses alignment.

      - Here i was exploring how to use subplots and their compatibility astropy WCSAxes

        - In the end the subplots required a lot of fine tuning.

        - A major issue seems to be with the subplot axis type I lose some of the WCSAxes funcitonality
          - could be i dont' understand it well enough, but my conclusion is AVOID SUBPLOTS
          - however, i had gotten this far...

        - Also, plotting the lines over the zoomed in images required manually changing the x_lim
          and y_lims with pixel coordinates. This is pretty crappy --> there's got to be a better way.

      - cmap_name (optional) is a string that specifies a sunpy or matplotlib colormap
    """
    # I don't want to accidentally modify the input map --> copy the map object just in case
    map = copy.deepcopy(map_in)

    # info from the map
    rs_obs = map.rsun_obs

    # size of the window for the limb plots
    winsize = 40*u.arcsec

    def _set_log_scaling(map):
        """
        Sub function to set the plot scaling plot
        """
        # setup the optional colormap
        if cmap_name is not None:
            cmap = plt.get_cmap(cmap_name)
            map.plot_settings['cmap'] = cmap

        # Set the map plot min/max
        pmin = 10.0**(log_min)
        pmax = 10.0**(log_max)
        map.plot_settings['norm'] = colors.LogNorm(pmin, pmax)

        # Change the colormap so undefined values don't show up white
        map.plot_settings['cmap'].set_bad(color='black')

    # Function to plot limb circles for plotting alignment
    def _plot_limb_circles(map, axis):
        """
        Sub function to plot the limb circle over a sub map
        """
        # construct the lines for plotting
        rs_obs = map.rsun_obs
        npts = 1000
        angles = np.linspace(0.0, np.pi*2, npts)
        x = np.cos(angles)*rs_obs
        y = np.sin(angles)*rs_obs
        c1 = SkyCoord(x*1.0, y*1.0, frame=map.coordinate_frame)
        c2 = SkyCoord(x*1.01, y*1.01, frame=map.coordinate_frame)
        c3 = SkyCoord(x*1.02, y*1.02, frame=map.coordinate_frame)
        lines = [c1, c2, c3]
        # set their styles
        ls = '-'
        lc = 'm'
        la = 0.5
        lw = 1.5
        # turn off axis labels
        axis.set_xlabel('')
        axis.set_ylabel('')
        # plot the lines
        for line in lines:
            axis.plot_coord(line, color=lc, linestyle=ls, alpha=la, linewidth=lw)
        # set the limits of the plot to be the limits of the image (otherwise it will show the whole line).
        # for more fine tuning you could use the map's world_to_pixel method
        axis.set_xlim(left=0, right=map.data.shape[1])
        axis.set_ylim(bottom=0, top=map.data.shape[0])

    # if saving a file, don't use an interactive backend unless you want to
    if outfile is not None and not save_interactive:
        matplotlib.use(mpl_backend_non_interactive)

    # setup the figure
    fig = plt.figure(figsize=(15, 5))

    # set up the subgrid (python orders it y,x ... ugh)
    subgrid = (2, 6)

    # draw the full sun image
    _set_log_scaling(map)
    ax1 = plt.subplot2grid(subgrid, (0, 0), colspan=2, rowspan=2, projection=map)
    map.plot()
    plt.colorbar()
    # manually adjust the position to make it look better (plt.tight_layout doesn't play well with WCSAxes)
    frac = 0.80
    location = (0 + (1 - frac)/3.5, 0 + (1 - frac)/2, 1./3.*frac, frac)  # [left, bottom, width, height]]
    ax1.set_position(location)

    # draw the north limb
    bot_left = SkyCoord(-2*winsize, rs_obs - winsize, frame=map.coordinate_frame)
    top_right = SkyCoord(2*winsize, rs_obs + winsize, frame=map.coordinate_frame)
    submap = map.submap(bot_left, top_right)
    _set_log_scaling(submap)
    ax = plt.subplot2grid(subgrid, (0, 2), colspan=2, projection=submap)
    submap.plot()
    ax.set_title('North Limb')
    offset_x = 0.01
    offset_y = 0.02
    scale_fac = 0.9
    box = ax.get_position()
    pos = (box.x0 + offset_x, box.y0 + offset_y, box.width*scale_fac, box.height*scale_fac)
    ax.set_position(pos)
    _plot_limb_circles(submap, ax)

    # draw the south limb
    bot_left = SkyCoord(-2*winsize, -rs_obs - winsize, frame=map.coordinate_frame)
    top_right = SkyCoord(2*winsize, -rs_obs + winsize, frame=map.coordinate_frame)
    submap = map.submap(bot_left, top_right)
    _set_log_scaling(submap)
    ax = plt.subplot2grid(subgrid, (1, 2), colspan=2, projection=submap)
    submap.plot()
    ax.set_title('South Limb')
    offset_x = 0.01
    offset_y = -0.02
    scale_fac = 0.9
    box = ax.get_position()
    pos = (box.x0 + offset_x, box.y0 + offset_y, box.width*scale_fac, box.height*scale_fac)
    ax.set_position(pos)
    _plot_limb_circles(submap, ax)

    # draw the east limb
    bot_left = SkyCoord(-rs_obs - winsize, -2*winsize, frame=map.coordinate_frame)
    top_right = SkyCoord(-rs_obs + winsize, 2*winsize, frame=map.coordinate_frame)
    submap = map.submap(bot_left, top_right)
    _set_log_scaling(submap)
    ax = plt.subplot2grid(subgrid, (0, 4), rowspan=2, projection=submap)
    submap.plot()
    ax.set_title('East Limb')
    offset_x = 0.01
    offset_y = 0.00
    scale_fac = 0.97
    box = ax.get_position()
    pos = (box.x0 + offset_x, box.y0 + offset_y, box.width*scale_fac, box.height*scale_fac)
    ax.set_position(pos)
    _plot_limb_circles(submap, ax)
    # help(ax)
    ax.tick_params(spacing=40*u.arcsec)

    # draw the west limb
    bot_left = SkyCoord(rs_obs - winsize, -2*winsize, frame=map.coordinate_frame)
    top_right = SkyCoord(rs_obs + winsize, 2*winsize, frame=map.coordinate_frame)
    submap = map.submap(bot_left, top_right)
    _set_log_scaling(submap)
    ax = plt.subplot2grid(subgrid, (0, 5), rowspan=2, projection=submap)
    submap.plot()
    ax.set_title('West Limb')
    offset_x = 0.08
    offset_y = 0.00
    scale_fac = 0.97
    box = ax.get_position()
    pos = (box.x0 + offset_x, box.y0 + offset_y, box.width*scale_fac, box.height*scale_fac)
    ax.set_position(pos)
    _plot_limb_circles(submap, ax)

    # save the plot (optional)
    if outfile is not None:
        print("Saving limb plot to: " + outfile)
        fig.savefig(outfile, dpi=dpi)
        # revert to the default MPL backend
        if not save_interactive:
            plt.close()
            matplotlib.use(mpl_backend_default)
        else:
            plt.show()
    else:
        plt.show()
