 # Function to bin a dataframe using groupby
import numpy as np
import pandas as pd

def bin2D_dataframe(data, x_column, y_column, x_range, y_range, nxbins, nybins):
    """
    Bin a dataframe along 2 axes, returning a groupby object.

    data: input dataframe to bin.
    x_column: Column name of "data" to use as X axis (first grouping)
    y_column: Column name of "data" to use as Y axis (second grouping)
    x_range: a 2-ple for the X range: (low_xedge, high_xedge)
    y_range: a 2-ple for the Y range: (low_yedge, high_yedge)
    nxbins:  number of bins to divide the x_range in
    nybins:  number of bins to divide the y_range in

    return: a groupby object
    """
    low_xrange_edge = x_range[0]
    high_xrange_edge = x_range[1]

    low_yrange_edge = y_range[0]
    high_yrange_edge = y_range[1]

    # num=6 because we want nbins=5 bins and so we need num = nbins + 1 = 6 bin "borders"
    xbins = np.linspace(low_xrange_edge,high_xrange_edge,num=nxbins + 1)
    ybins = np.linspace(low_yrange_edge,high_yrange_edge,num=nybins + 1)
    delta_xbins = xbins[1] - xbins[0]
    delta_ybins = ybins[1] - ybins[0]

    # bin centers, basically we create a list which is the bin edges above, moved forward by half bin width, and then
    # we remove the last one with [:-1] because the last one is half bin after the last edge at high_range_edge
    xbincenters = np.linspace(low_xrange_edge + delta_xbins/2, high_xrange_edge + delta_xbins/2, num=nxbins + 1).round(5)[:-1]
    ybincenters = np.linspace(low_yrange_edge + delta_ybins/2, high_yrange_edge + delta_ybins/2, num=nybins + 1).round(5)[:-1]

    # TODO Check efficiency of this method
    gruppi = data.groupby([pd.cut(data.loc[:,x_column], xbins, labels = xbincenters),
                         pd.cut(data.loc[:,y_column], ybins, labels = ybincenters)])

    return gruppi
