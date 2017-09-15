from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from root_pandas import read_root
import sys
from itertools import islice

df = pd.DataFrame(np.random.normal(loc=0, scale=2, size=(1500, 2)),
                     columns=['X', 'Y'])


low_range_edge = -4
high_range_edge = 4
nbins = 12

# num=6 because we want nbins=5 bins and so we need num = nbins + 1 = 6 bin "borders"
xbins = np.linspace(low_range_edge,high_range_edge,num=nbins+1)
ybins = np.linspace(low_range_edge,high_range_edge,num=nbins + 1)
delta_xbins = xbins[1] - xbins[0]
delta_ybins = ybins[1] - ybins[0]

# bin centers, basically we create a list which is the bin edges above, moved forward by half bin width, and then
# we remove the last one with [:-1] because the last one is half bin after the last edge at high_range_edge
xbincenters = np.linspace(low_range_edge + delta_xbins/2, high_range_edge + delta_xbins/2, num=nbins + 1)[:-1]
ybincenters = np.linspace(low_range_edge + delta_ybins/2, high_range_edge + delta_ybins/2,num=nbins + 1)[:-1]

gruppi = df.groupby([pd.cut(df.X, xbins, labels = xbincenters),
                     pd.cut(df.Y, ybins, labels = ybincenters)])

# colormaps list at https://matplotlib.org/1.3.0/examples/color/colormaps_reference.html
# To reverse append "_r" to the name
plt.hist2d(df.X,df.Y,nbins,range=[[-4,4],[-4,4]], cmap = "gnuplot2",alpha=0.8)
for i,v in gruppi:
        plt.plot(gruppi.get_group(i)["X"],gruppi.get_group(i)["Y"],".")

plt.show()
