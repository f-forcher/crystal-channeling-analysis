from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from root_pandas import read_root
import sys
from itertools import islice
import readline as rd
"""
Test on binning data with pandas
"""


arr = pd.Series([1,2,3,4,5,6,7,8,9,10, 1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5, 5,6,4,7])
arr2 = pd.Series([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])

data = pd.DataFrame({"col1": arr, "col2": arr2})

low_range_edge = 0
high_range_edge = 10
# num=6 because we want nbins=5 bins and so we need num = nbins + 1 = 6 bin "borders", namely [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], which results in...
bins = np.linspace(low_range_edge,high_range_edge,num=6)# ...these bins: [(0.0, 2.0] < (2.0, 4.0] < (4.0, 6.0] < (6.0, 8.0] < (8.0, 10.0]

# bin centers, basically we create a list which is the bin edges above, moved forward by half bin width, and then
# we remove the last one with [:-1] because the last one is half bin after the last edge at high_range_edge
bincenters = np.linspace(low_range_edge+1,high_range_edge+1,num=6)[:-1]

# Binned data, len(data) long list where the original data is replaced by the respective bin center
# [1,1,2,2,...]
# binned_data = pd.DataFrame(pd.cut(data.col1, bins,labels = list(bincenters)))
binned_data = pd.cut(data.col1, bins,labels = list(bincenters))
data["binned"] = binned_data
gruppi = data.groupby("binned")

# Histogram aka bidimensional nbin-long list of points,
# whose x coordinate is bin center and whose y coord is bin content aka how many data in the bin
# [(1,2),
#  (2,2), ...]
histogram = pd.DataFrame(pd.value_counts(data.binned))
# histogram.index.values gives the x as numbers.
