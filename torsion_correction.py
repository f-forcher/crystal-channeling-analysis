# SYSTEM LIBS
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from root_pandas import read_root
import sys
from itertools import islice
from scipy import signal # To find peaks, for edge-detecting the crystal
import os

# MY LIBS
import editable_input as ei # My script for editable text input

################# READ THE PARAMETERS
# Read the parameters from the .csv
# .csv example:
#
# parameter_name	value
# init_scan	1570674.0
# xmin	0.0
# xmax	0.475
#

file_name = sys.argv[1]

if os.path.isfile('crystal_analysis_parameters.csv'):
    parameters_table = pd.read_csv("crystal_analysis_parameters.csv", sep="\t", index_col=0)
else: #
    raise FileNotFoundError("[ERROR]: File crystal_analysis_parameters.csv not "
                            "found. Create it with save_as_hdf.py")

cut_left = float(parameters_table.loc['xmin'])
cut_right = float(parameters_table.loc['xmax'])
init_scan = float(parameters_table.loc['init_scan'])
#################



################# READ THE DATA
# Read the data $chunksize lines at a time. evts=iterator on the groups of lines
# DATAFRAME COLUMNS:
#  'Time', 'Date', 'Event_run', 'Event_evtnum', 'Event_nuclear',
#        'Event_nuclearRaw', 'GonioPos_x', 'GonioPos_y', 'GonioPos_z',
#        'MultiHits_thetaIn_x', 'MultiHits_thetaIn_y',
#        'MultiHits_thetaInErr_x', 'MultiHits_thetaInErr_y',
#        'MultiHits_d0_x', 'MultiHits_d0_y', 'MultiHits_d0Err_x',
#        'MultiHits_d0Err_y', 'Tracks_thetaIn_x', 'Tracks_thetaIn_y',
#        'Tracks_thetaOut_x', 'Tracks_thetaOut_y', 'Tracks_thetaInErr_x',
#        'Tracks_thetaInErr_y', 'Tracks_thetaOutErr_x',
#        'Tracks_thetaOutErr_y', 'Tracks_d0_x', 'Tracks_d0_y',
#        'Tracks_d0Err_x', 'Tracks_d0Err_y', 'Tracks_chi2_x',
#        'Tracks_chi2_y', 'SingleTrack', 'MultiHit'
chunksize = 2000000
interesting_columns = ["Tracks_d0_y", "Tracks_thetaOut_x", "Tracks_thetaIn_x"]
# Important to remember that the columns need to be indexed with
# data_columns=[...] when .hdf is created, to be able to use where on them
cuts_and_selections = ["SingleTrack == 1", "Tracks_d0_x > cut_left", "Tracks_d0_x < cut_right"]

evts = pd.read_hdf(file_name, chunksize = chunksize, columns=interesting_columns, where=cuts_and_selections)

loaded_rows = 0
events = pd.DataFrame(columns=interesting_columns)
print("[LOG]: Loading data...")
for df in evts:
    loaded_rows = loaded_rows + df.shape[0]
    print("\n[LOG] loaded ", loaded_rows, " rows\n")
    df.info()
    events = events.append(df,ignore_index=True) # join inner maybe unnecessary here
    # break; # Uncomment to get only the first chunk
print("\n[LOG]: Loaded data!\n")
events.info()
#################

events["Tracks_thetaIn_x_CORRECTED"] = events.loc[:,"Tracks_thetaIn_x"].values + \
                                       init_scan*1e-6

events.info()

################# BIN THE DATA
# low_xrange_edge = -4
# high_xrange_edge = 4
# nbins = 12
#
# # num=6 because we want nbins=5 bins and so we need num = nbins + 1 = 6 bin "borders"
# xbins = np.linspace(low_range_edge,high_range_edge,num=nbins+1)
# ybins = np.linspace(low_range_edge,high_range_edge,num=nbins + 1)
# delta_xbins = xbins[1] - xbins[0]
# delta_ybins = ybins[1] - ybins[0]
#
# # bin centers, basically we create a list which is the bin edges above, moved forward by half bin width, and then
# # we remove the last one with [:-1] because the last one is half bin after the last edge at high_range_edge
# xbincenters = np.linspace(low_range_edge + delta_xbins/2, high_range_edge + delta_xbins/2, num=nbins + 1)[:-1]
# ybincenters = np.linspace(low_range_edge + delta_ybins/2, high_range_edge + delta_ybins/2,num=nbins + 1)[:-1]
#
# gruppi = df.groupby([pd.cut(df.X, xbins, labels = xbincenters),
#                      pd.cut(df.Y, ybins, labels = ybincenters)])

#################
