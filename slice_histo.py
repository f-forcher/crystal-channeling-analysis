from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from itertools import islice
from scipy import signal # To find peaks, for edge-detecting the crystal
import os

import editable_input as ei # My script for editable text input

"""
Script #2
Plot x vs delta_theta_x to select a geometric cut for the crystal.

cli argument 1: the name of the .root.hdf file
"""

def robust_standard_deviation(x, lowest_percentage, highest_percentage):
    """
    Calculate the standard deviation on x using only the data contained
    in the percentiles between lower_perc and upper_perc.

    x: array-like of numbers
    lowest_percentage: lowest percentage to include (0<=lowest_percentage<=100)
    highest_percentage: highest percentage to include (0<=highest_percentage<=s100)
    return: robust standard deviation (scalar)
    """
    first_percentile = np.percentile(x, lowest_percentage)
    last_percentile = np.percentile(x, highest_percentage)
    #
    x_reduced = x[(x>=first_percentile) & (x<=last_percentile)]
    #
    robust_std = x_reduced.std()
    return robust_std



plt.ion()
file_name = sys.argv[1]


# LOAD DATAFRAME FROM HDF FILE
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
interesting_columns = ["Tracks_d0_x", "Tracks_thetaOut_x", "Tracks_thetaIn_x", "SingleTrack"]
evts = pd.read_hdf(file_name, chunksize = chunksize, columns=interesting_columns, where=["SingleTrack == 1"])#,"Tracks_d0_y>-1","Tracks_d0_y < 1"])TODO test peaks


# events = evts.next() # python 2!
# events = next(islice(evts, 1))
loaded_rows = 0
events = pd.DataFrame(columns=interesting_columns)
print("[LOG]: Loading data...")
for df in evts:
    loaded_rows = loaded_rows + df.shape[0]
    print("\n[LOG] loaded ", loaded_rows, " rows\n")
    df.info()
    events = events.append(df,ignore_index=True) # join inner maybe unnecessary here
    # break; # Uncomment to get only the first chunk
print("[LOG]: Loaded data!")


events["Delta_Theta_x"] = events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values

histo_range_x = [-5,5] # [mm]
histo_range_y = [-100e-6,100e-6] # [rad]
numbin = 400


################# BIN THE DATA
# num=numbin+1 because we want numbin bins and so we need num = numbin + 1 bin "borders", see test_bins.py for examples
x_bin_borders = np.linspace(histo_range_x[0], histo_range_x[1], num=numbin+1)
# bin centers, basically we create a list which is the bin edges above, moved forward by half bin width, and then
# we remove the last one with [:-1] because the last one is half bin after the last edge high_range_edge
x_bin_centers = np.linspace(histo_range_x[0], histo_range_x[1], num=numbin+1)[:-1]
x_bin_width = x_bin_centers[1]-x_bin_centers[0]


events["BINNED_Tracks_d0_x"] = pd.cut(events.loc[:,'Tracks_d0_x'].values, x_bin_borders, labels = x_bin_centers)
x_slices = events.groupby("BINNED_Tracks_d0_x") # TODO scoprire come funzionano sti oggetti "gruppi"

#################


################# FIND THE CRYSTAL BORDERS (or at least try to)
# Because the crystal increases suddenly the scattering of the particle, increasing their
# y standard deviation, the x edge of the crystal is when the scattering suddenly
# increases.
#
# To detect the edges, we then perform a wavelet peak detect on the x derivative of the standard deviation
# of the slices, namely d/dx(x_slices["Delta_Theta_x"].std), and then we find the two strongest peaks.
# We then shave off 5% from each side, to be sure to be inside the crystsal, and
# then these results are proposed to the user, that can decide now to use them or not.

# Find the edges in the intensity (square of the derivative)
peak_index = signal.find_peaks_cwt(np.diff(x_slices["Delta_Theta_x"]
             .apply(lambda x: robust_standard_deviation(x,10,90)).values)**2,
                                   [0.09])
                                  # 0.09 peak width because the peaks are very sharp.
# Notice that peak_index is in the diff array, which is 1 shorter that x_slices[...]
# So later we will need to add 1 to get the "correct" 'x's

# peaks = pd.DataFrame({'x': x_bin_centers[peak_index + 1], # +1: see above
#                       'intensity': (np.diff(x_slices["Delta_Theta_x"]
#                       .std().values)**2)[peak_index]}) # We dont need +1 here
peaks = pd.DataFrame({'x': x_bin_centers[peak_index + 1], # +1: see above
                      'intensity': (np.diff(x_slices["Delta_Theta_x"]
                      .apply(lambda x: robust_standard_deviation(x,10,90))
                      .values)**2)[peak_index]}) # We dont need +1 here


# Get the two strongest peaks, by sorting desceningly by intensity, and getting
# the first two entries
crystal_edge_locations = peaks.sort_values("intensity",ascending=False)[:2]

crystal_edge_locations.sort_values("x",inplace=True)
left_edge = crystal_edge_locations.iloc[0]["x"]  # [mm]
right_edge = crystal_edge_locations.iloc[1]["x"] # [mm]
crystal_length = right_edge - left_edge

# Get the inside of the crystal to avoid ambiguous bins
side_bins_to_cut = 1 # How many bins to remove from each side, to avoid border
                     # ambiguities
proposed_cut_left = left_edge + side_bins_to_cut * x_bin_width
proposed_cut_right = right_edge - side_bins_to_cut * x_bin_width
#################


################ PLOT HISTOGRAM
plt.hist2d(events.loc[:,'Tracks_d0_x'].values ,events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,\
bins=numbin, norm=LogNorm(), range=[histo_range_x,histo_range_y])
plt.axvline(x=proposed_cut_left, linestyle="dashed", color='Crimson', label="")
plt.axvline(x=proposed_cut_right, linestyle="dashed", color='Crimson', label="")


# print(events)
plt.colorbar()
plt.show()
#################


################# ASK USER TO CONFIRM CUTS
# We "propose" these cuts to the user, which can then decide to modify or
# accept them.
print("Proposed cuts [mm]:\n\t xmin = {:.4f}".format(proposed_cut_left),
      "\n\t xmax = {:.4f}".format(proposed_cut_right))
print("What cuts do you want to use?")
cut_left  = float(ei.edit_input("xmin = ", round(proposed_cut_left,4)))
cut_right = float(ei.edit_input("xmax = ", round(proposed_cut_right,4)))
#################

################# SAVE THE CUT TO FILE
# # Save them into the parameters table, a file read by all scripts to get
# # their data out.

# parameters_table = pd.DataFrame({'parameter_name': ['xmin', 'xmax'],
#                                  'value': [proposed_cut_left, proposed_cut_right]})
if os.path.isfile('crystal_analysis_parameters.csv'):
    # read up the already existing parameters (like init_scan)
    parameters_table = pd.read_csv("crystal_analysis_parameters.csv", sep="\t", index_col=0)
else: #
    raise FileNotFoundError("[ERROR]: File crystal_analysis_parameters.csv not "
                            "found. Create it with save_as_hdf.py")

parameters_table.loc['xmin'] = cut_left
parameters_table.loc['xmax'] = cut_right



parameters_table.to_csv("crystal_analysis_parameters.csv",sep='\t')

#################
