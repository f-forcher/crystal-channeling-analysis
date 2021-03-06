from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from itertools import islice
from scipy import signal # To find peaks, for edge-detecting the crystal
import os

import editable_input as ei # My script for editable text input
import mie_utils as my

"""
Script #2
Plot x vs delta_theta_x to select a geometric cut for the crystal.

cli argument 1: the name of the .root.hdf file
"""



def robust_standard_deviation(x, lowest_percentage, highest_percentage, low_data_threshold):
    """
    Calculate the standard deviation on x using only the data contained
    in the percentiles between lower_perc and upper_perc.

    x: array-like of numbers
    lowest_percentage: lowest percentage to include (0<=lowest_percentage<=100)
    highest_percentage: highest percentage to include (0<=highest_percentage<=s100)
    return: robust standard deviation (scalar)
    """

    if len(x) < low_data_threshold:
        print("[LOG]: too few data in slice, std set to zero")
        return 0

    first_percentile = np.percentile(x, lowest_percentage)
    last_percentile = np.percentile(x, highest_percentage)
    #CH3287095
    x_reduced = x[(x>=first_percentile) & (x<=last_percentile)]
    #
    robust_std = x_reduced.std()
    return robust_std


######################################
################# MAIN

################# GET CLI ARGUMENTS AND FIND THE CORRESPONDING CONF FILE
# Get CLI arguments of the scripts
plt.ion()
file_name = sys.argv[1]
crystal_name = sys.argv[2]
run_number = sys.argv[3]
particle_name = sys.argv[4]
particle_energy = sys.argv[5] # [GeV]

# Use a run specific params file, otherwise look for a crystal specific one,
# otherwise use the general one.
if os.path.isfile(run_number + '_analysis_configuration_params.csv'):
    analysis_configuration_params_file = run_number + '_analysis_configuration_params.csv'
elif os.path.isfile(crystal_name + '_analysis_configuration_params.csv.csv'):
    analysis_configuration_params_file = crystal_name + '_analysis_configuration_params.csv'
else:
    analysis_configuration_params_file = 'analysis_configuration_params.csv'
print("[LOG]: Reading crystal analysis parameters from ", analysis_configuration_params_file)


# Check if the run number is in the actual data file name, otherwise print a
# warning
if '_'+run_number+'_' not in file_name:
    print("[WARNING]: '_{}_' not found in file name '{}', maybe check if "
          "correct run number or correct file.".format(run_number, file_name))
#################

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
cut_y_low, cut_y_high = my.get_from_csv(analysis_configuration_params_file,
                                        "cut_y_low", "cut_y_high")
evts = pd.read_hdf(file_name, chunksize = chunksize, columns=interesting_columns,
                   where=["SingleTrack == 1", "Tracks_d0_y > cut_y_low",
                          "Tracks_d0_y < cut_y_high"])


# events = evts.next() # python 2!
# events = next(islice(evts, 1))
loaded_rows = 0
events = pd.DataFrame(columns=interesting_columns)
print("[LOG]: Loading data...")
for df in evts:
    loaded_rows = loaded_rows + df.shape[0]
    print("\n[LOG] loaded ", loaded_rows, " rows\n")
    # df.info() # Print info on the loaded rows
    events = events.append(df,ignore_index=True) # join inner maybe unnecessary here
    # break; # Uncomment to get only the first chunk
print("[LOG]: Loaded data!")


events["Delta_Theta_x"] = events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values



histo_range_x_low, histo_range_x_high = my.get_from_csv(analysis_configuration_params_file,
                                        "geocut_histo_range_x_low", "geocut_histo_range_x_high")
histo_range_dtx_low, histo_range_dtx_high = my.get_from_csv(analysis_configuration_params_file,
                                        "geocut_histo_range_dtx_low", "geocut_histo_range_dtx_high")
geocut_numberofbin_per_axis = my.get_from_csv(analysis_configuration_params_file,
                                        "geocut_numberofbin_per_axis")
histo_range_x = [histo_range_x_low, histo_range_x_high] # [mm]
histo_range_dtx = [histo_range_dtx_low, histo_range_dtx_high] # [murad]
numbin = geocut_numberofbin_per_axis

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


low_percentage, high_percentage, low_data_threshold = my.get_from_csv(analysis_configuration_params_file,
                                        "geocut_std_low_percentage", "geocut_std_high_percentage", "geocut_std_low_data_threshold")
robust_std = lambda x: robust_standard_deviation(x,low_percentage,high_percentage,low_data_threshold)
std_slices = x_slices["Delta_Theta_x"].apply(robust_std).values

x_for_derivative = x_bin_borders[1:-1]
std_slices_derivative_intensity = np.diff(std_slices)**2

peak_index = signal.find_peaks_cwt(np.diff(x_slices["Delta_Theta_x"].apply(robust_std).values)**2,
                                   [0.09])
                                  # 0.09 peak width because the peaks are very sharp.
# Notice that peak_index is in the diff array, which is 1 shorter that x_slices[...]
# So later we will need to add 1 to get the "correct" 'x's

# peaks = pd.DataFrame({'x': x_bin_centers[peak_index + 1], # +1: see above
#                       'intensity': (np.diff(x_slices["Delta_Theta_x"]
#                       .std().values)**2)[peak_index]}) # We dont need +1 here
peaks = pd.DataFrame({'x': x_bin_centers[peak_index + 1], # +1: see above
                      'intensity': (np.diff(x_slices["Delta_Theta_x"]
                      .apply(robust_std)
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
plt.figure()
plt.hist2d(events.loc[:,'Tracks_d0_x'].values, 1e6*(events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values),\
bins=numbin, norm=LogNorm(), range=[histo_range_x,histo_range_dtx])
plt.axvline(x=proposed_cut_left, linestyle="dashed", color='Crimson', label="")
plt.axvline(x=proposed_cut_right, linestyle="dashed", color='Crimson', label="")

plt.suptitle(r"Crystal {}, run {} — {} {} GeV".format(crystal_name, run_number, particle_name, particle_energy),fontweight='bold')
plt.title(r"Histogram: {}".format(crystal_name, run_number, r"$x_{in}$ vs $\Delta \theta_{x}$"))
plt.xlabel(r'$x_{in}\ [mm]$')
plt.ylabel(r'$\Delta \theta_{x}\ [\mu rad]$')
# print(events)
plt.colorbar()
#plt.tight_layout()
plt.savefig("latex/img/geocuts.pdf")
plt.show()
#################

################ PLOT SLICE STANDARD DEVIATION
plt.figure()
plt.plot(x_bin_centers,std_slices)

plt.title(r"Crystal {}, run {} - Robust standard deviation in slice".format(crystal_name, run_number))
plt.xlabel(r'$x_{in}\ [mm]$')
plt.ylabel('Standard deviation (5% - 95%)')
# print(events)
plt.tight_layout()
plt.savefig("latex/img/std_per_slice.pdf")
plt.show()
#################

################ PLOT SLICE STANDARD DEVIATION DERIVATIVE INTENSITY
plt.figure()
left_edge_intensity = crystal_edge_locations.iloc[0]["intensity"]  # [mm]
right_edge_intensity = crystal_edge_locations.iloc[1]["intensity"] # [mm]
plt.plot(x_for_derivative,std_slices_derivative_intensity)
plt.plot([left_edge,right_edge],[left_edge_intensity,right_edge_intensity],"v")

plt.title(r"Crystal {}, run {} - Derivative squared of std: {}".format(crystal_name, run_number, r"$\left(\frac{d\sigma(x_{in})}{dx_{in}}\right)^2$"))
plt.xlabel(r'$x_{in}\ [mm]$')
plt.ylabel(r"$\left(\frac{d\sigma(x_{in})}{dx_{in}}\right)^2$")
# print(events)
plt.tight_layout()
plt.savefig("latex/img/std_derivative_squared.pdf")
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
# if os.path.isfile('crystal_analysis_parameters.csv'):
#     # read up the already existing parameters (like init_scan)
#     parameters_table = pd.read_csv("crystal_analysis_parameters.csv", sep="\t", index_col=0)
# else: #
#     raise FileNotFoundError("[ERROR]: File crystal_analysis_parameters.csv not "
#                             "found. Create it with save_as_hdf.py")
#
# parameters_table.loc['xmin'] = cut_left
# parameters_table.loc['xmax'] = cut_right
#
#
#
# parameters_table.to_csv("crystal_analysis_parameters.csv",sep='\t')
my.save_in_csv("crystal_analysis_parameters.csv",
                            xmin=cut_left,
                            xmax=cut_right,
                            ymin=cut_y_low,
                            ymax=cut_y_high)
#################
