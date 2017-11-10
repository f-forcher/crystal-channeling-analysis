# For Python 2 cause pyROOT
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
from sklearn import mixture
import random
import ROOT as rt

# MY LIBS
import editable_input as ei # My script for editable text input
from bin_dataframe import bin2D_dataframe


######################################
################# FUNCTIONS

def fit_and_get_efficiency(input_groupby_obj):
    """
    Function to be applied on a groupby to get channeling efficiency.

    data: input dataset.

    return: channeling efficiency (0 < efficiency < 1), basically
            channeling peak weight.
    """
    clf = mixture.GaussianMixture(
        n_components=2,
        covariance_type='full',
        verbose=2,
        verbose_interval=10,
        random_state=random.SystemRandom().randrange(0, 4095),
        means_init=[[0], [50]],
#        weights_init=[1 / 2, 1 / 2],
        init_params="kmeans",
        n_init = 2,
        tol=1e-6,
        #warm_start=True,
        max_iter=200)

    ################# GET THE DATA FROM THE DATAFRAME
    data = input_groupby_obj.reshape(-1, 1)

    ################# FIT THE DATA
    # Check that we have enough data for a fit, otherwise just return eff=0
    efficiency = 0
    if data.size > 10:
        clf.fit(data)

        r_m1, r_m2 = clf.means_
        w1, w2 = clf.weights_
        m1, m2 = r_m1[0], r_m2[0]

        # Save the weights in the right array
        # Lower delta_thetax is the AM peak, higher CH
        if (m1 < m2):
            weights_AM = w1
            weights_CH = w2
            means_AM = m1
            means_CH = m2
        else:
            weights_AM = w2
            weights_CH = w1
            means_AM = m2
            means_CH = m1
        efficiency = weights_CH
    return efficiency

def fit_and_get_efficiency_pyROOT(input_groupby_obj):
    """
    Function to be applied on a groupby to get channeling efficiency.

    data: input dataset.

    return: channeling efficiency (0 < efficiency < 1), basically
            channeling peak weight.
    """

    rt.

    ################# GET THE DATA FROM THE DATAFRAME
    data = input_groupby_obj.reshape(-1, 1)

    ################# FIT THE DATA
    # Check that we have enough data for a fit, otherwise just return eff=0
    efficiency = 0
    if data.size > 10:
        clf.fit(data)

        r_m1, r_m2 = clf.means_
        w1, w2 = clf.weights_
        m1, m2 = r_m1[0], r_m2[0]

        # Save the weights in the right array
        # Lower delta_thetax is the AM peak, higher CH
        if (m1 < m2):
            weights_AM = w1
            weights_CH = w2
            means_AM = m1
            means_CH = m2
        else:
            weights_AM = w2
            weights_CH = w1
            means_AM = m2
            means_CH = m1
        efficiency = weights_CH
    return efficiency





######################################
################# MAIN

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
# init_scan = float(parameters_table.loc['init_scan']) # Not needed here
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
# data_columns=[...] when .hdf is created, to be able to use "where" on them
cuts_and_selections = ["SingleTrack == 1", "Tracks_d0_x > cut_left", "Tracks_d0_x < cut_right"]

evts = pd.read_hdf(file_name, chunksize = chunksize, columns=interesting_columns, where=cuts_and_selections)

loaded_rows = 0
events = pd.DataFrame(columns=interesting_columns)
print "[LOG]: Loading data..."
for df in evts:
    loaded_rows = loaded_rows + df.shape[0]
    print("\n[LOG] loaded ", loaded_rows, " rows\n")
    df.info()
    events = events.append(df,ignore_index=True) # join inner maybe unnecessary here
    # break; # Uncomment to get only the first chunk
print "\n[LOG]: Loaded data!\n"
events.info()

events["Delta_Theta_x"] = 1e6*events.loc[:,'Tracks_thetaOut_x'].values - 1e6*events.loc[:,'Tracks_thetaIn_x'].values
events["Tracks_thetaIn_x"] = 1e6*events["Tracks_thetaIn_x"]
#################


################# BIN THE DATA
gruppi = bin2D_dataframe(events, "Tracks_d0_y", "Tracks_thetaIn_x",
#                        (-2,2),(-30e-5,30e-5),17*4,12*4)
                        (-2,2),(-30,30),60,60)
# efficiency_histo = {}
# clf = mixture.GaussianMixture(
#     n_components=2,
#     covariance_type='full',
#     verbose=2,
#     verbose_interval=10,
#     random_state=random.SystemRandom().randrange(0, 4095),
# #    means_init=[[-17], [0]],
#     #                              weights_init=[1 / 2, 1 / 2],
#     init_params="kmeans",
#     tol=1e-6,
#     max_iter=1000)
# for index1, index2, gruppo in gruppi:
#     print("\n",index)
#     # This reshape transforms (an np.array) [1,2,3] into [ [1], [2], [3] ]
#     # Scikit wants a list of datapoints, here the datapoints coordinate are 1D, hence each one is a single-element list (of features/coordinates)
#     data = (gruppo.loc[:,"Tracks_thetaOut_x"].values - \
#             gruppo.loc[:,"Tracks_thetaIn_x"].values).reshape(-1, 1)
#     print(data[:10])
#     print("Tot size: ", data.size)
#
#     efficiency = 0
#     if data.size < 10:
#         continue
#     else:
#         clf.fit(data)
#
#         r_m1, r_m2 = clf.means_
#         w1, w2 = clf.weights_
#         m1, m2 = r_m1[0], r_m2[0]
#
#         # Save the weights in the right array
#         # Lower delta_thetax is the AM peak, higher CH
#         if (m1 < m2):
#             weights_AM = w1
#             weights_CH = w2
#             means_AM = m1
#             means_CH = m2
#         else:
#             weights_AM = w2
#             weights_CH = w1
#             means_AM = m2
#             means_CH = m1
#         efficiency = weights_CH
#
#     efficiency_histo[index] = efficiency
#     print("efficiency: ", efficiency)

# data = (gruppo.loc[:,"Tracks_thetaOut_x"].values - \
#         gruppo.loc[:,"Tracks_thetaIn_x"].values).reshape(-1, 1)
efficiencies = gruppi["Delta_Theta_x"].aggregate(fit_and_get_efficiency)

# grid_for_histo=np.array([list(v) for v in efficiencies.index.values])
# plt.hist2d(grid_for_histo[:,0],grid_for_histo[:,1],weights=efficiencies.values)
# plt.show()
eunst = efficiencies.unstack(fill_value=0.0)
eff_arr = np.transpose([list(eunst.iloc[i]) for i in range(eunst.index.size)])
plt.imshow(eff_arr)
plt.show()

#################
