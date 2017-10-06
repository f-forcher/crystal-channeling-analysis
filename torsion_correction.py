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
from bin_dataframe import bin2D_dataframe

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


################# BIN THE DATA
gruppi = bin2D_dataframe(events, "Tracks_d0_y", "Tracks_thetaIn_x",
                        (-2,2),(-30e-5,30e-5),17*2,12*2)
for i,v in gruppi:
    print(i)
#################
