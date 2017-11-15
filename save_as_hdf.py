# from matplotlib.colors import LogNorm
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from root_pandas import read_root
import sys
# from itertools import islice

file_name = sys.argv[1]

chunksize = 2000000
evts = read_root(file_name, chunksize=chunksize) # iterator iterating the chunks
print "Opened root file "
print file_name

# CRYSTAL DATAFRAME COLUMNS
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
chunks = 0
init_scan = -1
for df in evts:
    df.to_hdf(file_name+".hdf","simpleEvent", format="table", \
              fletcher32=True, mode="a", complevel=9,append=True, \
              data_columns=['SingleTrack','Tracks_thetaIn_x', 'Tracks_d0_x', 'Tracks_d0_y'])
    chunks = chunks + chunksize
    print("Written " + str(chunks) + " rows")
    init_scan_column = df.loc[:,"GonioPos_x"] # TODO bug misterioso
    init_scan = init_scan_column.mean()


# TODO Commentare qua sta roba convoluta
parameters_table = pd.DataFrame({"parameter_name": ["init_scan"], "value": [init_scan]})
parameters_table.set_index("parameter_name",inplace=True)

parameters_table.to_csv("crystal_analysis_parameters.csv",sep='\t')

print "Finished creating HDF file"
