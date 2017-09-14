# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
#
# XB = np.linspace(-1,1,20)
# YB = np.linspace(-1,1,20)
# X,Y = np.meshgrid(XB,YB)
# Z = np.exp(-(X**2+Y**2))
# plt.imshow(Z,interpolation='none')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from root_pandas import read_root
import sys
from itertools import islice

# # normal distribution center at x=0 and y=5
# x = np.random.randn(100000)
# y = np.random.randn(100000) + 5
#
# plt.hist2d(x, y, bins=4, norm=LogNorm())
# plt.colorbar()
# plt.show()

file_name = sys.argv[1]



# DATAFRAME COLUMNS
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
# evts = read_root(file_name, chunksize=1000000)
# print("Loaded dataframe from root")

chunksize = 2000000
evts = pd.read_hdf(file_name, chunksize = chunksize, columns=["Tracks_d0_x", "Tracks_thetaOut_x", "Tracks_thetaIn_x"])


# events = evts.next() # python 2!
# events = next(islice(evts, 1))
chunks = 0
for df in evts:
    chunks = chunks + chunksize
    print("chunks ", chunks)
    df.info()
    events = df # TODO Scoprire perche' getEntries e .info ritornano dei numeri diversi
    break;
# events = next(evts)

plt.hist2d(events.loc[:,'Tracks_d0_x'].values ,events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,\
bins=400, norm=LogNorm(), range=[[-5,5],[-100e-6,100e-6]])

# print(events)
plt.colorbar()
plt.show()
