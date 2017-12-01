# SYSTEM LIBS
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt; plt.ion();
import matplotlib
import numpy as np
import pandas as pd
# from root_pandas import read_root
import sys
from itertools import islice
from scipy import signal # To find peaks, for edge-detecting the crystal
from scipy.optimize import curve_fit
import os
from sklearn import mixture
import random
import math

ndata = 3000
s = pd.Series(np.random.randn(ndata))
nsamples = 2000
# drop=True does not save the old index in a column
boot_samples = (s.sample(ndata,replace=True).reset_index(drop=True) for i in range(nsamples))
df = pd.DataFrame(boot_samples)

#
plt.title("orig")
plt.hist(s,bins='auto')
plt.figure()
plt.title("std")
plt.hist(df.std(axis=1,ddof=1),bins=100,range=[0.5,1.5])
# plt.figure()
# plt.title("std WITH ddof=1")
# plt.hist(df.std(axis=1,ddof=1),bins=100,range=[0.5,1.5])
# plt.figure()
# plt.title("std WITH ddof=0")
# plt.hist(df.std(axis=1,ddof=0),bins=100,range=[0.5,1.5])
plt.figure()
plt.title("mean")
plt.hist(df.mean(axis=1),bins=100,range=[-0.8,0.8])

print("Mean of the mean ", df.mean(axis=1).mean()," +- ", df.mean(axis=1).std())
print("Theoretical error of the mean: ", s.std()/np.sqrt(ndata), ", delta = ", df.mean(axis=1).mean()-s.std()/np.sqrt(ndata))
