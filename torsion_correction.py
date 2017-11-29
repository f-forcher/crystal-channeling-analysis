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



# MY LIBS
import editable_input as ei # My script for editable text input
from bin_dataframe import bin2D_dataframe
import mie_utils as my

######################################
################# FUNCTIONS
def gaussian(x, mu, sig, c):
    return c*matplotlib.mlab.normpdf(x, mu, sig)

def line(x, m, q):
    return m*x + q

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
        verbose=0,
        verbose_interval=10,
        random_state=random.SystemRandom().randrange(0,2147483647), # 2**31-1
        means_init=[[0], [50]],
    #        weights_init=[1 / 2, 1 / 2],
        init_params="kmeans",
        n_init = 2,
        tol=1e-6,
        precisions_init = [[[1/16]],[[1/16]]],
        #warm_start=True,
        max_iter=200)

    ################# GET THE DATA FROM THE DATAFRAME
    lowest_percentage = 5
    highest_percentage = 95
    first_percentile = np.percentile(input_groupby_obj, lowest_percentage)
    last_percentile = np.percentile(input_groupby_obj, highest_percentage)
    data_reduced = input_groupby_obj.values[(input_groupby_obj.values>=first_percentile) & (input_groupby_obj.values<=last_percentile)]
    data = data_reduced.reshape(-1, 1)

    #data = input_groupby_obj.reshape(-1, 1)

    ################# FIT THE DATA
    # Check that we have enough data for a fit, otherwise just return eff=0
    efficiency = np.NaN
    if data.size > 50:
        clf.fit(data)

        if not clf.converged_:
            print("[LOG]: Fit did not converge in this bin, bin ignored")
            efficiency = np.NaN


        r_m1, r_m2 = clf.means_
        w1, w2 = clf.weights_
        m1, m2 = r_m1[0], r_m2[0]
        r_c1, r_c2 = clf.covariances_
        c1, c2 = r_c1[0][0], r_c2[0][0]

        # print("Means: ", clf.means_, "\n")
        # print("Weights: ", clf.weights_, "\n")
        # print("Precisions: ",  1/c1, " ", 1/c2, "\n")
        # print("Covariances: ", c1, " ", c2, "\n")

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
    else:
        print("[LOG]: Too few data for this bin, bin ignored")
        efficiency = np.NaN
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
crystal_name = sys.argv[2]
run_number = sys.argv[3]



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

# Change angular units to microradians
events["Delta_Theta_x"] = 1e6*(events.loc[:,'Tracks_thetaOut_x'].values -
                               events.loc[:,'Tracks_thetaIn_x'].values)
events["Tracks_thetaIn_x"] = 1e6*events["Tracks_thetaIn_x"]
events['Tracks_thetaOut_x'] = 1e6*events['Tracks_thetaOut_x']
#################


################# BIN THE DATA
d0y_nbins  = 60
thetain_x_nbins = 60
gruppi = bin2D_dataframe(events, "Tracks_d0_y", "Tracks_thetaIn_x",
#                        (-2,2),(-30e-5,30e-5),17*4,12*4)
                        (-2,2), (-30,30), d0y_nbins, thetain_x_nbins)
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




# FIT THE TORSION
avg_Delta_Theta_x = [np.average(efficiencies.dropna().xs(xx,level=0).index.values, \
                    weights=efficiencies.dropna().xs(xx,level=0).values) for xx \
                    in efficiencies.dropna().xs(0.5,level=1).index.values]
avg_Delta_Theta_x_fit_noNaN = [curve_fit(gaussian,efficiencies.dropna().xs(xx,level=0).index.values,efficiencies.dropna().xs(xx,level=0).values,method="dogbox",loss="cauchy")[0][0] for xx \
                    in efficiencies.dropna().xs(0.5,level=1).index.values]
# avg_Delta_Theta_x_fit_NaNzero = [curve_fit(gaussian,efficiencies.fillna(0).xs(xx,level=0).index.values,efficiencies.fillna(0).xs(xx,level=0).values,method="dogbox",loss="cauchy")[0][0] for xx \
#                     in efficiencies.fillna(0).xs(0.5,level=1).index.values]
#plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x, "-", label="Avg")
#plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x_fit_noNaN, "-", label="Filtered fit")
# plt.plot(avg_Delta_Theta_x_fit_NaNzero, "-", label="fit NaN zero")
line_par, line_par_cov = curve_fit(line,efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x_fit_noNaN, method="dogbox", loss="cauchy")
p, pc = curve_fit(line,efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x)


#
# # Plot as 2D array
plt.figure()
grid_for_histo=np.array([list(v) for v in efficiencies.index.values])
plt.hist2d(grid_for_histo[:,0],grid_for_histo[:,1], weights=efficiencies.values,
           bins=[d0y_nbins, thetain_x_nbins]) # TODO
plt.title(r"Crystal {}, run {} - Efficiency as function of {}".format(crystal_name, run_number, r"$x_{in}$ and $\Delta \theta_{x}$"))
#plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x, "-", label="Avg")
#plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x_fit_noNaN, "-", label="Filtered fit")
plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),line(efficiencies.xs(0.5,level=1).index.get_values(), *line_par), label="Torsion linear fit", color = 'r')
plt.xlabel(r'$y_{in}\ [mm]$')
plt.ylabel(r'$\theta_{x_{in}}\ [\mu rad]$')
# print(events)
plt.colorbar()
plt.tight_layout()
plt.savefig("latex/img/efficiency_histo.pdf")
plt.show()




################# SAVE FIT PLOT TO FILE
plt.figure()
plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x, "-", label="Avg")
plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),avg_Delta_Theta_x_fit_noNaN, "-", label="Filtered fit")
plt.plot(efficiencies.xs(0.5,level=1).index.get_values(),line(efficiencies.xs(0.5,level=1).index.get_values(), *line_par), label="Torsion linear fit", color = 'r')
plt.title(r"Crystal {}, run {} — Torsion fit: {}".format(crystal_name, run_number, r"$y_{in}$ vs $\Delta \theta_{x}$"))
plt.xlabel(r'$y_{in}\ [mm]$')
plt.ylabel(r'$\theta_{x_{in}}\ [\mu rad]$')
plt.tight_layout()
plt.legend()
plt.savefig("latex/img/torsion_fit.pdf")
plt.show()
#################

line_par_err = np.sqrt(np.diag(line_par_cov))
pe = np.sqrt(np.diag(pc))

print("m: {:.5} +- {:.5}\nq: {:.5} +- {:.5}".format(line_par[0], line_par_err[0],
                                                  line_par[1], line_par_err[1]))

################# SAVE PARAMETERS TO FILE
tor_m = line_par[0]
tor_q = line_par[1]
my.save_parameters_in_csv("crystal_analysis_parameters.csv",
                            torsion_m=line_par[0],
                            torsion_m_err=line_par_err[0],
                            torsion_q=line_par[1],
                            torsion_q_err=line_par_err[1],)
#################


################# CORRECT FOR TORSION AND SHOW THE PLOT
events["Tracks_thetaIn_x"] = (events["Tracks_thetaIn_x"] -
                              (tor_m*events["Tracks_d0_y"]+tor_q))# + init_scan
plt.figure()
plt.hist2d(events.loc[:,'Tracks_thetaIn_x'].values ,events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,\
bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
plt.title(r"Crystal {}, run {} — Histogram: {}".format(crystal_name, run_number, r"$y_{in}$ vs $\Delta \theta_{x}$"))
plt.xlabel(r'$\theta_{x_{in}}\ [\mu rad]$')
plt.ylabel(r'$\Delta \theta_{x}\ [\mu rad]$')
# print(events)
plt.colorbar()
plt.tight_layout()
plt.savefig("latex/img/corrected_histo.pdf")
plt.show()
#################


################# SAVE TO HDF FILE THE CORRECTED DATA
events.to_hdf("torsion_corrected_"+file_name+".hdf","simpleEvent",
                format="table",
                fletcher32=True, mode="a", complevel=9, append=False,
                data_columns=['Tracks_thetaIn_x', 'Tracks_thetaOut_x'])
#################
