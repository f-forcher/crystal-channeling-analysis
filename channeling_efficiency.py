# SYSTEM LIBS
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt; plt.ion();
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
# import editable_input as ei # My script for editable text input
# from bin_dataframe import bin2D_dataframe
import mie_utils as my

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
crystal_name = "STF110"
file_name = sys.argv[1]

# Probably read by chunk not needed by now.
events = pd.read_hdf(file_name)

# Angles in microradians from torsion_correction.py lines 171-174
events["Delta_Theta_x"] = events.loc[:,'Tracks_thetaOut_x'].values - \
                               events.loc[:,'Tracks_thetaIn_x'].values

# # Read crystal parameters
# cpars = pd.read_csv("crystal_physical_characteristics.csv", index_col=0)
# crystal_params = cpars[~cpars.index.isnull()] # Remove empty (like ,,,,,,) lines
#
# crystal_lenght = float(crystal_params.loc[crystal_name,"Lenght (z) (mm)"]) # [mm]
#
# # Taken from my thesis code
# # Initial guesses for crystal parameter, uses either data from
# # crystal_physical_characteristics.csv or a default value if the latter is not
# # found.
# particle_energy = crystal_params[crystal_name]['particle_energy[GeV]'] * 1e9 # [eV] TODO generalize to pions!
# critical_radius = 1 # [m] TODO at 400 GeV
# pot_well = 21.34 # [eV] Potential well between crystal planes
# theta_bending = float(crystal_params.loc[crystal_name,"H8 bending angle (urad)"]) # [murad]
# crystal_curvature_radius = crystal_lenght / theta_bending
# theta_c = math.sqrt(2*pot_well/particle_energy) * (1 - critical_radius/crystal_curvature_radius)*1e6 # [murad]
# c1_thetavr, c2_thetavr = (-1.5, 1.66666)
# theta_vr =  c1_thetavr * theta_c * (1 - c2_thetavr*critical_radius/crystal_curvature_radius) # [murad]

ang_cut_low = [-2,-5,-10,-70]
ang_cut_high = [2,5,10,70]

for low_cut, high_cut in zip(ang_cut_low,ang_cut_high):
    plt.figure()
    geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
    # plt.hist2d(geocut_df.loc[:,'Tracks_thetaIn_x'].values ,geocut_df.loc[:,'Tracks_thetaOut_x'].values - geocut_df.loc[:,'Tracks_thetaIn_x'].values,\
    # bins=[400,200], norm=LogNorm(),
    #  range=[[-100,100], [-80,120]])
    plt.hist(geocut_df.Delta_Theta_x, bins=200, range=[-100,100])
    eff = fit_and_get_efficiency(geocut_df.Delta_Theta_x)
    print(eff)
    plt.show()
