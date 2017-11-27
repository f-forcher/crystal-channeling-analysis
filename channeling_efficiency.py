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

# MY LIBS
# import editable_input as ei # My script for editable text input
# from bin_dataframe import bin2D_dataframe
import mie_utils as my

def fit_channeling(input_df):
    """
    Fit the histogram for two gaussian peaks (CH for channeling and AM for
    amorphous), then return the fit ibject for further processing.

    data: input dataset.

    return: dict cointaining the parameters:
            "weight_AM"
            "weight_CH"
            "mean_AM"
            "mean_CH"
            "sigma_AM"
            "sigma_CH"

    """
    clf = mixture.GaussianMixture(
        n_components=2,
        covariance_type='full',
        verbose=0,
        verbose_interval=10,
        random_state=random.SystemRandom().randrange(0,2147483647), # 2**31-1
#        means_init=[[-15], [50]],
#        weights_init=[0.4, 0.6],
        init_params="kmeans",
        n_init = 2,
        tol=1e-6,
#         precisions_init = [[[1/16]],[[1/16]]],
        #warm_start=True,
        max_iter=500)

    ################# GET THE DATA FROM THE DATAFRAME
    lowest_percentage = 0.3
    highest_percentage = 99.7
    first_percentile = np.percentile(input_df, lowest_percentage)
    last_percentile = np.percentile(input_df, highest_percentage)
    data_reduced = input_df.values[(input_df.values >= \
              first_percentile) & (input_df.values <= last_percentile)]
    data = data_reduced.reshape(-1, 1)

    ################# FIT THE DATA
    # Check that we have enough data for a fit, otherwise just return eff=0
    clf.fit(data)

    if not clf.converged_:
        print("[LOG]: Fit did not converge in this bin, bin ignored")
        efficiency = np.NaN


    r_m1, r_m2 = clf.means_
    w1, w2 = clf.weights_
    m1, m2 = r_m1[0], r_m2[0]
    r_c1, r_c2 = clf.covariances_
    # r_c1 = clf.covariances_
    # r_c2 = clf.covariances_
    c1, c2 = np.sqrt(r_c1[0][0]), np.sqrt(r_c2[0][0])

    # print("Means: ", clf.means_, "\n")
    # print("Weights: ", clf.weights_, "\n")
    # print("Precisions: ",  1/c1, " ", 1/c2, "\n")
    # print("Covariances: ", c1, " ", c2, "\n")

    fit_results = {}
    # Save the weights in the right array
    # Lower delta_thetax is the AM peak, higher CH
    fit_results["nevents"] = len(data)
    if (m1 < m2):
        fit_results["weight_AM"] = w1
        fit_results["weight_CH"] = w2
        fit_results["mean_AM"] = m1
        fit_results["mean_CH"] = m2
        fit_results["sigma_AM"] = c1
        fit_results["sigma_CH"] = c2
    else:
        fit_results["weight_AM"] = w2
        fit_results["weight_CH" ]= w1
        fit_results["mean_AM"] = m2
        fit_results["mean_CH"] = m1
        fit_results["sigma_AM"] = c2
        fit_results["sigma_CH"] = c1

    return fit_results,data


######################################
################# MAIN
file_name = sys.argv[1]
crystal_name = sys.argv[2]
particle_energy_input = sys.argv[3] # [GeV]
run_number = sys.argv[4]

# Probably read by chunk not needed by now.
events = pd.read_hdf(file_name)

# Angles in microradians from torsion_correction.py lines 171-174
events["Delta_Theta_x"] = events.loc[:,'Tracks_thetaOut_x'].values - \
                               events.loc[:,'Tracks_thetaIn_x'].values

# # Read crystal parameters
cpars = pd.read_csv("crystal_physical_characteristics.csv", index_col=0)
crystal_params = cpars[~cpars.index.isnull()] # Remove empty (like ,,,,,,) lines

crystal_lenght = float(crystal_params.loc[crystal_name,"Lenght (z) (mm)"])*1e-3 # [m]

# Taken from my thesis code
# Initial guesses for crystal parameter, uses either data from
# crystal_physical_characteristics.csv or a default value if the latter is not
# found.
particle_energy = float(particle_energy_input)*1e9 # [eV] TODO generalize to pions!
critical_radius = particle_energy / 550e9 # [m] 550 GeV/m electric field strength from byriukov
pot_well = 21.34 # [eV] Potential well between crystal planes
theta_bending = float(crystal_params.loc[crystal_name,"H8 bending angle (urad)"]) * 1e-6 # [rad]
crystal_curvature_radius = crystal_lenght / theta_bending
theta_c = math.sqrt(2*pot_well/particle_energy) * (1 - critical_radius/crystal_curvature_radius)*1e6 # [murad]
# c1_thetavr, c2_thetavr = (-1.5, 1.66666)
# theta_vr =  c1_thetavr * theta_c * (1 - c2_thetavr*critical_radius/crystal_curvature_radius) # [murad]

x_histo = range(-100,100,1) # [murad]

################# FIT USING 5 and 10 AS CUTS
# ang_cut_low = [-5,-10] # [murad]
# ang_cut_high = [5,10] # [murad]
# for low_cut, high_cut in zip(ang_cut_low,ang_cut_high):
#     plt.figure()
#     geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & \
#                                   (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
#     # plt.hist2d(geocut_df.loc[:,'Tracks_thetaIn_x'].values, \
#     #  geocut_df.loc[:,'Tracks_thetaOut_x'].values - geocut_df.loc[:,'Tracks_thetaIn_x'].values,\
#     #  bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
#     fit_results = fit_channeling(geocut_df.Delta_Theta_x)[0]
#     filtered_data = fit_channeling(geocut_df.Delta_Theta_x)[1]
#     plt.hist(filtered_data, bins=200, range=[-100,100], normed=False) # [murad]
#
#
#     total_number_of_events = fit_results["nevents"]
#     gauss_AM = total_number_of_events * fit_results["weight_AM"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_AM"], fit_results["sigma_AM"])
#     gauss_CH = total_number_of_events * fit_results["weight_CH"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_CH"], fit_results["sigma_CH"])
#
#     plt.plot(x_histo, gauss_AM, label="Amorphous Peak", color='r')
#     plt.plot(x_histo, gauss_CH, label="Channeling Peak", color='Orange')
#     plt.suptitle(r"Crystal {}, run {} — Channeling, cut ±{:.3} ".format(crystal_name, run_number, float(high_cut)),fontweight='bold')
#     plt.title(r"Efficiency {:.3}% — bending angle {:.3} ".format(fit_results["weight_CH"]*100, fit_results["mean_CH"]) + r"$[\mu rad]$")
#     plt.xlabel(r'$\Delta \theta_{x}\ [\mu rad]$')
#     plt.ylabel('Frequency')
#     plt.legend()
#     #plt.tight_layout()
#     plt.savefig("latex/img/" + str(high_cut) + "_chan_histo.pdf")
#     plt.show()
#
#
#     print("\nCut: +-",low_cut)
#     print(pd.Series(fit_results))
#################


################# FIT USING CRITICAL ANGLE AS CUT
# theta_bending = fit_results["mean_CH"]
# crystal_curvature_radius = crystal_lenght / (theta_bending*1e-6)
# theta_c = math.sqrt(2*pot_well/particle_energy)*1e6 * (1 - critical_radius/crystal_curvature_radius) # [murad]


ang_cut_low = [-theta_c / 2, -theta_c]
ang_cut_high = [theta_c / 2, theta_c]

print("New Thetac: ", theta_c)

i = 0
for low_cut, high_cut in zip(ang_cut_low,ang_cut_high):
    plt.figure()
    geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & \
                                  (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
    # plt.hist2d(geocut_df.loc[:,'Tracks_thetaIn_x'].values, \
    #  geocut_df.loc[:,'Tracks_thetaOut_x'].values - geocut_df.loc[:,'Tracks_thetaIn_x'].values,\
    #  bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
    fit_results = fit_channeling(geocut_df.Delta_Theta_x)[0]
    filtered_data = fit_channeling(geocut_df.Delta_Theta_x)[1]
    plt.hist(filtered_data, bins=200, range=[-100,100], normed=False) # [murad]


    total_number_of_events = fit_results["nevents"]
    gauss_AM = total_number_of_events * fit_results["weight_AM"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_AM"], fit_results["sigma_AM"])
    gauss_CH = total_number_of_events * fit_results["weight_CH"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_CH"], fit_results["sigma_CH"])

    plt.plot(x_histo, gauss_AM, label="Amorphous Peak", color='r')
    plt.plot(x_histo, gauss_CH, label="Channeling Peak", color='Orange')
    thetac_title = r"$\theta_c/2$" if i == 0 else r"$\theta_c$"
    cut_value = theta_c/2 if i == 0 else theta_c
    plt.suptitle(r"Crystal {}, run {} — Channeling, cut ± {} = ±{:.3}".format(crystal_name, run_number,thetac_title, cut_value),fontweight='bold')
    plt.title(r"Efficiency {:.3}% — Bending Angle {:.3} {}".format(fit_results["weight_CH"]*100, fit_results["mean_CH"],r"$[\mu rad]$"))
    plt.xlabel(r'$\Delta \theta_{x}\ [\mu rad]$')
    plt.ylabel('Frequency')
    plt.legend()
    #plt.tight_layout()


    thetac_filename = 'half_thetac' if i == 0 else 'thetac'
    plt.savefig("latex/img/"+ thetac_filename + "_chan_histo.pdf")
    plt.show()


    print("\nCut: +-",low_cut)
    print(pd.Series(fit_results))
    i=i+1
#################
