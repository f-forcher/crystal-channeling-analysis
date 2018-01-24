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

# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def gaussian(x, mu, sig, c):
    return c*matplotlib.mlab.normpdf(x, mu, sig)


def gaussian_sum(x, c1, mu1, sig1, mu2, sig2):
    return c1*matplotlib.mlab.normpdf(x, mu1, sig1) + \
           (1-c1)*matplotlib.mlab.normpdf(x, mu2, sig2)


def fit_channeling(input_df,lowest_percentage, highest_percentage,
                   fit_tolerance,max_iterations):
    """
    Fit the histogram for two gaussian peaks (CH for channeling and AM for
    amorphous), then return the fit object for further processing.

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
#        means_init=[[-5], [77]],
#        weights_init=[0.5, 0.5],
        init_params="kmeans",
        n_init = 5,
        tol=fit_tolerance,
#        precisions_init = [[[1/10**2]],[[1/10**2]]],
        #warm_start=True,
        max_iter=max_iterations)

    ################# GET THE DATA FROM THE DATAFRAME
    lowest_percentage = 3
    highest_percentage = 97
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
    #r_c1 = clf.covariances_
    #r_c2 = clf.covariances_
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
        fit_results["weight_CH"]= w1
        fit_results["mean_AM"] = m2
        fit_results["mean_CH"] = m1
        fit_results["sigma_AM"] = c2
        fit_results["sigma_CH"] = c1

    # Calculate errors plugging the found parameters in a chi2 fit.
    data_histo = np.histogram(data,bins=200,normed=True)
    histo_bin_centers = (data_histo[1] + (data_histo[1][1] - data_histo[1][0])/2)[:-1]
    initial_guess = [fit_results["weight_AM"], fit_results["mean_AM"], fit_results["sigma_AM"],
        fit_results["mean_CH"], fit_results["sigma_CH"]]
    popt, pcov = curve_fit(gaussian_sum, histo_bin_centers, data_histo[0],
                            p0=initial_guess)

    print(popt)

    # # Plot the chi2 fit, for debug purposes
    # plt.figure()
    # plt.plot(histo_bin_centers,data_histo[0],".")
    # plt.plot(histo_bin_centers,gaussian_sum(histo_bin_centers,*popt))
    # plt.plot(histo_bin_centers,gaussian_sum(histo_bin_centers,*initial_guess))
    #
    # plt.show()

    perr = np.sqrt(np.diag(pcov))
    # Should be in same order as in p0 of curve_fit
    fit_results["weight_AM_err"] = perr[0]
    fit_results["weight_CH_err"] = perr[0] # c2=1-c1, by propagation same error
    fit_results["mean_AM_err"]   = perr[1]
    fit_results["sigma_AM_err"]  = perr[2]
    fit_results["mean_CH_err"]   = perr[3]
    fit_results["sigma_CH_err"]  = perr[4]


    return fit_results,data


######################################
################# MAIN
file_name = sys.argv[1]
crystal_name = sys.argv[2]
run_number = sys.argv[3]
particle_name = sys.argv[4]
particle_energy_input = sys.argv[5] # [GeV]
run_date = sys.argv[6] # [GeV]


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


# Read by chunk not needed by now probably.
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

#### How much to move the absolute position of the cuts
# Example, with cuts [-5,5] and offset +3, we have an actual cut of [-2,8]
# Useful if torsion correction is not employed, to center the cuts
center_offset = float(my.get_from_csv(analysis_configuration_params_file,
                                             "chan_center_offset"
                                             ))

ang_cut_low = [center_offset - theta_c / 2, center_offset - theta_c]
ang_cut_high = [center_offset + theta_c / 2, center_offset + theta_c]

dtx_low, dtx_high = my.get_from_csv(analysis_configuration_params_file,
                                             "chan_hist_range_dtx_low",
                                             "chan_hist_range_dtx_high",
                                             )
dtx_nbins = int(my.get_from_csv(analysis_configuration_params_file,
                                             "chan_hist_tx_nbins"))
x_histo = np.linspace(dtx_low,dtx_high,dtx_nbins + 1) # [murad]



print("New Thetac: ", theta_c)

lowest_percentage, highest_percentage = my.get_from_csv(analysis_configuration_params_file,
                                     "chan_low_percentage",
                                     "chan_high_percentage")
chan_fit_tolerance = my.get_from_csv(analysis_configuration_params_file,
                                 "chan_fit_tolerance")
max_iterations = int(my.get_from_csv(analysis_configuration_params_file,
                                 "chan_max_iterations"))

i = 0
plt.figure()
plt.hist2d(events.loc[:,'Tracks_thetaIn_x'].values, \
 events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,\
 bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
#plt.figure(); plt.hist2d(events.loc[:,'Tracks_thetaIn_x'].values, events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values, bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])

for low_cut, high_cut in zip(ang_cut_low,ang_cut_high):
    plt.figure()
    geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & \
                                  (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
    # plt.hist2d(geocut_df.loc[:,'Tracks_thetaIn_x'].values, \
    #  geocut_df.loc[:,'Tracks_thetaOut_x'].values - geocut_df.loc[:,'Tracks_thetaIn_x'].values,\
    #  bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
    fit_and_data = fit_channeling(geocut_df.Delta_Theta_x,
                                  lowest_percentage, highest_percentage,
                                  chan_fit_tolerance, max_iterations)
    fit_results = fit_and_data[0]
    filtered_data = fit_and_data[1]
    #plt.yscale('log', nonposy='clip')
    plt.hist(geocut_df.Delta_Theta_x, bins=dtx_nbins, range=[dtx_low,dtx_high], normed=False) # [murad]
    #plt.hist(filtered_data, bins=dtx_nbins, range=[dtx_low,dtx_high], normed=False) # [murad]


    total_number_of_events = fit_results["nevents"]
    gauss_AM = total_number_of_events * fit_results["weight_AM"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_AM"], fit_results["sigma_AM"])
    gauss_CH = total_number_of_events * fit_results["weight_CH"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_CH"], fit_results["sigma_CH"])

    plt.plot(x_histo, gauss_AM, label="Amorphous Peak", color='r')
    plt.plot(x_histo, gauss_CH, label="Channeling Peak", color='Orange')
    thetac_title = r"$\theta_c/2$" if i == 0 else r"$\theta_c$"
    cut_value = theta_c/2 if i == 0 else theta_c
    plt.suptitle(r"{} run {}, {} {} GeV — Channeling, cut ± {} = ±{:.3}".format(crystal_name,run_number,particle_name,particle_energy_input,thetac_title,cut_value),fontweight='bold')
    plt.title(r"Efficiency {:.3}% ± {:.1f}% — Bending Angle {:.4} ± {:.1f} {}".format(fit_results["weight_CH"]*100, fit_results["weight_CH_err"]*100,
                                                                                fit_results["mean_CH"],fit_results["mean_CH_err"],r"$[\mu rad]$"))
    plt.xlabel(r'$\Delta \theta_{x}\ [\mu rad]$')
    plt.ylabel('Frequency')
    plt.legend()
    #plt.tight_layout()


    thetac_filename = 'half_thetac' if i == 0 else 'thetac'
    plt.savefig("latex/img/"+ thetac_filename + "_chan_histo.pdf")
    plt.show()


    print("\nCut low: ", low_cut)
    print("\nCut high: ", high_cut)
    print(pd.Series(fit_results))

    # my.save_parameters_in_csv("crystal_analysis_parameters.csv",**fit_results)


    i=i+1
#################



################# WRITE TO LATEX THE PARAMS
cut_x_left, cut_x_right = my.get_from_csv("crystal_analysis_parameters.csv", "xmin", "xmax")

tor_m,tor_q,tor_m_err,tor_q_err = my.get_from_csv("crystal_analysis_parameters.csv",\
                                  "torsion_m", "torsion_q", "torsion_m_err", "torsion_q_err")

cut_y_low, cut_y_high = my.get_from_csv(analysis_configuration_params_file, "cut_y_low", "cut_y_high")

#Example \newcommand{\myname}{Francesco Forcher}

#with open("latex/text_gen-definitions.tex", "a") as myfile:
file_name = sys.argv[1]
crystal_name = sys.argv[2]
run_number = sys.argv[3]
particle_name = sys.argv[4]
particle_energy_input = sys.argv[5] # [GeV]
run_date = sys.argv[6] # [GeV]

with open("latex/test_gen-definitions.tex","w") as myfile:
    myfile.write(r"% FILE GENERATED AUTOMATICALLY")
    myfile.write("\n\n")
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\myname","Francesco Forcher"))
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\crystalname",crystal_name))
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\runnumber", run_number))
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\rundate", run_date))
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\particletype", particle_name))
    myfile.write("\\newcommand{{{}}}{{{}}}\n".format("\\particleenergy", particle_energy_input + " GeV"))

    myfile.write("\n")

    myfile.write("\\newcommand{{{}}}{{{:.3f}}}\n".format("\\xmin",float(cut_x_left)))
    myfile.write("\\newcommand{{{}}}{{{:.3f}}}\n".format("\\xmax",float(cut_x_right)))
    myfile.write("\\newcommand{{{}}}{{{:.3f}}}\n".format("\\ymin",float(cut_y_low)))
    myfile.write("\\newcommand{{{}}}{{{:.3f}}}\n".format("\\ymax",float(cut_y_high)))

    myfile.write("\n")

    myfile.write("\\newcommand{{{}}}{{{:.1f}}}\n".format("\\torsionm",   float(tor_m)))
    myfile.write("\\newcommand{{{}}}{{{:.1f}}}\n".format("\\torsionq",   float(tor_q)))
    myfile.write("\\newcommand{{{}}}{{{:.1f}}}\n".format("\\torsionmerr",float(tor_m_err)))
    myfile.write("\\newcommand{{{}}}{{{:.1f}}}\n".format("\\torsionqerr",float(tor_q_err)))



#################
