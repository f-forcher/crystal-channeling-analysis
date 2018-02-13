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
import scipy
import os
from sklearn import mixture
import random
import math

from mpl_toolkits.mplot3d import Axes3D

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


# def histobinwidth_cost_func(x, y, bw_x, bw_y):
#     """
#     From http://www.neuralengine.org/res/histogram.html
#
#     Target cost function of the binwidth bw, minimize to get best binwidth from
#     the shimazaki-shimonoto rule.
#
#     x,y: ordered data arrays, the same passed to np.histogram2d
#     bw_x,bw_y: Bin width parameters along the two axes to optimize
#     """
#     bin_x_num =
#
#
#     hist, x1, y1 = np.histogram2d(events.loc[:,'Tracks_thetaIn_x'].values,
#     events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,
#     bins=[bw_x,bw_y], range=[[-100,100], [-80,120]])

def best_binwidth(x, y):
    x_max = 100
    x_min = -100

    y_max = 120
    y_min = -80


    Nx_MIN = 10  #Minimum number of bins in x (integer)
    Nx_MAX = 200  #Maximum number of bins in x (integer)

    Ny_MIN = 10 #Minimum number of bins in y (integer)
    Ny_MAX = 200  #Maximum number of bins in y (integer)


    Nx = np.arange(Nx_MIN, Nx_MAX,5) # #of Bins
    Ny = np.arange(Ny_MIN, Ny_MAX,5) # #of Bins

    Dx = (x_max - x_min) / Nx    #Bin size vector
    Dy = (y_max - y_min) / Ny    #Bin size vector

    Dxy=[]
    for i in Dx:    #Bin size vector
        a=[]
        for j in Dy:    #Bin size vector
            a.append((i,j))
        Dxy.append(a)
    Dxy=np.array( Dxy, dtype=[('x', float),('y', float)]) #matrix of bin size vector


    Cxy=np.zeros(np.shape(Dxy))

    Cxy__Dxy_plot=[] #save data to plot in scatterplot x,y,z


    #Computation of the cost function to x and y
    for i in range(np.size(Nx)):
        for j in range(np.size(Ny)):
            print(Nx[i], " ", Ny[j])
            ki = np.histogram2d(x,y, bins=(Nx[i],Ny[j]))
            ki = ki[0]   #The mean and the variance are simply computed from the event counts in all the bins of the 2-dimensional histogram.
            k = np.mean(ki) #Mean of event count
            v = np.var(ki)  #Variance of event count
            Cxy[i,j] = (2 * k - v) / ( (Dxy[i,j][0]*Dxy[i,j][1])**2 )  #The cost Function

                                #(Cxy      , Dx          ,  Dy)
            Cxy__Dxy_plot.append((Cxy[i,j] , Dxy[i,j][0] , Dxy[i,j][1]))#Save result of cost function to scatterplot

    Cxy__Dxy_plot = np.array( Cxy__Dxy_plot , dtype=[('Cxy', float),('Dx', float), ('Dy', float)])  #Save result of cost function to scatterplot

    #Optimal Bin Size Selection

    #combination of i and j that produces the minimum cost function
    idx_min_Cxy=np.where(Cxy == np.min(Cxy)) #get the index of the min Cxy

    Cxymin=Cxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]] #value of the min Cxy

    print(sum(Cxy==Cxymin)) #check if there is only one min value

    optDxy=Dxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]]#get the bins size pairs that produces the minimum cost function

    optDx=optDxy[0]
    optDy=optDxy[1]

    idx_Nx=idx_min_Cxy[0][0]#get the index in x that produces the minimum cost function
    idx_Ny=idx_min_Cxy[1][0]#get the index in y that produces the minimum cost function

    print('Cxymin', Cxymin, Nx[idx_Nx], optDx)
    print('Cxymin', Cxymin, Ny[idx_Ny], optDy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x=Cxy__Dxy_plot['Dx']
    y=Cxy__Dxy_plot['Dy']
    z =Cxy__Dxy_plot['Cxy']
    ax.scatter(x, y, z, c=z, marker='o')

    ax.set_xlabel('Dx')
    ax.set_ylabel('Dy')
    ax.set_zlabel('Cxy')
    plt.draw()

    ax.scatter( [optDx], [optDy],[Cxymin], marker='v', s=150,c="red")
    ax.text(optDx, optDy,Cxymin, "Cxy min", color='red')
    plt.draw()
    plt.show()


    return Nx[idx_Nx],Ny[idx_Ny]



def sgolay2d ( z, window_size, order, derivative=None):
    """
    Taken from https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')




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
# events["Delta_Theta_x"] = events.loc[:,'Tracks_thetaOut_x'].values - \
#                                events.loc[:,'Tracks_thetaIn_x'].values

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
# center_offset = float(my.get_from_csv(analysis_configuration_params_file,
#                                              "chan_center_offset"
#                                              ))


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
dech_start, dech_end = my.get_from_csv(analysis_configuration_params_file,
                                     "dech_start",
                                     "dech_end")
# dech_start, dech_end = 1e6*dech_start, 1e6*dech_end # convert to mura
chan_fit_tolerance = my.get_from_csv(analysis_configuration_params_file,
                                 "chan_fit_tolerance")
max_iterations = int(my.get_from_csv(analysis_configuration_params_file,
                                 "chan_max_iterations"))

i = 0
plt.figure()


# TODO SavitzkyGolay filter
hist, x1, y1, img = plt.hist2d(events.loc[:,'Tracks_thetaIn_x'].values, \
 events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values,\
 bins=[200,dtx_nbins], norm=LogNorm(), range=[[-100,100], [dtx_low,dtx_high]]) # ideal 29,17
plt.suptitle(r"Crystal {}, run {} — {} {} GeV".format(crystal_name, run_number, particle_name, particle_energy_input),fontweight='bold')
plt.title(r"Original histogram and calculated offset: {}".format(r"$\theta_{x}$ vs $\Delta \theta_{x}$"))
plt.xlabel(r'$\theta_{x_{in}}\ [\mu rad]$')
plt.ylabel(r'$\Delta \theta_{x}\ [\mu rad]$')
# print(events)
plt.colorbar()
# Window size = next odd number after rounded thetac
window_size_sg = int(np.round(theta_c)) + int(np.round(theta_c))%2 + 1
newhist = sgolay2d(hist, window_size=window_size_sg, order=3);

# Find the maximum only in the channeling spot.
# To do so, consider only the part of the smoothed histogram for which dtx>half thetab
half_thetab_index = np.argwhere(y1==my.take_closest(y1,theta_bending*1e6/2))[0,0]
newhist_upper_half = newhist[:,half_thetab_index:]

ind = np.unravel_index(np.argmax(np.rot90(newhist_upper_half), axis=None), np.rot90(newhist_upper_half).shape);
angular_offset = x1[ind[1]]
print("Calculated offset = ",angular_offset)
plt.axvline(x=angular_offset, linestyle="dashed", color='Crimson', label="")
nocorr_offset_filename = 'offset_nocorr_histo'
plt.savefig("latex/img/"+ nocorr_offset_filename + ".pdf")

plt.matshow(np.rot90(newhist)); plt.plot(ind[1],ind[0],'r.')
filtered_filename = 'offset_filtered_histo'
plt.savefig("latex/img/"+ filtered_filename + ".pdf")
#plt.show()

#plt.figure(); plt.hist2d(events.loc[:,'Tracks_thetaIn_x'].values, events.loc[:,'Tracks_thetaOut_x'].values - events.loc[:,'Tracks_thetaIn_x'].values, bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])

# SET THE CUTS
center_offset = angular_offset
ang_cut_low = [center_offset - theta_c / 2, center_offset - theta_c]
ang_cut_high = [center_offset + theta_c / 2, center_offset + theta_c]


#input("Proceed?")


for low_cut, high_cut in zip(ang_cut_low,ang_cut_high):
    plt.figure()
    geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & \
                                  (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
    totcut_df = geocut_df.loc[(events.loc[:,'Delta_Theta_x'] < dech_start) | \
                                  (events.loc[:,'Delta_Theta_x'] > dech_end)]
    # geocut_df = events.loc[(events.loc[:,'Tracks_thetaIn_x'] > low_cut) & \
    #                               (events.loc[:,'Tracks_thetaIn_x'] < high_cut)]
    # plt.hist2d(geocut_df.loc[:,'Tracks_thetaIn_x'].values, \
    #  geocut_df.loc[:,'Tracks_thetaOut_x'].values - geocut_df.loc[:,'Tracks_thetaIn_x'].values,\
    #  bins=[400,200], norm=LogNorm(), range=[[-100,100], [-80,120]])
    fit_and_data = fit_channeling(totcut_df.Delta_Theta_x,
                                  lowest_percentage, highest_percentage,
                                  chan_fit_tolerance, max_iterations)
    fit_results = fit_and_data[0]
    filtered_data = fit_and_data[1]
    #plt.yscale('log', nonposy='clip')
    plt.hist(geocut_df.Delta_Theta_x, bins=dtx_nbins, range=[dtx_low,dtx_high], normed=False) # [murad]
    # plt.hist(filtered_data, bins=dtx_nbins, range=[dtx_low,dtx_high], normed=False) # [murad]


    total_number_of_events = len(filtered_data)#fit_results["nevents"]
    area_bin = (dtx_high-dtx_low)/dtx_nbins * 1
    gauss_AM = area_bin*total_number_of_events * fit_results["weight_AM"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_AM"], fit_results["sigma_AM"])
    gauss_CH = area_bin*total_number_of_events * fit_results["weight_CH"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_CH"], fit_results["sigma_CH"])
    # gauss_AM = fit_results["weight_AM"]  * matplotlib.mlab.normpdf(x_histo, fit_results["mean_AM"], fit_results["sigma_AM"])
    # gauss_CH = fit_results["weight_CH"] * matplotlib.mlab.normpdf(x_histo, fit_results["mean_CH"], fit_results["sigma_CH"])


    plt.plot(x_histo, gauss_AM, label="Amorphous Peak", color='r')
    plt.plot(x_histo, gauss_CH, label="Channeling Peak", color='Orange')
    thetac_title = r"$\theta_c/2$" if i == 0 else r"$\theta_c$"
    cut_value = theta_c/2 if i == 0 else theta_c
    plt.suptitle(r"{} run {}, {} {} GeV — Chan., cut ± {} = {:.1f}±{:.3}".format(crystal_name,run_number,particle_name,particle_energy_input,thetac_title,angular_offset,cut_value),fontweight='bold')
    plt.title(r"Efficiency {:.3}% ± {:.1f}% — Bending Angle {:.3} ± {:.1f} {}".format(fit_results["weight_CH"]*100, fit_results["weight_CH_err"]*100,
                                                                                fit_results["mean_CH"],fit_results["mean_CH_err"],r"$[\mu rad]$"))
    plt.xlabel(r'$\Delta \theta_{x}\ [\mu rad]$')
    plt.ylabel('Frequency')
    plt.legend()
    #plt.tight_layout()


    thetac_filename = 'offset_half_thetac' if i == 0 else 'offset_thetac'
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
