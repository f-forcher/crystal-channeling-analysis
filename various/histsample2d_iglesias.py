from numpy import array, arange, argmin, sum, mean, var, size, zeros,	where, histogram
from numpy.random import normal
from matplotlib.pyplot import figure, plot, hist, bar, xlabel, ylabel,title, show, savefig
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import inf

import pandas as pd
import sys

file_name = sys.argv[1]

events = pd.read_hdf(file_name).sample(100000)



# x = normal(0, 100, 1e4) # Generate n pseudo-random numbers whit(mu,sigma,n)
# y = normal(0, 100, 1e4) # Generate n pseudo-random numbers whit(mu,sigma,n)
x = events.loc[:,'Tracks_thetaIn_x']
y = events.loc[:,'Tracks_thetaOut_x'] - events.loc[:,'Tracks_thetaIn_x']


x_min = -100
x_max = 100

y_min = -80
y_max = 120



Nx_MIN = 25   #Minimum number of bins in x (integer)
Nx_MAX = 40 #Maximum number of bins in x (integer)

Ny_MIN = 25 #Minimum number of bins in y (integer)
Ny_MAX = 40  #Maximum number of bins in y (integer)


Nx = arange(Nx_MIN, Nx_MAX,1) # #of Bins
Ny = arange(Ny_MIN, Ny_MAX,1) # #of Bins

Dx = float(x_max - x_min) / Nx    #Bin size vector
Dy = float(y_max - y_min) / Ny    #Bin size vector

Dxy=[]
for i in Dx:    #Bin size vector
    a=[]
    for j in Dy:    #Bin size vector
        a.append((i,j))
    Dxy.append(a)
Dxy=array( Dxy, dtype=[('x', float),('y', float)]) #matrix of bin size vector


Cxy=zeros(np.shape(Dxy))

Cxy__Dxy_plot=[] #save data to plot in scatterplot x,y,z

num_of_partitions = 20

#Computation of the cost function to x and y
for i in xrange(size(Nx)):
    for j in xrange(size(Ny)):
        print Nx[i], Ny[j]
        C_part=[]
        for l in range(0,num_of_partitions-1):
            events_part = events.sample(frac=1,replace=True)
            xl = events_part.loc[:,'Tracks_thetaIn_x']
            yl = events_part.loc[:,'Tracks_thetaOut_x'] - events_part.loc[:,'Tracks_thetaIn_x']
            ki = np.histogram2d(xl, yl, bins=(Nx[i],Ny[j]),range=[[x_min,x_max],[y_min,y_max]])
            ki2 = ki[0]
            # ki2 = np.log(ki[0])   #The mean and the variance are simply computed from the event counts in all the bins of the 2-dimensional histogram.
            # ki2[ki2 == -inf] = 0
            k = mean(ki2) #Mean of event count
            v = var(ki2,ddof=0)  #Variance of event count
            C_part.append( (2 * k - v) / ( (Dxy[i,j][0]*Dxy[i,j][1])**2 ) )  #The cost Function for the partition
        Cxy[i,j] = mean(C_part)
                            #(Cxy      , Dx          ,  Dy)
        Cxy__Dxy_plot.append((Cxy[i,j] , Dxy[i,j][0] , Dxy[i,j][1]))#Save result of cost function to scatterplot

Cxy__Dxy_plot = np.array( Cxy__Dxy_plot , dtype=[('Cxy', float),('Dx', float), ('Dy', float)])  #Save result of cost function to scatterplot

#Optimal Bin Size Selection

#combination of i and j that produces the minimum cost function
idx_min_Cxy=np.where(Cxy == np.min(Cxy)) #get the index of the min Cxy

Cxymin=Cxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]] #value of the min Cxy

print sum(Cxy==Cxymin) #check if there is only one min value

optDxy=Dxy[idx_min_Cxy[0][0],idx_min_Cxy[1][0]]#get the bins size pairs that produces the minimum cost function

optDx=optDxy[0]
optDy=optDxy[1]

idx_Nx=idx_min_Cxy[0][0]#get the index in x that produces the minimum cost function
idx_Ny=idx_min_Cxy[1][0]#get the index in y that produces the minimum cost function

print '#', Cxymin, Nx[idx_Nx], optDx
print '#', Cxymin, Ny[idx_Ny], optDy


#PLOTS

#plot histogram2d
fig = figure()
H, xedges, yedges = np.histogram2d(x, y,bins=[Nx[idx_Nx],Ny[idx_Ny]],range=[[x_min,x_max],[y_min,y_max]])
Hmasked = np.ma.masked_where(H==0,H)
plt.imshow( Hmasked.T,extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ] ,interpolation='nearest',origin='lower',aspect='auto',cmap=plt.cm.Spectral)
plt.ylabel("y")
plt.xlabel("x")
plt.colorbar().set_label('z')
plt.show()

#plot scatterplot3d to Dx,Dy and Cxy
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
