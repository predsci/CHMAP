"""
Created on Thu Oct 25 11:32:47 2012

Histogram Binwidth Optimization Method

Shimazaki and Shinomoto, Neural Comput 19 1503-1527, 2007
2006 Author Hideaki Shimazaki, Matlab
Department of Physics, Kyoto University
shimazaki at ton.scphys.kyoto-u.ac.jp
Please feel free to use/modify this program.

Version in python adapted Érbet Almeida Costa

Data: the duration for eruptions of
the Old Faithful geyser in Yellowstone National Park (in minutes)
or normal distribuition.
Version in python adapted Érbet Almeida Costa
Bugfix by Takuma Torii 2.24.2013

"""

import numpy as np
from numpy import mean, size, zeros, where
from scipy import linspace, stats

import chmap.utilities.datatypes.datatypes as PSI_dtypes

from matplotlib.pyplot import figure, plot, xlabel, ylabel,\
    title, hist

lat_band = [-np.pi/64., np.pi/64.]
R0 = 1.01
hdf_fname = '/Volumes/Work/CHD_DB/processed_images/2011/03/26/aia_lvl2_20110326T040031_193.h5'

los_image = PSI_dtypes.read_euv_image(hdf_fname)
los_image.get_coordinates(R0=R0)

lat_band_index = np.logical_and(los_image.lat <= max(lat_band), los_image.lat >= min(lat_band))
mu_min = 0.14
mu_max = 1.0
mu_index = np.logical_and(los_image.mu >= mu_min, los_image.mu <= mu_max)
use_index = np.logical_and(mu_index, lat_band_index)

use_mu = los_image.mu[use_index]
use_data = los_image.data[use_index]

use_data[use_data < 1.] = 1.
x = np.log10(use_data)

# calc Freedman-Diaconis bin size
iqr = stats.iqr(x)
FDh = 2*iqr/(len(x)**(1./3.))

# x = normal(0, 100, 1e2) # Generate n pseudo-random numbers whit(mu,sigma,n)
#x = [4.37,3.87,4.00,4.03,3.50,4.08,2.25,4.70,1.73,4.93,1.73,4.62,\
#3.43,4.25,1.68,3.92,3.68,3.10,4.03,1.77,4.08,1.75,3.20,1.85,\
#4.62,1.97,4.50,3.92,4.35,2.33,3.83,1.88,4.60,1.80,4.73,1.77,\
#4.57,1.85,3.52,4.00,3.70,3.72,4.25,3.58,3.80,3.77,3.75,2.50,\
#4.50,4.10,3.70,3.80,3.43,4.00,2.27,4.40,4.05,4.25,3.33,2.00,\
#4.33,2.93,4.58,1.90,3.58,3.73,3.73,1.82,4.63,3.50,4.00,3.67,\
#1.67,4.60,1.67,4.00,1.80,4.42,1.90,4.63,2.93,3.50,1.97,4.28,\
#1.83,4.13,1.83,4.65,4.20,3.93,4.33,1.83,4.53,2.03,4.18,4.43,\
#4.07,4.13,3.95,4.10,2.27,4.58,1.90,4.50,1.95,4.83,4.12]

x_max = max(x)
x_min = min(x)
N_MIN = 20      # Minimum number of bins (integer)
                # N_MIN must be more than 1 (N_MIN > 1).
N_MAX = 700     # Maximum number of bins (integer)
N = range(N_MIN, N_MAX)  # #of Bins
N = np.array(N)
D = (x_max-x_min)/N     # Bin size vector
C = zeros(shape=(size(D), 1))

# Computation of the cost function
for i in range(size(N)):
    edges = linspace(x_min, x_max, N[i]+1)  # Bin edges
    ki = hist(x, edges)                     # Count # of events in bins
    ki = ki[0]
    k = mean(ki)                            # Mean of event count
    v = sum((ki-k)**2)/N[i]                 # Variance of event count
    C[i] = (2*k-v)/((D[i])**2)              # The cost Function

# Optimal Bin Size Selection
cmin = min(C)
idx  = where(C == cmin)
idx  = int(idx[0])
optD = D[idx]

edges = linspace(x_min, x_max, N[idx]+1)
fig = figure()
ax = fig.add_subplot(111)
ax.hist(x, edges)
title(u"AIA Lat-band Histogram")
ylabel(u"Frequency")
xlabel(u"log10 Intensity")
# savefig('Hist.png')
fig = figure()
plot(D, C, '.b', optD, cmin, '*r')
xlabel('Bin Size')
ylabel('Cost Function')
# savefig('Fobj.png')



