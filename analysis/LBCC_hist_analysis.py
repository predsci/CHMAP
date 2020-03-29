
"""
Analyze intensity bin size, averaging window, image cadence, and averaging cadence.


"""

import os
from scipy import interpolate, stats
import numpy as np
import pickle


# load aia histograms for 2011
f = open('/Users/turtle/GitReps/CHD/test_data/mu-hists-2011_AIA.pkl', 'rb')
aia_hists = pickle.load(f)
f.close()

all_hists = aia_hists['all_hists']

intensity_bins = aia_hists['intensity_bin_edges']
intensity_centers = intensity_bins[0:-1] + np.diff(intensity_bins)/2

tck, fp, flag = interpolate.splrep(intensity_centers, all_hists[1, :, 1])
