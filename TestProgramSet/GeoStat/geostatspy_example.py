# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/21/2022 4:48 PM
@file: geostatspy_example.py
"""

import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats
import matplotlib.pyplot as plt
import scipy.stats

# Make a 2d simulation
nx = 100
ny = 100
cell_size = 10
xmin = 0.0
ymin = 0.0
xmax = xmin + nx * cell_size
ymax = ymin + ny * cell_size
seed = 74073
range_max = 1800
range_min = 500
azimuth = 65
vario = GSLIB.make_variogram(0.0, nst=1, it1=1, cc1=1.0, azi1=65, hmaj1=1800, hmin1=500)
mean = 10.0
stdev = 2.0
vmin = 4
vmax = 16
cmap = plt.cm.plasma

# calculate a stochastic realization with standard normal distribution
sim = GSLIB.sgsim_uncond(1, nx, ny, cell_size, seed, vario, "simulation")
sim = GSLIB.affine(sim, mean, stdev)

# extract samples from the 2D realization
sampling_ncell = 10  # sample every 10th node from the model
samples = GSLIB.regular_sample(sim, xmin, xmax, ymin, ymax, sampling_ncell, 10, 10, nx, ny, 'Realization')

# remove samples to create a sample bias (preferentially removed low values to bias high)
samples_cluster = samples.drop([80, 79, 78, 73, 72, 71, 70, 65, 64, 63, 61, 57, 56, 54, 53, 47, 45, 42])
samples_cluster = samples_cluster.reset_index(drop=True)
GSLIB.locpix(sim, xmin, xmax, ymin, ymax, cell_size, vmin, vmax, samples_cluster, 'X', 'Y', 'Realization',
             'Porosity Realization and Regular Samples', 'X(m)', 'Y(m)', 'Porosity (%)', cmap, "Por_Samples")

# apply the declus program convert to Python
wts, cell_sizes, averages = geostats.declus(samples_cluster, 'X', 'Y', 'Realization', iminmax=1, noff=5, ncell=100,
                                            cmin=1, cmax=2000)
samples_cluster['wts'] = wts            # add the weights to the sample data
samples_cluster.head()

# plot the results and diagnostics for the declustering
plt.subplot(321)
GSLIB.locmap_st(samples_cluster, 'X', 'Y', 'wts', xmin, xmax, ymin, ymax, 0.0, 2.0,
                'Declustering Weights', 'X (m)', 'Y (m)', 'Weights', cmap)

plt.subplot(322)
GSLIB.hist_st(samples_cluster['wts'], 0.0, 2.0, log=False, cumul=False, bins=20, weights=None, xlabel="Weights",
              title="Declustering Weights")
plt.ylim(0.0, 20)

plt.subplot(323)
GSLIB.hist_st(samples_cluster['Realization'], 0.0, 20.0, log=False, cumul=False, bins=20, weights=None,
              xlabel="Porosity", title="Naive Porosity")
plt.ylim(0.0, 20)

plt.subplot(324)
GSLIB.hist_st(samples_cluster['Realization'], 0.0, 20.0, log=False, cumul=False, bins=20,
              weights=samples_cluster['wts'], xlabel="Porosity", title="Naive Porosity")
plt.ylim(0.0, 20)

# Plot the declustered mean vs. cell size to check the cell size selection
plt.subplot(325)
plt.scatter(cell_sizes, averages, c="black", marker='o', alpha=0.2, edgecolors="none")
plt.xlabel('Cell Size (m)')
plt.ylabel('Porosity Average (%)')
plt.title('Porosity Average vs. Cell Size')
plt.ylim(8, 12)
plt.xlim(0, 2000)

print(scipy.stats.describe(wts))

plt.show()
