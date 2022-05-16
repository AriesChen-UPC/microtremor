# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 4/14/2022 10:10 AM
@file: scikit_gstat_example.py
"""

import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skgstat as skg
from matplotlib import cm
from matplotlib.colors import ListedColormap
from skgstat import Variogram
from math import ceil
import time
import warnings
warnings.filterwarnings("ignore")

#%% OrdinaryKriging, read and visualize data

print('\033[0;31mPlease select the data(.xls, .xlsx) for plotting ...\033[0m')
root = tkinter.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print('\033[0;32mFile path is: %s.\033[0m' % file_path)
data = pd.read_excel(file_path)  # data stored in title 'x',''y',''vs','x','y' are coordinates, 'vs' is the value
# data = data.dropna()  # drop nan in orignal data
x = data['x'].to_numpy()
y = data['y'].to_numpy()
vs = data['vs'].to_numpy()

# view the data points
# fig_points, ax_points = plt.subplots(1, 1, figsize=(9, 9))
# art_points = ax_points.scatter(x, y, s=50, c=vs.flatten(), cmap='viridis')
# ax_points.set_title('Points', fontsize=15, fontweight='bold', pad=25)
# ax_points.set_xlabel('Distance (m)', fontsize=15, labelpad=20, fontweight="bold")
# ax_points.set_ylabel('Depth (m)', fontsize=15, labelpad=20, fontweight="bold")
# cb_points = plt.colorbar(art_points)
# cb_points.set_label('Vs (m/s)', fontsize=15, fontweight="bold")
# cb_points.ax.tick_params(labelsize=15, rotation=90)
# cb_points.ax.invert_yaxis()
# plt.show()

# parameters for ordinary kriging
V_test = skg.Variogram(coordinates=data[['x', 'y']].values, values=vs)
V_test.model = 'spherical'
V_test.n_lags = 15
V_test.maxlag = 500
fig = V_test.plot()


#%% Ordinary Kriging

print('\033[0;31mOrdinary Kriging ...\033[0m')
time_start = time.time()
V = Variogram(data[['x', 'y']].values, data.vs.values, model='spherical', bin_func='kmeans',
              maxlag='median')
ok = skg.OrdinaryKriging(V, min_points=1, max_points=15)  # todo: min_points, max_points
# build the target grid
interval = 0.1
xx, yy = np.mgrid[x.min():x.max():interval, y.min():y.max():interval]
x_grid_num = int((x.max() - x.min()) / interval)
y_grid_num = int((y.max() - y.min()) / interval)
field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2 = ok.sigma.reshape(xx.shape)
fig, axes = plt.subplots(1, 1, figsize=(int(x_grid_num/10), int(y_grid_num/10)))
fig_font_size = int((x_grid_num/10 + y_grid_num/10)/2) + 5  # todo: font size
# rescale the coordinates to fit the interpolation raster
x_ = (x - x.min()) / (x.max() - x.min()) * x_grid_num
y_ = (y - y.min()) / (y.max() - y.min()) * y_grid_num
# define my own colormap, using 'jet' colormap as reference
colormap_min = 200
colormap_max = 800
colormap_range = colormap_max - colormap_min
colormap_scale_min = (vs.min() - colormap_min) / colormap_range * 0.7 + 0.15
colormap_scale_max = (vs.max() - colormap_min) / colormap_range * 0.7 + 0.15
jet_big = cm.get_cmap('jet', 512)  # cmap='RdYlBu_r'
mycmap = ListedColormap(jet_big(np.linspace(colormap_scale_min, colormap_scale_max, 256)))
# todo: matshow, imshow
art = axes.matshow(field.T, origin='lower', cmap=mycmap, vmin=vs.min(), vmax=vs.max())  # _r means reverse
axes.set_title('Vs Profile', fontsize=fig_font_size, fontweight='bold', pad=25)
# axes.plot(x_, y_, '+w')  # plot the data points
axes.set_xlim((0, x_grid_num))
axes.set_ylim((0, y_grid_num))
if x.max() - x.min() >= 50:  # if the x data range is large(>50m), set the x ticks = 10m
    if x_grid_num % 100 == 0:
        x_ticks = [i for i in range(0, x_grid_num + 100, 100)]
        x_label_mile = ['%03d' % i for i in range(0, int(x_grid_num / 10) + 10, 10)]
        x_labels = [('ZDK0+' + str(i)) for i in x_label_mile]
    else:
        x_ticks = [i for i in range(0, x_grid_num, 100)]
        x_label_mile = ['%03d' % i for i in range(0, int(x_grid_num / 10), 10)]
        x_labels = [('ZDK0+' + str(i)) for i in x_label_mile]
else:
    if x_grid_num % 50 == 0:
        x_ticks = [i for i in range(0, x_grid_num + 50, 50)]
        x_label_mile = ['%03d' % i for i in range(0, int(x_grid_num / 10) + 5, 5)]
        x_labels = [('ZDK0+' + str(i)) for i in x_label_mile]
    else:
        x_ticks = [i for i in range(0, x_grid_num, 50)]
        x_label_mile = ['%03d' % i for i in range(0, int(x_grid_num / 10), 5)]
        x_labels = [('ZDK0+' + str(i)) for i in x_label_mile]  # todo: 'ZDK0+' is decided by the data
axes.set_xticks(x_ticks)
axes.set_xticklabels(x_labels, fontsize=fig_font_size)
axes.tick_params(axis='x', which='major', pad=25)  # space between x-axis and x-label
axes.xaxis.set_ticks_position("bottom")
axes.set_xlabel('Mileage Interval', fontsize=fig_font_size, labelpad=20, fontweight="bold")
y_ticks = [i for i in range(0, y_grid_num + 50, 50)]
y_labels = [int(i/10) for i in y_ticks]
y_labels.sort(reverse=True)
axes.set_yticks(y_ticks)
axes.set_yticklabels(y_labels, fontsize=fig_font_size)
axes.yaxis.set_ticks_position("left")
axes.set_ylabel('Depth (m)', fontsize=fig_font_size,  fontweight="bold")
# colorbar legend setting
if vs.max() - vs.min() > 300:  # if the z data range is large(>300), set the colorbar ticks = 100
    color_bar_min = int(vs.min() / 100) * 100
    color_bar_max = ceil(vs.max() / 100) * 100
    cb = plt.colorbar(art, ax=axes, shrink=0.8, aspect=10, pad=0.05, fraction=0.1, format='%.0f',
                      ticks=np.arange(color_bar_min, color_bar_max + 100, 100))
else:
    color_bar_min = int(vs.min() / 50) * 50
    color_bar_max = ceil(vs.max() / 50) * 50
    cb = plt.colorbar(art, ax=axes, shrink=0.8, aspect=10, pad=0.05, fraction=0.1, format='%.0f',
                      ticks=np.arange(color_bar_min, color_bar_max + 50, 50))
cb.set_label('Vs (m/s)', fontsize=fig_font_size,  fontweight="bold")
cb.ax.tick_params(labelsize=fig_font_size, rotation=90)
cb.ax.invert_yaxis()
plt.show()
time_end = time.time()
print("\033[0;33mThe time consumed is %.2f seconds. \033[0m" % (time_end - time_start))
print('\033[0;32mDone!\033[0m')
