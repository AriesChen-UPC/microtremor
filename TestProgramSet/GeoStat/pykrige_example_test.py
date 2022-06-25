# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/15/2022 11:26 AM
@file: pykrige_example_package.py: This script is used to test the pykrige package.
"""

import os
import tkinter
import time
from math import ceil
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pykrige import UniversalKriging
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
import colorama

colorama.init(autoreset=True)
matplotlib.rc('font', family='SimHei', weight='bold')  # set the Chinese font to SimHei
matplotlib.rcParams['axes.unicode_minus'] = False  # display minus sign normally

#%% read and visualize data, x, y, vs

print('\033[0;31mPlease select the data(.xls, .xlsx) for plotting ...\033[0m')
root = tkinter.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print('\033[0;32mFile path is: %s.\033[0m' % file_path)
data = pd.read_excel(file_path)  # data stored in title 'x',''y',''vs'. 'x','y' are coordinates, 'vs' is the value
is_interpolation = False
# TODO: check if the x distance is too large
data_check = data[['x', 'y', 'vs']].dropna(subset=['vs'])
x_check = pd.DataFrame(data_check['x'].drop_duplicates().reset_index(drop=True))
x_distance = np.diff(data_check['x'].drop_duplicates())
x_distance_value = []
x_index = []
x_value = []
x_inter_value = []

for i in range(len(x_distance)):
    if x_distance[i] >= 20:
        is_interpolation = True
        x_distance_value.append(x_distance[i])
        x_index.append(i)
        x_value.append(x_check.loc[i]['x'])

if is_interpolation:
    x_insert = pd.DataFrame(columns=['x', 'y', 'vs'])
    for j in range(len(x_distance_value)):
        x_temp = data_check[data_check['x'] == x_value[j]]
        x_temp['x'].replace(x_value[j], x_value[j] + x_distance_value[j] / 2, inplace=True)
        x_inter_value.append(x_value[j] + x_distance_value[j] / 2)
        x_insert = x_insert.append(x_temp)
    data_insert = pd.concat([data_check, x_insert], axis=0, ignore_index=True)
    data_plot = data_insert
    print('The max distance between two points is: %d' % max(x_distance))
    print('Interpolation has been performed, %d points were added.' % len(x_distance_value))
else:
    data_plot = data[['x', 'y', 'vs']].dropna(subset=['vs'])
    print('No interpolation has been performed automatically.')

# data_plot = data.dropna(subset=['vs'])
x = data_plot['x'].to_numpy()
y = data_plot['y'].to_numpy()
vs = data_plot['vs'].to_numpy()

# scatter plot
sns.scatterplot(data=data, x="x", y="y", hue="vs")
plt.show()

# histogram plot
# sns.set_style("whitegrid")
sns.histplot(data=data, x="vs", kde=True)
# sns.displot(data=data, x="vs", hue="x", kde=True)
plt.show()

#%% Kriging

time_start = time.time()
# build the target grid
gridx = np.arange(x.min(), x.max(), 0.1)
gridy = np.arange(y.min(), y.max(), 0.1)
# Ordinary Kriging
print("\033[0;36mThe current method is: Ordinary Kriging.\033[0m")
OK = OrdinaryKriging(x, y, vs, variogram_model="spherical", nlags=30, weight=True, anisotropy_scaling=2.0,
                     enable_plotting=True, enable_statistics=True, exact_values=False, pseudo_inv=True)
# OK = OrdinaryKriging(x, y, vs, variogram_model="spherical")
vs_, ss = OK.execute("grid", gridx, gridy)  # TODO: vs_ is a masked array
# Universal Kriging
# print("\033[0;36mThe current method is: Universal Kriging.\033[0m")
# UK = UniversalKriging(x, y, vs, variogram_model="spherical", anisotropy_scaling=5.0, exact_values=False)
# vs_, ss = UK.execute("grid", gridx, gridy)

#%% plot the result of Kriging

fig, axes = plt.subplots(1, 1, figsize=(len(gridx) / 10, len(gridy) / 10))
fig_font_size = int((len(gridx) / 10 + len(gridy) / 10) / 2) + 20
# define my own colormap, spectrum and jet
minColor, maxColor = 100, 800
colorRange = maxColor - minColor
minColorIndex = (vs.min() - minColor) / colorRange
maxColorIndex = (vs.max() - minColor) / colorRange
# spectrum colormap
# color_dict = ["blue", "royalblue", "cyan", "yellow", "orange", "red"]
color_dict = ["blue", "cyan", "yellow", "red"]
spectrum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', color_dict, N=512)
spectrum_reference_big = cm.get_cmap(spectrum_cmap, 512)
spectrum_cmap_ = ListedColormap(spectrum_reference_big(np.linspace(0.0, 1.0, 256)))
# mycmap = ListedColormap(spectrum_reference_big(np.linspace(minColorIndex, maxColorIndex, 256)))
mycmap = ListedColormap(spectrum_reference_big(np.linspace(0.0, 0.55, 256)))
# jet colormap
# cmap_reference_big = cm.get_cmap('jet', 512)
# mycmap = ListedColormap(cmap_reference_big(np.linspace(colormap_scale_min, colormap_scale_max, 256)))
# TODO: matshow, imshow
plt.imshow(vs_, cmap=mycmap, zorder=20)  # cmap='RdYlBu_r'
# plt.imshow(vs_, cmap=mycmap, vmin=int(vs.min()), vmax=int(vs.max()), zorder=20)  # cmap='RdYlBu_r'

#%% set the x, y ticks information
if 0 < x.max() - x.min() <= 50:
    if len(gridx) % 50 == 0:
        x_ticks = [i for i in range(0, len(gridx) + 50, 50)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 5, 5)]
        x_labels = [i[-3:] for i in x_label_mile]
    else:
        x_ticks = [i for i in range(0, len(gridx), 50)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 5)]
        x_labels = [i[-3:] for i in x_label_mile]
elif 50 < x.max() - x.min() <= 150:
    if len(gridx) % 100 == 0:
        x_ticks = [i for i in range(0, len(gridx) + 100, 100)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 10, 10)]
        x_labels = [i[-3:] for i in x_label_mile]
    else:
        x_ticks = [i for i in range(0, len(gridx), 100)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 10)]
        x_labels = [i[-3:] for i in x_label_mile]
else:
    if len(gridx) % 200 == 0:
        x_ticks = [i for i in range(0, len(gridx) + 200, 200)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 20, 20)]
        x_labels = [i[-3:] for i in x_label_mile]
    else:
        x_ticks = [i for i in range(0, len(gridx), 200)]
        x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 20)]
        x_labels = [i[-3:] for i in x_label_mile]
# set the x label information
try:
    line_number = data['里程编号'][data['里程编号'].notnull()].unique()[0]
except KeyError:
    line_number = 'Line'
kilo_meter = np.unique(x // 1000)
if len(kilo_meter) == 1:
    line_kilo_num = line_number + str(int(kilo_meter[0])) + '+'
elif len(kilo_meter) == 2:  # TODO: judge the miles between two kilo meters
    line_kilo_num = line_number + str(int(kilo_meter[0])) + '/' + str(int(kilo_meter[1])) + '+'
axes.set_xticks(x_ticks)
axes.set_xticklabels(x_labels, fontsize=fig_font_size)
axes.tick_params(axis='x',
                 which='major',
                 pad=25,
                 length=20, width=2)  # space between x-axis and x-ticks
axes.xaxis.set_ticks_position("bottom")
axes.set_xlabel('里程区间' + '(' + line_kilo_num + ')', fontsize=fig_font_size, labelpad=20, fontweight="bold")
y_ticks = [i for i in range(0, len(gridy) + 10, 50)]
y_labels = [int(i/10) for i in y_ticks]
y_labels.sort(reverse=True)
axes.set_yticks(y_ticks)
axes.set_yticklabels(y_labels, fontsize=fig_font_size)
axes.tick_params(axis='y',
                 direction='in',
                 length=20, width=2)  # space between y-axis and y-ticks
# axes.yaxis.set_ticks_position("left")
axes.set_ylabel('深度(米)', fontsize=fig_font_size,  fontweight="bold")
# set the colorbar
norm = mpl.colors.Normalize(vmin=200, vmax=800)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=spectrum_cmap_),
                    ax=axes, shrink=0.7, pad=0.005, aspect=20)
cb.outline.set_visible(False)
cb.set_label('横波速度(米/秒)', fontsize=fig_font_size,  fontweight="bold")
cb.ax.tick_params(labelsize=fig_font_size, rotation=90, length=20, width=2)
cb.ax.invert_yaxis()

# if vs.max() - vs.min() > 400:  # if the z data range is large(>400), set the colorbar ticks = 100
#     color_bar_min = int(vs.min() / 100) * 100
#     color_bar_max = ceil(vs.max() / 100) * 100
#     cb = plt.colorbar(ax=axes, shrink=0.6, aspect=20, pad=0.005, fraction=0.1, format='%.0f',
#                       ticks=np.arange(color_bar_min, color_bar_max + 100, 100))
# else:
#     color_bar_min = int(vs.min() / 50) * 50
#     color_bar_max = ceil(vs.max() / 50) * 50
#     cb = plt.colorbar(ax=axes, shrink=0.6, aspect=20, pad=0.005, fraction=0.1, format='%.0f',
#                       ticks=np.arange(color_bar_min, color_bar_max + 50, 50))  # FIXME: the label is not complete
# cb.outline.set_visible(False)
# cb.set_label('横波速度(米/秒)', fontsize=fig_font_size,  fontweight="bold")
# cb.ax.tick_params(labelsize=fig_font_size, rotation=90, length=20, width=2)
# cb.ax.invert_yaxis()

# plot the metro lines
try:
    metro_x = (data['隧道里程'].dropna() - x.min()) / 0.1
    metro_y_up = len(gridy) - data['隧道顶面'].dropna() / 0.1
    metro_y_down = len(gridy) - (data['隧道顶面'].dropna() + data['隧道范围'].dropna()[0]) / 0.1
    plt.plot(metro_x, metro_y_up, metro_x, metro_y_down, color='black', linewidth=5.0, zorder=30)
except KeyError:
    pass
# plot the microtremor points
points_label = data['x'].dropna().drop_duplicates().reset_index(drop=True)
try:
    is_real = data['isReal'].dropna().reset_index(drop=True)
    if 0 in is_real:
        is_man_interpolation = True
    points_all = pd.concat([points_label, is_real], axis=1)
    points_real = points_all[points_all['isReal'] == 1].reset_index(drop=True)
    points_unreal = points_all[points_all['isReal'] == 0].reset_index(drop=True)
    point_x_real = points_real['x'].values
except KeyError:
    is_man_interpolation = False
    point_x_real = points_label.values
points_x = (point_x_real - point_x_real.min()) / 0.1
points_y = np.ones(len(points_x)) * (len(gridy) + 5)
label_size = len(gridx) if len(gridx) > len(gridy) else len(gridy)  # set the size of the label
plt.scatter(points_x, points_y, marker='v', c='c', s=label_size, zorder=10)
# plot mask for interpolation points
if is_interpolation:
    if is_man_interpolation:
        for k in range(len(points_unreal)):
            x_inter_value.append(points_unreal['x'].values[k])
else:
    if is_man_interpolation:
        for k in range(len(points_unreal)):
            x_inter_value.append(points_unreal['x'].values[k])
if x_inter_value:
    for value in x_inter_value:
        mask_value = (value - x.min()) / 0.1 - 30
        axes.add_patch(patches.Rectangle((mask_value, 0), 60, len(gridy), color='white', alpha=0.5, zorder=40))
# set the border of the figure
axes.set_xlim((-5, len(gridx) + 5))
axes.set_ylim((0, len(gridy) + 10))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)  # remove the figure border
plt.show()
fig_save_name = os.path.split(file_path)[0] + '/' + os.path.split(file_path)[1].split('.')[0] + '.png'
fig.savefig(fig_save_name, format='png', bbox_inches='tight', dpi=96, transparent=True)
print('The figure is saved as %s.' % fig_save_name)
time_end = time.time()
print("\033[0;33mThe time consumed is %.2f seconds. \033[0m" % (time_end - time_start))
print('\033[0;32mDone!\033[0m')
input('Press Enter to exit...\n')

#%% main

if __name__ == '__main__':
    pass
