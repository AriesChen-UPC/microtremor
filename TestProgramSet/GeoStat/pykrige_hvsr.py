# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/19/2022 2:48 PM
@file: pykrige_hvsr.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige import OrdinaryKriging

#%% read the hvsr data from .hv file, edit the headers and stored in dataframe format

print('Please select the hvsr data folder：')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;36mThe hvsr data folder is %s.\033[0m' % folder_path)
root.destroy()
fs = glob(folder_path + '/*.hv')
with open(fs[0]) as f:
    lines = f.readlines()
    skip_rows = 0  # count the number of skipped rows
    for line in lines:
        if line.startswith('#'):
            skip_rows += 1
hvsr_data = []
for i in range(len(fs)):
    hvsr_data.append(pd.read_table(filepath_or_buffer=fs[i], sep='\t', skiprows=skip_rows,
                                   names=['freq', 'avg', 'max', 'min']))
    x_data = float(os.path.basename(fs[i]).split('.')[0])  # todo: choose the x data from the file name
    hvsr_data[i].insert(hvsr_data[i].shape[1], 'x', x_data)
    hvsr_data[i].insert(hvsr_data[i].shape[1], 'y', 230/4/hvsr_data[i]['freq'])  # vs30 = 230m/s, freq to depth
# define the min and max of freq(depth)
freq_min = input('Please input the min freq: \n')
print('\033[0;33mThe min freq is %s Hz.\033[0m' % freq_min)
freq_max = input('Please input the max freq: \n')
print('\033[0;33mThe max freq is %s Hz.\033[0m' % freq_max)
for i in range(len(hvsr_data)):
    hvsr_data[i] = hvsr_data[i][(hvsr_data[i]['freq'] >= float(freq_min)) & (hvsr_data[i]['freq'] <= float(freq_max))]
# combine the hvsr data
hvsr_plot = hvsr_data[0][['x', 'y', 'avg']][0:len(hvsr_data[0]):10]  # resample the data
for i in range(1, len(hvsr_data)):
    hvsr_plot = pd.concat([hvsr_plot, hvsr_data[i][['x', 'y', 'avg']][0:len(hvsr_data[i]):10]])

#%% Ordinary Kriging

print("\033[0;36mThe current method is: Ordinary Kriging.\033[0m")
gridx = np.arange(hvsr_plot['x'].min(), hvsr_plot['x'].max(), 0.1)
gridy = np.arange(hvsr_plot['y'].min(), hvsr_plot['y'].max(), 0.1)
OK = OrdinaryKriging(hvsr_plot['x'], hvsr_plot['y'], hvsr_plot['avg'], variogram_model="spherical",
                     verbose=False, enable_plotting=False)
vs_, ss = OK.execute("grid", gridx, gridy)  # todo: z_ is a masked array
# todo: matshow, imshow
plt.figure()
ax = plt.gca()
im = ax.imshow(vs_, cmap='RdYlBu_r')  # cmap='RdYlBu_r'
x_ticks = [i for i in range(0, len(gridx), 50)]
x_labels = [int(i/10) for i in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.xaxis.set_ticks_position("bottom")
ax.set_xlabel('Distance (m)', fontweight="bold")
y_ticks = [i for i in range(0, len(gridy), 50)]
y_labels = [int(i/10) for i in y_ticks]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.yaxis.set_ticks_position("left")
ax.set_ylabel('Depth (m)', fontweight="bold")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax)
cb.ax.invert_yaxis()
cb.set_label('HVSR', fontweight="bold", rotation=90)
plt.show()
# contour
plt.figure()
ax_contour = plt.gca()
plt.contour(gridx, gridy, vs_, 15, colors='black')
plt.contourf(gridx, gridy, vs_, 15, cmap='RdYlBu_r')
ax_contour.invert_yaxis()
plt.colorbar()
plt.show()
print('\033[0;32mDone!\033[0m')
