# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 6/7/2022 12:05 PM
@file: spac_slice.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from read_target import read_target

# select the directory of the data
print('\033[0;31mPlease select the folder of data: \033[0m')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;34mThe path of the data is: %s.\033[0m' % folder_path)
print('------------------------------')
# initialize the global variables
sapc_path = glob(os.path.join(folder_path, '*.target'))
ring_all, freq_all, spac_all, is_ring_same = read_target(sapc_path)
print('\033[0;34mPlease input the min of the frequency: \033[0m')
freq_min = float(input())
print('\033[0;34mPlease input the max of the frequency: \033[0m')
freq_max = float(input())
slice_data = np.arange(freq_min, freq_max + 1.0, 1.0)
for data in slice_data:
    index = 0
    for i in range(len(freq_all[0][0])):
        if freq_all[0][0][i] > data:
            index = i
            break
    spac_slice_min = []
    for j in range(len(spac_all)):
        spac_slice_min.append(spac_all[j][0][index])
    spac_slice_min = np.array(spac_slice_min).reshape(54, 5)
    # plot the slice data
    fig, axes = plt.subplots(figsize=(15, 30))
    plt.imshow(spac_slice_min, cmap='jet', aspect='auto')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title('Frequency: %.2f Hz' % data, fontsize=30)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=30)
    fig_name = folder_path + '/spac_slice_%.0f.png' % data
    plt.savefig(fig_name, dpi=300)
    plt.show()

print('\033[0;34mThe slice data is saved in the folder of data.\033[0m')
