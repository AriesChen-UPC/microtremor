# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/19/2022 6:28 PM
@file: spac_hvsr_compare_fix.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import pandas as pd
from read_target import read_target
from target_plotly import target_plotly
from hvsr_plotly import hvsr_plotly
import colorama
colorama.init(autoreset=True)

# select the directory of the data
print('\033[0;31mPlease select the folder of data: \033[0m')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
period_path = glob(os.path.join(folder_path, '*'))
print('\033[0;34mThe path of the data is: %s.\033[0m' % folder_path)
print('\033[0;34mThe number of the period is: %d.\033[0m' % len(period_path))
# initialize the global variables
period_spac_total = []
period_hvsr_total = []
period_name = []
# read the data
for i in range(len(period_path)):
    subfolder_name = os.listdir(period_path[i])
    format_name = [subfolder_name[i].replace('+', '') for i in range(len(subfolder_name))]
    period_spac = {}
    perido_hvsr = {}
    for j in range(len(subfolder_name)):
        period_spac[format_name[j]] = glob(period_path[i] + '/' + subfolder_name[j] + '/*.target')
        perido_hvsr[format_name[j]] = glob(period_path[i] + '/' + subfolder_name[j] + '/*.hv')
    period_spac_total.append(period_spac)
    period_hvsr_total.append(perido_hvsr)
    period_name.append(period_path[i].split('\\')[-1])
# store the period name and data in dict
period_spac_dict = dict(zip(period_name, period_spac_total))
period_hvsr_dict = dict(zip(period_name, period_hvsr_total))
# merge the spac data by name, and plot the spac data
spac_combine = {}
for key in period_spac_dict:
    for k, v in period_spac_dict[key].items():
        if k in spac_combine:
            spac_combine[k].extend(v)
        else:
            spac_combine[k] = v
for key in spac_combine.keys():
    ring_all, freq_all, spac_all, is_ring_same = read_target(spac_combine[key])
    period_spac_label = []
    for m in range(len(spac_combine[key])):
        period_spac_label.append(spac_combine[key][m].replace('\\', '/').split('/')[-3])
    if is_ring_same:
        ring_name = ring_all[0]
        target_plotly(freq_all, spac_all, ring_name, key, period_spac_label, folder_path)
    else:
        print('\033[0;31mThe ring is not the same in the point %s.\033[0m' % key)
# merge the hvsr data by name, and plot the hvsr data
hvsr_combine = {}
for key in period_hvsr_dict:
    for k, v in period_hvsr_dict[key].items():
        if k in hvsr_combine:
            hvsr_combine[k].extend(v)
        else:
            hvsr_combine[k] = v
for key in hvsr_combine.keys():
    hvsr_all = []
    for hvsr in hvsr_combine[key]:
        hvsr_all.append(pd.read_table(filepath_or_buffer=hvsr, sep='\t', skiprows=9,
                                      names=['freq', 'avg', 'max', 'min']))
    period_hvsr_label = []
    for n in range(len(hvsr_combine[key])):
        period_hvsr_label.append(hvsr_combine[key][n].replace('\\', '/').split('/')[-3])
    hvsr_plotly(hvsr_all, key, period_hvsr_label, folder_path)
print('\033[0;32mThe program is finished.\033[0m')
