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
from spac_plotly import spac_plotly
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
print('------------------------------')
print('\033[0;34mThe number of the period is: %d.\033[0m' % len(period_path))
print('------------------------------')
# initialize the global variables
period_spac_total = []
period_hvsr_total = []
period_name = []
# read the data
for i in range(len(period_path)):
    subfolder_name = os.listdir(period_path[i])
    format_name = [subfolder_name[i].replace('+', '') for i in range(len(subfolder_name))]  # remove '+' in the name
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
spac_combine = {}  # TODO: streamline code here
for spac_key in period_spac_dict:
    for spac_k, spac_v in period_spac_dict[spac_key].items():
        if spac_k in spac_combine:
            spac_combine[spac_k].extend(spac_v)
        else:
            spac_combine[spac_k] = spac_v
for spac_all_key in spac_combine.keys():  # TODO: change the method of reading .target file
    ring_all, freq_all, spac_all, is_ring_same = read_target(spac_combine[spac_all_key])
    period_spac_label = []
    for m in range(len(spac_combine[spac_all_key])):
        period_spac_label.append(spac_combine[spac_all_key][m].replace('\\', '/').split('/')[-3])  # get the period name
    if is_ring_same:
        print('\033[0;32mThe ring is the same in the point of %s.\033[0m' % spac_all_key)
        ring_name = ring_all[0]
        spac_plotly(freq_all, spac_all, ring_name, spac_all_key, period_spac_label, folder_path)
    else:
        print('\033[0;31mThe ring is not the same in the point of %s.\033[0m' % spac_all_key)
# merge the hvsr data by name, and plot the hvsr data
hvsr_combine = {}
for hvsr_key in period_hvsr_dict:
    for hvsr_k, hvsr_v in period_hvsr_dict[hvsr_key].items():
        if hvsr_k in hvsr_combine:
            hvsr_combine[hvsr_k].extend(hvsr_v)
        else:
            hvsr_combine[hvsr_k] = hvsr_v
for hvsr_all_key in hvsr_combine.keys():
    hvsr_all = []
    for hvsr in hvsr_combine[hvsr_all_key]:
        hvsr_all.append(pd.read_table(filepath_or_buffer=hvsr, sep='\t', skiprows=9,  # header lines = 9
                                      names=['freq', 'avg', 'max', 'min']))
    period_hvsr_label = []
    for n in range(len(hvsr_combine[hvsr_all_key])):
        period_hvsr_label.append(hvsr_combine[hvsr_all_key][n].replace('\\', '/').split('/')[-3])
    hvsr_plotly(hvsr_all, hvsr_all_key, period_hvsr_label, folder_path)
print('------------------------------')
print('\033[0;32mThe program is finished.\033[0m')
