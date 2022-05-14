# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/26/2022 10:19 AM
@file: spac_hvsr_compare_test.py
"""

import os
import tkinter
import tkinter as tk
from tkinter import filedialog
from glob import glob
import pandas as pd
from read_target import read_target
from target_plotly import target_plotly
from hvsr_plotly import hvsr_plotly
from spac_cps import spac_cps
import colorama
colorama.init(autoreset=True)

# initialize the gobal variables
print('\033[0;31mPlease input the numbers of periods: \033[0m')
period_num = int(input())
period_spac_total = []
period_hvsr_total = []
period_name = []
period_init = 1
# read the target&hv file from different periods, write the index name in dict, save the data in list
for i in range(period_num):
    print('\033[0;31mPlease selsect the period of %d: \033[0m' % period_init)
    period_init += 1
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    print('\033[0;34mThe path of %d is: %s \033[0m' % (period_init, folder_path))
    subfolder_name = os.listdir(folder_path)  # the filename format is not same, with or without '+'
    format_name = [subfolder_name[i].replace('+', '') for i in range(len(subfolder_name))]
    period_spac = {}
    perido_hvsr = {}
    for j in range(len(subfolder_name)):
        period_spac[format_name[j]] = glob(folder_path + '/' + subfolder_name[j] + '/*.target')
        perido_hvsr[format_name[j]] = glob(folder_path + '/' + subfolder_name[j] + '/*.hv')
    period_spac_total.append(period_spac)
    period_hvsr_total.append(perido_hvsr)
    period_name.append(folder_path.split('/')[-1])
# select the same target&hv file in different periods, using the key of each dict
for i in range(period_num - 1):
    point_same = period_spac_total[i].keys() & period_spac_total[i + 1].keys()
point_same_list = list(point_same)
ring_sample, freq_sample, spac_sample, is_ring_same_sample = read_target(period_spac_total[0][point_same_list[0]])
print('\033[0;32mThe same points are: %s \033[0m' % point_same_list)
# choose whether to use CPS or not
is_cps = False
print('\033[0;31mDo you want to use CPS? (y/n) \033[0m')
use_cps = input().upper()
if use_cps == 'Y':
    is_cps = True
    print('\033[0;31mPlease select the model from inversion ... \033[0m')
    root = tkinter.Tk()
    root.withdraw()
    model = filedialog.askopenfilename()
    theory_freq_all = []
    theory_spac_all = []
    for radius in ring_sample[0]:
        theory_freq, theory_spac = spac_cps(model, radius)
        theory_freq_all.append(theory_freq)
        theory_spac_all.append(theory_spac)
    print('\033[0;32mDispersion curve and SPAC curve calculated by CPS were done!\033[0m')
# read the target&hv file and plot the result
print('\033[0;31mStart ...\033[0m')
for point_name in point_same:
    point_spac_same_list = []
    point_hvsr_same_list = []
    for i in range(period_num):
        point_spac_same_list.append(period_spac_total[i][point_name][0])
        point_hvsr_same_list.append(period_hvsr_total[i][point_name][0])
    # spac
    # todo: if the target has different rings number
    ring_all, freq_all, spac_all, is_ring_same = read_target(point_spac_same_list)
    if is_ring_same:
        ring_name = ring_all[0]
        if is_cps:
            target_plotly(freq_all, spac_all, ring_name, point_name, period_name, folder_path, is_cps=True,
                          theory_freq=theory_freq_all, theory_spac=theory_spac_all)
        else:
            target_plotly(freq_all, spac_all, ring_name, point_name, period_name, folder_path)
    else:
        pass
    # hvsr
    hvsr_all = []
    for hvsr in point_hvsr_same_list:
        hvsr_all.append(pd.read_table(filepath_or_buffer=hvsr, sep='\t', skiprows=9,
                                      names=['freq', 'avg', 'max', 'min']))
    hvsr_plotly(hvsr_all, point_name, period_name, folder_path)
print('\033[0;32mDone!\033[0m')
input('Press Enter to exit...\n')

if __name__ == '__main__':
    pass
