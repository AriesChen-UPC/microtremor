# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 3/16/2022 8:50 AM
@file: sac_slice_detrend.py
"""

import os
from glob import glob
import numpy as np
from obspy import read
from obspy import UTCDateTime
import tkinter as tk
from tkinter import filedialog
from scipy import signal
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from itertools import combinations


def get_list_files(folder_path):
    all_file_path = []
    for root, dirs, files in os.walk(folder_path):
        for files_path in files:
            all_file_path.append(os.path.join(root, files_path))
    return all_file_path


root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
all_file_path = get_list_files(folder_path)
z_component_path = []  # Z component sac file path for example
for i in all_file_path:
    if "Z.SAC" in i or "Z.Q.SAC" in i:  # sac file name format for CGS
        z_component_path.append(i)

start_time = UTCDateTime(input('Please input the start time, format in yyyy-mm-ddThh:mm:ss \n'))
end_time = UTCDateTime(input('Please input the end time, format in yyyy-mm-ddThh:mm:ss \n'))
# tr_slice_data = []
sac_slice_detrend_path = os.path.dirname(folder_path) + '/sac_slice_detrend'
if not os.path.exists(sac_slice_detrend_path):
    os.mkdir(os.path.dirname(folder_path) + '/sac_slice_detrend')
for i in trange(len(z_component_path)):
    tr = read(z_component_path[i])
    tr_slice = tr.slice(start_time, end_time)
    # tr_slice.detrend("spline", order=3, dspline=500)
    tr_slice.write(sac_slice_detrend_path + '/' + os.path.basename(z_component_path[i]).split('.')[7] + '_' +
                   os.path.basename(z_component_path[i]).split('.')[9] + '.SAC', format='SAC')
    # tr_slice_data.append(tr_slice)