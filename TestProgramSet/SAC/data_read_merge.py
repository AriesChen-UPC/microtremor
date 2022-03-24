# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:52:23 2019
@author: LijiongChen
This code is designed to merge the .sac data.
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from obspy import read
import matplotlib.pyplot as plt
from obspy.signal.detrend import polynomial


root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
# Filepath = filedialog.askopenfilename()
print('Folderpath:', Folderpath)
# print('Filepath:',Filepath)
filetype = 'N.sac'


def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + 'N.sac' for item in name]
    return final_name

FileName = get_filename(Folderpath,filetype)
print(FileName)
Data_Number = len(FileName)
st = read(Folderpath + '/' + FileName[0])
for i in range(Data_Number-1):
    st = st + read(Folderpath + '/' + FileName[i+1])
# sort
st.sort(['starttime'])
# start time in plot equals 0
dt = st[0].stats.starttime.timestamp
# Go through the stream object, determine time range in julian seconds
# and plot the data with a shared x axis
# ax = plt.subplot(Data_Number, 1, 1) # dummy for tying axis
# for i in range(Data_Number-1):
#     plt.subplot(Data_Number, 1, i + 1, sharex = ax)
#     t = np.linspace(st[i].stats.starttime.timestamp - dt,st[i].stats.endtime.timestamp - dt,st[i].stats.npts)
# plt.plot(t, st[i].data)

# Merge the data together and show plot in a similar way
st.merge(method=1)
# plt.subplot(Data_Number, 1, Data_Number, sharex = ax)
t = np.linspace(st[0].stats.starttime.timestamp - dt,
st[0].stats.endtime.timestamp - dt,
st[0].stats.npts)
# plt.plot(t, st[0].data, 'r')
# plt.show()
for tr in st:
    if isinstance(tr.data, np.ma.masked_array):
        tr.data = tr.data.filled()
# polynomial(tr.data, order=3, plot=True)  # detrend
tr.data[tr.data >= 1e+20] = 0
# tr.data[tr.data < -9999] = 0
# Name the Merge_Data
Data_merge_name = input('Please input the Filename:')
Data_merge_name = Data_merge_name + '.sac'
file_path = Folderpath + '/' + Data_merge_name
tr.write(file_path, format='sac')
