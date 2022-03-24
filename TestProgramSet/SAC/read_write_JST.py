# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:52:23 2020

@author: Chenlj

This code is designed to read and Convert the data of JST.

"""

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from readJST import *
from obspy import read
import numpy as np
import os

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:',Folderpath)
filetype_dat = '.dat'
filetype_sac = '.sac'

def get_dat_filename(Folderpath,filetype_dat):
    name = []
    final_name = []
    for root,dirs,files in os.walk(Folderpath):
        for i in files:
            if filetype_dat in i:
                 name.append(i.replace(filetype_dat,''))
    final_name = [item + '.dat' for item in name]
    return final_name

FileName_dat = get_dat_filename(Folderpath,filetype_dat)
print('The names of Raw data are:','\n',FileName_dat)

num = len(FileName_dat)

st_sac = [0 for x in range(0,num)]
for i in range(num):
    st_sac[i] = 'st' + str(i) + '.sac'

for i in range(num):
    names_dat = locals()
    names_dat['st%s' % i] = read_JST_int(Folderpath + '/' + FileName_dat[i])
    file_path = Folderpath + '/' + st_sac[i]
    names_dat['st%s' % i].write(file_path, format='sac')

def get_sac_filename(Folderpath, filetype_sac):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype_sac in i:
                name.append(i.replace('.sac', ''))
    final_name = [item + '.sac' for item in name]
    return final_name

FileName_sac = get_sac_filename(Folderpath,filetype_sac)
print('The names of Single sac data are:','\n',FileName_sac)

st = read(Folderpath + '/' + FileName_sac[0])

for i in range(num - 1):
    st = st + read(Folderpath + '/' + FileName_sac[i + 1])
# sort
st.sort(['starttime'])
# start time in plot equals 0
dt = st[0].stats.starttime.timestamp
ax = plt.subplot(num, 1, 1)  # dummy for tying axis

for i in range(num - 1):
    plt.subplot(num, 1, i + 1, sharex=ax)
    t = np.linspace(st[i].stats.starttime.timestamp - dt, st[i].stats.endtime.timestamp - dt, st[i].stats.npts)
plt.plot(t, st[i].data)

# Merge the data together and show plot in a similar way
st.merge(method=1)
plt.subplot(num, 1, num, sharex=ax)
t = np.linspace(st[0].stats.starttime.timestamp - dt,
                st[0].stats.endtime.timestamp - dt,
                st[0].stats.npts)
plt.plot(t, st[0].data, 'r')
plt.show()

for tr in st:
    if isinstance(tr.data, np.ma.masked_array):
        tr.data = tr.data.filled()

#tr.data[tr.data > 999] = 0
#tr.data[tr.data < -999] = 0
#Name the Merge_Data
Data_merge_name = input('Please input the Synthetic data name:')
Data_merge_name = Data_merge_name + '.SAC'
print('The Synthetic data name is:','\n',Data_merge_name)
file_path_merge = Folderpath + '/' + Data_merge_name
tr.write(file_path_merge, format='sac')

judgment = input('Do you want to delete the single .sac data ?' '\n' '[Y/N]')
if judgment in ['y','Y']:
    for maindir, subdir, file_name_list in os.walk(Folderpath):
        for filename in file_name_list:
            if (filename.endswith(".sac")):
                os.remove(maindir + "\\" + filename)
            else:
                print("None")
else:
    print("Done")


