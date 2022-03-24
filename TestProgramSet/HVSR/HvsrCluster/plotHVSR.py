"""
Created on Jun 23,2021
Update on Jun 23,2021
@author: LijiongChen
This code is designed to plot the all H/V data from Geopsy & cloud platform
hv data: .hv
"""

import os
import tkinter as tk
from tkinter import filedialog
from hvDataPlot import hvDataPlot

# choose the FilePath and the .hv DataFileName
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
filetype = '.hv'


def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.hv' for item in name]
    return final_name


FileName = get_filename(Folderpath, filetype)
print(FileName)
# plot the .hv data using plotly
numOfHVSR = len(FileName)

dirs = Folderpath + '/html'
if not os.path.exists(dirs):
    os.makedirs(dirs)
for i in range(numOfHVSR):
    hvDataPlot(Folderpath + '/' + FileName[i])
