# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time:19/06/2021
@update:06/11/2021
@file: hv_cluster.py
       This code is designead to sort the HVSR using keans and plot the result with plotly.
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, vq
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import time
from tqdm import trange

# Load HV
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
root.destroy()
# find the .hv data
fs = glob(Folderpath + '/*.hv')
# define the range of Freq
freq = np.geomspace(0.1, 100.1, 400)
# define the min and max Freq you wanted
min_freq = float(input('Min_freq:\n'))
max_freq = float(input('Max_freq:\n'))
# read the HVSR data
sel = (freq >= min_freq) & (freq <= max_freq)
freq_sel = freq[sel]
data = []
data_sel = np.zeros(shape=(len(fs), len(freq_sel)))
data2 = np.zeros(shape=(len(fs), len(freq)))
print('\033[0;31m----------------------Loading...----------------------\033[0m')
for i in trange(len(fs)):
    time.sleep(0.1)
    # here skiprows=9 is the headers of .hv data
    data.append(pd.read_table(filepath_or_buffer=fs[i], sep='\t', skiprows=9, names=['Freq', 'Aver', 'max', 'min']))
    data2[i] = np.interp(freq, data[i].Freq, data[i].Aver)  # interpolation for the same data dimension
    data_sel[i] = data2[i][sel]
# change numpy array to pandas dataframe
data2 = pd.DataFrame(data2).T
# get the name of dataframe.columns
hvname = [f'hv{n}' for n in range(len(fs))]
# change the dataframe.columns name to what you create
data2.columns = hvname
# change numpy array to pandas dataframe
freq = pd.DataFrame(freq)
# change the dataframe.columns name to what you want
freq.columns = ['freq']
# get the .hv file name
names = []
for i in range(len(fs)):
    names.append(Path(fs[i]).stem)
# define the number which you want to sort
n_clusters = int(input('Please input the number of clusters:\n'))
centroid, _ = kmeans(data_sel, n_clusters)
result1, _ = vq(data_sel, centroid)
result1 = result1.tolist()
print(result1)
# define the plotly subplot
pio.templates.default = "plotly_white"  # set the plotly templates
if n_clusters == 1:
    fig = make_subplots(rows=1, cols=1, print_grid=True)
    count = 0
    for k in range(len(fs)):
        count = count + 1
        fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                      row=1, col=1)
elif n_clusters == 2:
    fig = make_subplots(rows=1, cols=2, print_grid=True)
    count = 0
    for k in range(len(fs)):
        count = count + 1
        if result1[k] == 0:
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=2)
elif n_clusters == 3:
    fig = make_subplots(rows=1, cols=3, print_grid=True)
    count = 0
    for k in range(len(fs)):
        count = count + 1
        if result1[k] == 0:
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=1)
        elif result1[k] == 1:
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=3)
else:
    numRow = 2
    numCol = int(np.ceil(n_clusters / 2))
    fig = make_subplots(rows=numRow, cols=numCol, print_grid=True)
    # plot the result of classification
    for z in range(n_clusters):  # z is the number of classification
        count = 0
        for k in range(len(fs)):
            if result1[k] == z:
                count = count + 1
                if z < numCol:
                    fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                                  row=1, col=z+1)
                else:
                    fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                                  row=2, col=z+1-numCol)
# update the plot frame
fig.update_xaxes(type="log")
fig.update_xaxes(range=[0.3, 2])
fig.update_yaxes(range=[0.0, 7.0], tick0=0.0, dtick=1.0)
fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_yaxes(title_text="Amplitude")
fig.update_layout(title='HV cluster', showlegend=False)
# print the Classification result of HVSR
for z in range(n_clusters):
    print('Group {:.0f} '.format(z + 1))
    for i in range(len(fs)):
        if result1[i] == z:
            print(names[i])
# save the .html file
htmlFileName = Folderpath.split("/")[-1] + '_HV_Cluster' + '.html'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
print('\033[0;32m----------------------Done!----------------------\033[0m')

