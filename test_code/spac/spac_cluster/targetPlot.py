# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 18/10/2021 下午5:09
@file: targetPlot.py
"""
import os
import tkinter as tk
from tkinter import filedialog
from ioGpy import AutocorrTarget
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)


root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
filetype = '.target'

# get the FileName
def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.target' for item in name]
    return final_name

# get the specify FileName
FileName = get_filename(Folderpath, filetype)
print(FileName)

# instance
targetInstance = AutocorrTarget()
# load .target file
spacData = []
for i in range(len(FileName)):
    targetInstance.load(Folderpath + '/' + FileName[i])
    numRings = len(targetInstance.AutocorrCurves.ModalCurve)
    for j in range(numRings):
        spacData.append(targetInstance.AutocorrCurves.ModalCurve[j].RealStatisticalPoint)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Ring 1', 'Ring 2'), x_title='Frequency (Hz)',
                    y_title='Autocorr ratio')
for i in range(len(FileName)):
    fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[2 * i])['x'], y=pd.DataFrame(spacData[2 * i])['mean'],
                             mode='lines', name=FileName[i][0:]), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[2 * i + 1])['x'], y=pd.DataFrame(spacData[2 * i + 1])['mean'],
                             mode='lines', name=FileName[i], showlegend=False), row=1, col=2)
print("Please set the path to store the .html files")
Folderpath = filedialog.askdirectory()
fig.update_xaxes(type="log")
fig.update_xaxes(range=[0, 2])
# fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
fig.update_yaxes(range=[-1, 1], tick0=0.00, dtick=0.1)
plotly.offline.plot(fig, filename=Folderpath + '/' + 'SPAC.html')
