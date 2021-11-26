# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 13-Jul-21 04:45 PM
@file: instanceTest.py
"""

import tkinter as tk
from tkinter import filedialog
from ioGpy import AutocorrTarget
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import plotly
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
# instance
targetInstance = AutocorrTarget()
# load .target file
targetInstance.load(file_path)
numRings = len(targetInstance.AutocorrCurves.ModalCurve)
# store SPAC data in a list
spacData = []
for i in range(numRings):
    spacData.append(targetInstance.AutocorrCurves.ModalCurve[i].RealStatisticalPoint)
# plot SPAC data which selected using Matplotlib
print('The number of rings is : ' + str(numRings))
numPlot = int(input('Please select the number of rings :'))
figureName = str(targetInstance.AutocorrCurves.AutocorrRing[numPlot-1])
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[numPlot-1])['x'],
                         y=pd.DataFrame(spacData[numPlot-1])['mean'] - pd.DataFrame(spacData[numPlot-1])['stddev'],
                         mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[numPlot-1])['x'],
                         y=pd.DataFrame(spacData[numPlot-1])['mean'] + pd.DataFrame(spacData[numPlot-1])['stddev'],
                         mode='lines', line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False))
fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[numPlot-1])['x'], y=pd.DataFrame(spacData[numPlot-1])['mean'],
                         mode='lines', name='SPAC_Mean'))
fig.add_trace(go.Scatter(x=pd.DataFrame(spacData[numPlot-1])['x'], y=pd.DataFrame(spacData[numPlot-1])['imag'],
                         mode='lines', name='SPAC_Imag'))
# set the path to store the .html files
print("Please set the path to store the .html files")
Folderpath = filedialog.askdirectory()
fig.update_xaxes(type="log")
fig.update_xaxes(range=[-0.3, 2])
# fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
fig.update_yaxes(range=[-1, 1], tick0=0.00, dtick=0.1)

fig.update_layout(
    # title=file_path.split('/')[-1],
    title="Rings=" + figureName + "m",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Autocorr ratio",
    legend_title="SPAC",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

# fig.update_traces(showlegend=False)
htmlFileName = file_path.split('/')[-1] + '_SPAC' + '.html'  # todo: filename split problem with '.'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
