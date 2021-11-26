"""
Created on May 10,2021
Update on May 15,2021
@author: LijiongChen
This code is designead to red the spac data form cloud platform.
read all rings data and specially plot 3 rings: gpy_spac[0].csv, gpy_spac[1].csv, gpy_spac[2].csv
"""

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)
import plotly.io as pio

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
filetype = '.csv'


# get the FileName
def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.csv' for item in name]
    return final_name


# get the specify FileName
FileName = get_filename(Folderpath, filetype)
print(FileName)
SpecifyFileName = []
for i in range(len(FileName)):
    if "gpy_spac[" in FileName[i]:  # "gpy_spac[" is the character
        SpecifyFileName.append(FileName[i])
print(SpecifyFileName)
# # define the dataFileName
# dataFileName = []
# for i in range(len(SpecifyFileName)):
#     dataFileName.append('spac_ring' + str(i))
#     locals()['spac_ring'+str(i)] = pd.read_csv(Folderpath + '/' + SpecifyFileName[i])
# read the spac data from 3 rings
spac_ring1 = pd.read_csv(Folderpath + '/' + SpecifyFileName[0])
spac_ring1.columns = ['freq1', 'ring1real', 'ring1imag', 'ring1std', 'ring1up', 'ring1low']
spac_ring2 = pd.read_csv(Folderpath + '/' + SpecifyFileName[1])
spac_ring2.columns = ['freq2', 'ring2real', 'ring2imag', 'ring2std', 'ring2up', 'ring2low']
spac_ring3 = pd.read_csv(Folderpath + '/' + SpecifyFileName[2])
spac_ring3.columns = ['freq3', 'ring3real', 'ring3imag', 'ring3std', 'ring3up', 'ring3low']
# concat the data to one DataFrame format
spac = pd.concat([spac_ring1, spac_ring2, spac_ring3], axis=1)
# plot and analysis the data
(ring1_title, extension1) = os.path.splitext(SpecifyFileName[0])
(ring2_title, extension2) = os.path.splitext(SpecifyFileName[1])
(ring3_title, extension3) = os.path.splitext(SpecifyFileName[2])

fig = make_subplots(rows=1, cols=3, subplot_titles=(ring1_title, ring2_title, ring3_title))
# ring1
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=1)
# fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1real'], mode='lines',
#                          name='SPAC_Ring1'), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1real'], mode='lines',
                         showlegend=False), row=1, col=1)
# ring2
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2real'], mode='lines',
                         showlegend=False), row=1, col=2)
# ring3
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3real'], mode='lines',
                         showlegend=False), row=1, col=3)
# design the frame properties todo: Optional frame properties
# print("Whether to use log axis? Please type Y or N")
# axisChoose = input()
# if (axisChoose == 'Y') | (axisChoose == 'y'):
#     fig.update_xaxes(type="log")
# # fig.update_xaxes(type="log")
# print("Please input the min of xaxis:")
# xaxisMin = float(input());
# print("Please input the max of xaxis:")
# xaxisMax = float(input());
# print("Please input the min of yaxis:")
# yaxisMin = float(input());
# print("Please input the max of yaxis:")
# yaxisMax = float(input());
# fig.update_xaxes(range=[xaxisMin, xaxisMax])
# # fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
# fig.update_yaxes(range=[yaxisMin, yaxisMax], tick0=0.00, dtick=0.2)
# fig.update_layout(title=Folderpath.split("/")[-1])
# # plotly.offline.plot(fig, filename=Folderpath + '/' + 'SPAC.html')
# htmlFileName = Folderpath.split("/")[-1] + '.html'
# plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
# define a log frame properties todo: a specify frame properties
fig.update_xaxes(type="log")
fig.update_xaxes(range=[0, 2])
# fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
fig.update_yaxes(range=[-0.6, 1], tick0=0.00, dtick=0.2)
# fig.update_layout(title=Folderpath.split("/")[-1])
fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_yaxes(title_text="Autocorr ratio")
fig.update_layout(template="plotly_white")  # set background to white
# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Black')
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Black')
# plotly.offline.plot(fig, filename=Folderpath + '/' + 'SPAC.html')
htmlFileName = Folderpath.split("/")[-1] + '_SPAC' + '.html'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
# save picture of .png format
if not os.path.exists("images"):
    os.mkdir("images")
pictureName = Folderpath.split("/")[-1] + ".png"
fig.write_image("images" + "/" + pictureName, width=1920, height=1080, scale=5)
