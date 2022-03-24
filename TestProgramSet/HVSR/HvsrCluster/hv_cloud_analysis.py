"""
Created on 04-21-2021
Update on 05-14-2021
@author: LijiongChen
This code is designed to analysis H/V data from Geopsy & cloud platform
hv data: .hv
"""

import io
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)

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


def read_hv(file):
    with open(file, encoding='utf-8') as f0:
        tmp = f0.readlines()
    head = [x for x in tmp if x[0] == '#']
    tab = [x for x in tmp if x[0] != '#']
    keys = ['geo_version', 'num_window', 'f0', 'num_window_f0', 'position', 'category', 'tab_head']
    h0 = dict.fromkeys(keys)
    h0['geo_version'] = float(head[0].split(' ')[-1])
    h0['num_window'] = int(head[1].split('=')[-1])
    h0['f0'] = float(head[2].split('\t')[-1])
    h0['num_window_f0'] = int(head[3].split('=')[-1])
    h0['postion'] = [float(x) for x in head[6].split('\t')[-1].split(' ')]
    h0['category'] = head[7].split('\t')[-1][:-1]
    h0['tab_head'] = head[8][2:].split('\t')
    t0 = np.loadtxt(io.StringIO(''.join(tab)))
    return h0, t0


# read H/V data from the .hv files
headers1, st1 = read_hv(Folderpath + '/' + FileName[0])
headers2, st2 = read_hv(Folderpath + '/' + FileName[1])
headers3, st3 = read_hv(Folderpath + '/' + FileName[2])
headers4, st4 = read_hv(Folderpath + '/' + FileName[3])
headers5, st5 = read_hv(Folderpath + '/' + FileName[4])
headers6, st6 = read_hv(Folderpath + '/' + FileName[5])
# numpy array change into pandas dataframe
st1 = pd.DataFrame(st1)
st1.columns = ['f1', 'ave1', 'min1', 'max1']
st2 = pd.DataFrame(st2)
st2.columns = ['f2', 'ave2', 'min2', 'max2']
st3 = pd.DataFrame(st3)
st3.columns = ['f3', 'ave3', 'min3', 'max3']
st4 = pd.DataFrame(st4)
st4.columns = ['f4', 'ave4', 'min4', 'max4']
st5 = pd.DataFrame(st5)
st5.columns = ['f5', 'ave5', 'min5', 'max5']
st6 = pd.DataFrame(st6)
st6.columns = ['f6', 'ave6', 'min6', 'max6']
# concat into one
hv = pd.concat([st1, st2, st3, st4, st5, st6], axis=1)
# define the plotly subplot
fig = make_subplots(rows=3, cols=3,
                    specs=[[{"rowspan": 3, "colspan": 1}, {}, {}],
                           [None, {}, {}],
                           [None, {}, {}]],
                    print_grid=True,
                    subplot_titles=('H/V_all', 'H/V_1', 'H/V_2', 'H/V_3', 'H/V_4', 'H/V_5', 'H/V_6'), x_title='Frequency (Hz)',
                    y_title='Amplitude')
# plot the all station's H/V curve in one figure
fig.add_trace(go.Scatter(x=hv['f1'], y=hv['ave1'], mode='lines', marker_color='rgba(255, 151, 255, .8)',
                         name='H/V_1'), row=1, col=1)
fig.add_trace(go.Scatter(x=hv['f2'], y=hv['ave2'], mode='lines', marker_color='rgba(239, 85, 59, .8)',
                         name='H/V_2'), row=1, col=1)
fig.add_trace(go.Scatter(x=hv['f3'], y=hv['ave3'], mode='lines', marker_color='rgba(255, 161, 90, .8)',
                         name='H/V_3'), row=1, col=1)
fig.add_trace(go.Scatter(x=hv['f4'], y=hv['ave4'], mode='lines', marker_color='rgba(182, 232, 128, .8)',
                         name='H/V_4'), row=1, col=1)
fig.add_trace(go.Scatter(x=hv['f5'], y=hv['ave5'], mode='lines', marker_color='rgba(99, 110, 250, .8)',
                         name='H/V_5'), row=1, col=1)
fig.add_trace(go.Scatter(x=hv['f6'], y=hv['ave6'], mode='lines', marker_color='rgba(175, 107, 250, .8)',
                         name='H/V_6'), row=1, col=1)
# plot the single H/V curve in each figure
fig.add_trace(go.Scatter(x=hv['f1'], y=hv['max1'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=hv['f1'], y=hv['min1'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=hv['f1'], y=hv['ave1'], mode='lines',
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=hv['f2'], y=hv['max2'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=hv['f2'], y=hv['min2'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=hv['f2'], y=hv['ave2'], mode='lines',
                         showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=hv['f3'], y=hv['max3'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=hv['f3'], y=hv['min3'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=hv['f3'], y=hv['ave3'], mode='lines',
                         showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=hv['f4'], y=hv['max4'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(x=hv['f4'], y=hv['min4'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(x=hv['f4'], y=hv['ave4'], mode='lines',
                         showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(x=hv['f5'], y=hv['max5'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=hv['f5'], y=hv['min5'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=hv['f5'], y=hv['ave5'], mode='lines',
                         showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=hv['f6'], y=hv['max6'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=3, col=3)
fig.add_trace(go.Scatter(x=hv['f6'], y=hv['min6'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=3, col=3)
fig.add_trace(go.Scatter(x=hv['f6'], y=hv['ave6'], mode='lines',
                         showlegend=False), row=3, col=3)
# fig.add_trace(go.Scatter(x=hv['f1'], y=hv['ave1'], name='st1'), row=1, col=2)
# fig.add_trace(go.Scatter(x=hv['f1'], y=hv['min1'], mode="markers", marker=dict(size=1), name='st1'), row=1, col=2)
# fig.add_trace(go.Scatter(x=hv['f1'], y=hv['max1'], mode="markers", marker=dict(size=1), name='st1'), row=1, col=2)
# fig.add_trace(go.Scatter(x=hv['f2'], y=hv['ave2'], name='st2'), row=1, col=3)
# fig.add_trace(go.Scatter(x=hv['f2'], y=hv['min2'], mode="markers", marker=dict(size=1), name='st2'), row=1, col=3)
# fig.add_trace(go.Scatter(x=hv['f2'], y=hv['max2'], mode="markers", marker=dict(size=1), name='st2'), row=1, col=3)
# fig.add_trace(go.Scatter(x=hv['f3'], y=hv['ave3'], name='st3'), row=2, col=2)
# fig.add_trace(go.Scatter(x=hv['f3'], y=hv['min3'], mode="markers", marker=dict(size=1), name='st3'), row=2, col=2)
# fig.add_trace(go.Scatter(x=hv['f3'], y=hv['max3'], mode="markers", marker=dict(size=1), name='st3'), row=2, col=2)
# fig.add_trace(go.Scatter(x=hv['f4'], y=hv['ave4'], name='st4'), row=2, col=3)
# fig.add_trace(go.Scatter(x=hv['f4'], y=hv['min4'], mode="markers", marker=dict(size=1), name='st4'), row=2, col=3)
# fig.add_trace(go.Scatter(x=hv['f4'], y=hv['max4'], mode="markers", marker=dict(size=1), name='st4'), row=2, col=3)
# fig.add_trace(go.Scatter(x=hv['f5'], y=hv['ave5'], name='st5'), row=3, col=2)
# fig.add_trace(go.Scatter(x=hv['f5'], y=hv['min5'], mode="markers", marker=dict(size=1), name='st5'), row=3, col=2)
# fig.add_trace(go.Scatter(x=hv['f5'], y=hv['max5'], mode="markers", marker=dict(size=1), name='st5'), row=3, col=2)
# fig.add_trace(go.Scatter(x=hv['f6'], y=hv['ave6'], name='st6'), row=3, col=3)
# fig.add_trace(go.Scatter(x=hv['f6'], y=hv['min6'], mode="markers", marker=dict(size=1), name='st6'), row=3, col=3)
# fig.add_trace(go.Scatter(x=hv['f6'], y=hv['max6'], mode="markers", marker=dict(size=1), name='st6'), row=3, col=3)
# update the figure parameter
fig.update_xaxes(type="log")
fig.update_xaxes(range=[-0.3, 2])
fig.update_yaxes(range=[0, 10], tick0=0.00, dtick=2)
fig.update_layout(title=Folderpath.split("/")[-2])
htmlFileName = Folderpath.split("/")[-2] + '.html'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
