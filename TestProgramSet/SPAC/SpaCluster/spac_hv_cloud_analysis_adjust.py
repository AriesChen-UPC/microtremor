"""
Created on May 10,2021
Update on May 17,2021
@author: LijiongChen
This code is designead to read the spac&hv data form cloud platform(core:Geopsy)
    read all rings' data and specially plot 3 rings curve: gpy_spac[0].csv, gpy_spac[1].csv, gpy_spac[2].csv
    read hv data and specially plot the avg hvsr curve
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
from termcolor import colored

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print(colored('Folderpath:', 'green', attrs=['bold']))
print(Folderpath)
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
print(colored('CSV data file:', 'green', attrs=['bold']))
print(FileName)
SpecifyFileName = []
for i in range(len(FileName)):
    if "gpy_spac[" in FileName[i]:  # "gpy_spac[" is the character
        SpecifyFileName.append(FileName[i])
print(colored('SPAC file:', 'green', attrs=['bold']))
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

# .hv data get
hvNames = os.listdir(Folderpath)
hvGet = []
for hvName in hvNames:
    if os.path.splitext(hvName)[1] == '.hv':
        hvGet.append(hvName)  # todo: if hv is't the avg data
        # print(hvName)       # fixme: calculate the avg hv data
print(colored('HVSR file:', 'green', attrs=['bold']))
print(hvGet)

print(colored("SPAC curve's information: ", 'green', attrs=['bold']))
print("There are %d SPAC rings data." % len(SpecifyFileName))
print(colored("HVSR curve's information: ", 'green', attrs=['bold']))


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
    # return h0, t0
    return t0  # todo: don't output the headers


# judge the number of .hv data, calculate the avg .hv data
if len(hvGet) == 0:
    print("There is't any HVSR data.")
elif len(hvGet) == 1:
    print("There is only one HVSR data.")
    hv_avg = read_hv(Folderpath + '/' + hvGet[0])
else:
    print("There are %d HVSR data." % len(hvGet))  # print str and number
    hvsr = read_hv(Folderpath + '/' + hvGet[0])
    for i in range(1, len(hvGet)):
        hvsr += read_hv(Folderpath + '/' + hvGet[i])
    hv_avg = hvsr / len(hvGet)

# hvsr = read_hv(Folderpath + '/' + hvGet[0])
# for i in range(1, len(hvGet)):
#     hvsr += read_hv(Folderpath + '/' + hvGet[i])
# hv_avg = hvsr / 4;
# hv_avg = read_hv(Folderpath + '/' + hvGet[0])
# headers_avg, hv_avg = read_hv(Folderpath + '/' + hvGet[0])
hv_avg = pd.DataFrame(hv_avg)
hv_avg.columns = ['f1', 'ave1', 'min1', 'max1']
# make fig
fig = make_subplots(rows=1, cols=3, subplot_titles=(ring1_title, ring2_title, ring3_title),
                   specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
                    horizontal_spacing=0.063, x_title='Frequency (Hz)')
# ring1 & hvsr
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1low'], mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1real'], mode='lines',
                         name='SPAC_Ring1'), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['max1'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['min1'], mode='lines',line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['ave1'], mode='lines',
                         name='HVSR'), row=1, col=1, secondary_y=True)
# ring2 & hvsr
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2low'], mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2real'], mode='lines',
                         name='SPAC_Ring2'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['max1'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['min1'], mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['ave1'], mode='lines',
                         name='HVSR'), row=1, col=2, secondary_y=True)
# ring3 & hvsr
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3low'], mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3real'], mode='lines',
                         name='SPAC_Ring3'), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['max1'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['min1'], mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         fill='tonexty', showlegend=False), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=hv_avg['f1'], y=hv_avg['ave1'], mode='lines',
                         name='HVSR'), row=1, col=3, secondary_y=True)
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
fig.update_yaxes(range=[-0.6, 1], tick0=0.00, dtick=0.2, secondary_y=False)
fig.update_yaxes(range=[0, 10], tick0=0.00, dtick=2.0, secondary_y=True)
# fig update：title, xaxes, yaxes...
titleName = Folderpath.split("/")[-1]
fig.update_layout(title=titleName)
# fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_yaxes(
        title_text="Autocorr ratio",
        secondary_y=False)
fig.update_yaxes(
        title_text="Amplitude",
        secondary_y=True)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=1.02
))
# save .html file
htmlFileName = Folderpath.split("/")[-1] + '_SPAC_HVSR' + '.html'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
# # plot gpy2disp
# z_data = pd.read_csv(Folderpath + '/' + 'gpy_spac2disp.csv')
# fig2disp = go.Figure(data=[go.Contour(z=z_data.values)])
# plotly.offline.plot(fig2disp, filename=Folderpath + '/' + Folderpath.split("/")[-1] + '_spac2disp.html')
