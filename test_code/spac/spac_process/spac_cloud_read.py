"""
Created on 2020.12.22-13:09:12
@author: LijiongChen
This code is designed to read the spac data.
spac data: psd、spac_imag.csv、spac_real.csv、 spac_std.csv
"""

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
# Filepath = filedialog.askopenfilename()

print('Folderpath:', Folderpath)
# print('Filepath:',Filepath)

filetype = '.csv'


def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.csv' for item in name]
    return final_name


FileName = get_filename(Folderpath, filetype)
print(FileName)

# spac_real plot
# spac_real = pd.read_csv(Folderpath + '/' +FileName[2])
# spac_real_fig = spac_real.plot(subplots = True, layout = [1,3], x = 'Frequency', xlim = [0,100], ylim = [-0.6,1],
#                     grid = True)

spac_real = pd.read_csv(Folderpath + '/' + FileName[2])
spac_real.columns = ['freqreal', 'ring1real', 'ring2real', 'ring3real']
# ring1_real = spac_real[['freqreal', 'ring1real']]
# ring2_real = spac_real[['freqreal', 'ring2real']]
# ring3_real = spac_real[['freqreal', 'ring3real']]

spac_imag = pd.read_csv(Folderpath + '/' +FileName[1])
spac_imag.columns = ['freqimag', 'ring1imag', 'ring2imag', 'ring3imag']
# ring1_imag = spac_imag[['freqimag', 'ring1imag']]
# ring2_imag = spac_imag[['freqimag', 'ring2imag']]
# ring3_imag = spac_imag[['freqimag', 'ring3imag']]

spac_std = pd.read_csv(Folderpath + '/' +FileName[3])
spac_std.columns = ['freqstd', 'ring1std', 'ring2std', 'ring3std']
# ring1_std = spac_imag[['freqstd', 'ring1std']]
# ring2_std = spac_imag[['freqstd', 'ring2std']]
# ring3_std = spac_imag[['freqstd', 'ring3std']]

spac = pd.concat([spac_real, spac_imag, spac_std], axis=1)

fig = make_subplots(rows=1, cols=3, subplot_titles=('Ring 1', 'Ring 2', 'Ring 3'), x_title='Frequency (Hz)',
                    y_title='Autocorr ratio')

fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring1real']+spac['ring1std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring1real'], mode='lines', fill='tonexty',
                         name='Ring1_real'), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring1real']-spac['ring1std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=1)

fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring2real']+spac['ring2std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring2real'], mode='lines', fill='tonexty',
                         name='Ring2_real'), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring2real']-spac['ring2std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=2)

fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring3real']+spac['ring3std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring3real'], mode='lines', fill='tonexty',
                         name='Ring3_real'), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freqreal'], y=spac['ring3real']-spac['ring3std'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=3)

fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
fig.update_yaxes(range=[-0.6, 1], tick0=0.00, dtick=0.2)
fig.update_layout(title='SPAC')
plotly.offline.plot(fig)


# fig = make_subplots(rows=1, cols=3, subplot_titles=('Ring 1', 'Ring 2', 'Ring 3'))
# fig.add_trace(go.Scatter(x=ring1_real["freq"], y=ring1_real['ring1'], name='Ring1_real', legendgroup='Ring1_real'),
#               row=1, col=1)
# fig.add_trace(go.Scatter(x=ring1_imag["freq"], y=ring1_imag['ring1'], name='Ring1_imag', legendgroup='Ring1_imag'),
#               row=1, col=1)
# fig.add_trace(go.Scatter(x=ring2_real["freq"], y=ring2_real['ring2'], name='Ring2_real', legendgroup='Ring2_real'),
#               row=1, col=2)
# fig.add_trace(go.Scatter(x=ring2_imag["freq"], y=ring2_imag['ring2'], name='Ring2_imag', legendgroup='Ring2_imag'),
#               row=1, col=2)
# fig.add_trace(go.Scatter(x=ring3_real["freq"], y=ring3_real['ring3'], name='Ring3_real', legendgroup='Ring3_real'),
#               row=1, col=3)
# fig.add_trace(go.Scatter(x=ring3_imag["freq"], y=ring3_imag['ring3'], name='Ring3_imag', legendgroup='Ring3_imag'),
#               row=1, col=3)
#
# plotly.offline.plot(fig)
