"""
Created on May 10,2021
@author: LijiongChen
This code is designed to read the spac data form cloud platform.
specially for 3 rings: gpy_spac[0].csv, gpy_spac[1].csv, gpy_spac[2].csv
"""

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
# read the spac data from 3 rings
spac_ring1 = pd.read_csv(Folderpath + '/' + 'gpy_spac[0]300cm.csv')
spac_ring1.columns = ['freq1', 'ring1real', 'ring1imag', 'ring1std', 'ring1up', 'ring1low']
spac_ring2 = pd.read_csv(Folderpath + '/' + 'gpy_spac[1]350cm.csv')
spac_ring2.columns = ['freq2', 'ring2real', 'ring2imag', 'ring2std', 'ring2up', 'ring2low']
spac_ring3 = pd.read_csv(Folderpath + '/' + 'gpy_spac[2]570cm.csv')
spac_ring3.columns = ['freq3', 'ring3real', 'ring3imag', 'ring3std', 'ring3up', 'ring3low']
# concat the data to one DataFrame format
spac = pd.concat([spac_ring1, spac_ring2, spac_ring3], axis=1)
# plot and analysis the data
fig = make_subplots(rows=1, cols=3, subplot_titles=('Ring 1', 'Ring 2', 'Ring 3'), x_title='Frequency (Hz)',
                    y_title='Autocorr ratio')
# ring1
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1real'], mode='lines', fill='tonexty',
                         name='Ring1_real'), row=1, col=1)
fig.add_trace(go.Scatter(x=spac['freq1'], y=spac['ring1low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=1)
# ring2
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2real'], mode='lines', fill='tonexty',
                         name='Ring2_real'), row=1, col=2)
fig.add_trace(go.Scatter(x=spac['freq2'], y=spac['ring2low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=2)
# ring3
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3up'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), name='upper bound'), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3real'], mode='lines', fill='tonexty',
                         name='Ring3_real'), row=1, col=3)
fig.add_trace(go.Scatter(x=spac['freq3'], y=spac['ring3low'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='lower bound'), row=1, col=3)
# design the frame properties
fig.update_xaxes(type="log")
fig.update_xaxes(range=[-0.3, 2])
# fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
fig.update_yaxes(range=[-0.6, 1], tick0=0.00, dtick=0.2)
fig.update_layout(title=Folderpath)
plotly.offline.plot(fig, filename=Folderpath + '/' + 'SPAC.html')
