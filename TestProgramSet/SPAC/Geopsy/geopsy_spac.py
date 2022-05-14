"""
This code is designed to calculate the theory Disp、SPAC (Geopsy).
tools: gpdc(Geopsy)
       gpspac(Geopsy)
Created on 2020.12.28
"""

import numpy as np
import pandas as pd
import tkinter
from tkinter import filedialog
from scipy.special import j0
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from TestProgramSet.SPAC.Geopsy.gpdc import gpdc
plotly.offline.init_notebook_mode(connected=True)


r = float(input('Please set the ring of the SPAC:'))
# choose .model
root = tkinter.Tk()
root.withdraw()
# print('Please set the work path:\n')
# model_path = filedialog.askdirectory()
# # define the model
# model_param = np.array([
#     (6, 0, 0, 0),  # todo: the model format is different from the model from Geopsy
#     (2, 500, 200, 1700),
#     (7, 800, 400, 1900),
#     (3, 900, 450, 1900),
#     (4, 1200, 700, 2000),
#     (8, 1700, 800, 2300),
#     (0, 2000, 1000, 2500)
# ])
# np.savetxt(model_path + '/' + 'model.txt', model_param, fmt='%d %d %d %d')

print('Please select the model file:')
model = filedialog.askopenfilename()
# set the gpdc path, gpdc is the tool of Geopsy
command = 'D:/MyProject/Geopsy/geopsypack-win64-3.4.2/bin/gpdc.exe'
R_matrix_F, L_matrix_F, R_matrix_S, L_matrix_S = gpdc(command, model)
R_matrix_F = pd.DataFrame(R_matrix_F).T
L_matrix_F = pd.DataFrame(L_matrix_F).T
R_matrix_S = pd.DataFrame(R_matrix_S).T
L_matrix_S = pd.DataFrame(L_matrix_S).T

# print('Input 最小频率')
# plt_min = float(input())
plt_min = 0.1
# print('Input 最大频率')
# plt_max = float(input())
plt_max = 100.1
# print('展示Mode个数')
# mode_fig = int(input())
mode_fig = 3

fig = make_subplots(rows=2, cols=2, subplot_titles=("Rayleigh Disp", "Love Disp", "Rayleigh SPAC", "Love SPAC"))

for i in range(0, mode_fig):
    fig.add_trace(go.Scatter(x=R_matrix_F[i], y=1/R_matrix_S[i], name='Rayleigh Disp'), row=1, col=1)

for i in range(0, mode_fig):
    fig.add_trace(go.Scatter(x=L_matrix_F[i], y=1/L_matrix_S[i], name='Love Disp'), row=1, col=2)

for i in range(0, mode_fig):
    spac_R = j0(r*2*np.pi*R_matrix_F[0]*R_matrix_S[0])
    fig.add_trace(go.Scatter(x=R_matrix_F[0], y=spac_R, name='Rayleigh SPAC'), row=2, col=1)

for i in range(0, mode_fig):
    spac_L = j0(r*2*np.pi*L_matrix_F[0]*L_matrix_S[0])
    fig.add_trace(go.Scatter(x=L_matrix_F[0], y=spac_L, name='Love SPAC'), row=2, col=2)

fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=10)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)

fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
fig.update_yaxes(title_text="Autocorr ratio", row=2, col=1)
fig.update_yaxes(title_text="Autocorr ratio", row=2, col=2)

plotly.offline.plot(fig)
