# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 4/1/2022 10:14 AM
@file: hvsr_view_folder.py
"""


import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from glob import glob
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import math


import colorama
colorama.init(autoreset=True)
print('\033[0;31mThe folder path example is :\033[0m')
print('HVSR\n'
         '--|A\n'
         '----|station\n'
         '------------|1.hv\n'
         '------------|2.hv\n'
         '------------|....hv\n'
         '--|B\n'
         '----|station\n'
         '------------|1.hv\n'
         '------------|2.hv\n'
         '------------|....hv\n'
         '--|C')
print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
print('Please select the HVSR file：')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;31mFolderPath\033[0m:', folder_path)
sub_path_name = os.listdir(folder_path)
sub_path = []
hvsr_file_path = []
for i in range(len(sub_path_name)):
    sub_path.append(folder_path + '/' + sub_path_name[i])
    hvsr_file_path.append(sub_path[i] + '/' + 'station')

for i in range(len(hvsr_file_path)):
    fs = glob(hvsr_file_path[i] + '/*.hv')
    hvsr_data = []
    max_y_value = 0
    for j in range(len(fs)):
        hvsr_data.append(pd.read_table(filepath_or_buffer=fs[j], sep='\t', skiprows=9,
                                       names=['Freq', 'Aver', 'max', 'min']))
        max_y_value = max(max_y_value, hvsr_data[j]['Aver'].loc[175:332].max())
    pio.templates.default = "plotly_white"  # set the plotly templates
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.001)
    for k in range(len(hvsr_data)):
        fig.add_trace(go.Scatter(x=hvsr_data[k]['Freq'], y=hvsr_data[k]['Aver'], mode='lines', name=Path(fs[k]).stem),
                      row=1, col=1)
    # update the figure parameter
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[math.log10(1), math.log10(100)])
    fig.update_yaxes(range=[0.00, int(max_y_value) + 1], tick0=0.00, dtick=int(max_y_value / 5))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(
        title='HVSR curves of folder ' + folder_path + '. ' + 'The number of HVSR curves is %d.' % len(fs),
        showlegend=True)
    html_file_Name = hvsr_file_path[i].split("/")[-2] + '.html'
    plotly.offline.plot(fig, filename=hvsr_file_path[i] + '/' + html_file_Name)
    print('\033[0;36mThe HVSR curves of %s have been plotted and saved.\033[0m' % hvsr_file_path[i].split("/")[-2])
print('\033[0;32m------------------------Done!------------------------\033[0m')
input('Press Enter to exit … \n')