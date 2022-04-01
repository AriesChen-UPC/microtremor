# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 4/1/2022 8:37 AM
@file: hv_view_adjust.py
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
from tqdm import trange
import math


print('Please select the HVSR file：')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
print('\033[0;31mFolderPath\033[0m:', folder_path)
root.destroy()
# file type ifo in the selected file path
all_files = os.listdir(folder_path)
type_dict = dict()
for each_file in all_files:
    if os.path.isdir(each_file):
        type_dict.setdefault('Folders', 0)
        type_dict['Folders'] += 1
    else:
        ext = os.path.splitext(each_file)[1]
        type_dict.setdefault(ext, 0)
        type_dict[ext] += 1
for each_type in type_dict.keys():
    print('\033[0;34mFileType\033[0m: This folder has a total of \033[0;36m[%s]\033[0m file %d '
          % (each_type, type_dict[each_type]))  # highlight the filetype
print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
# find the .hv data
fs = glob(folder_path + '/*.hv')
data = []
max_y_value = 0
print('\033[0;31m----------------------Loading...---------------------\033[0m')
for i in trange(len(fs)):
    # here skiprows=9 is the headers of .hv data
    # todo: skiprows=9 is not the best way to skip the headers, try to skip the headers with # in the first line
    data.append(pd.read_table(filepath_or_buffer=fs[i], sep='\t', skiprows=9, names=['Freq', 'Aver', 'max', 'min']))
    max_y_value = max(max_y_value, data[i]['Aver'].loc[175:332].max())  # find the max value of the HVSR from freq[2,30]
print('\033[0;32m-----------------------Loaded!-----------------------\033[0m')
pio.templates.default = "plotly_white"  # set the plotly templates
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.001)
for i in range(len(data)):
    fig.add_trace(go.Scatter(x=data[i]['Freq'], y=data[i]['Aver'], mode='lines', name=Path(fs[i]).stem), row=1, col=1)
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
fig.update_layout(title='HVSR curves of folder ' + folder_path + '. ' + 'The number of HVSR curves is %d.' % len(fs),
                  showlegend=True)
html_file_Name = folder_path.split("/")[-1] + '.html'
plotly.offline.plot(fig, filename=folder_path + '/' + html_file_Name)
print('\033[0;36mThe HVSR curves have been plotted and saved.\033[0m')
print('\033[0;32m------------------------Done!------------------------\033[0m')
