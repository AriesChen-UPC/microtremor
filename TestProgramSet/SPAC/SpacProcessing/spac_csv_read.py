"""
Created on 2020.12.22-13:09:12
@author: LijiongChen
This code is designed to read the .csv data.
.csv data:
          col1: Frequency
          col2: SPAC
          col3: Standard deviation
"""

import pandas as pd
import glob
import os
from tkinter import filedialog
import tkinter as tk
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plotly.offline.init_notebook_mode(connected=True)

root = tk.Tk()
root.withdraw()
path = filedialog.askdirectory()

file = glob.glob(os.path.join(path, "*.csv"))
print(file)
dl = []
for f in file:
    dl.append(pd.read_csv(f))

fig = make_subplots(rows=1, cols=1, x_title='Frequency (Hz)', y_title='Autocorr ratio')

for i in range(len(file)):
    fig.add_trace(go.Scatter(x=dl[i]['x'], y=dl[i]['y'], name=os.path.splitext(os.path.basename(file[i]))[0],
                             legendgroup=os.path.splitext(os.path.basename(file[i]))[0]), row=1, col=1)

fig.update_layout(title='XiAn', height=800, width=1000, margin=dict(r=100, l=100, b=100, t=100),
                  font=dict(size=20))
fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=10)
fig.update_yaxes(range=[-0.2, 1], tick0=0.00, dtick=0.2)
plotly.offline.plot(fig)
