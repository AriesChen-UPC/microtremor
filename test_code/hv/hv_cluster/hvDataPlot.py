"""
Created on Jun 18,2021
Update on Jun 18,2021
@author: LijiongChen
This code is designed to plot the H/V data from Geopsy & cloud platform
hv data: .hv
"""

def hvDataPlot(file_path):
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


    # root = tk.Tk()
    # root.withdraw()
    # # choose the hv data
    # file_path = filedialog.askopenfilename()


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
    headers, st = read_hv(file_path)
    st = pd.DataFrame(st)
    st.columns = ['f', 'avg', 'min', 'max']

    # Add traces
    fig = make_subplots(rows=3, cols=2,
                        specs=[[{"rowspan": 3, "colspan": 1}, {}],
                               [None, {}],
                               [None, {}]],
                        print_grid=True,
                        subplot_titles=('HVSR', 'Max', 'Avg', 'Min'))
    # plot hv data of Max,Avg,Min in one figure
    fig.add_trace(go.Scatter(x=st['f'], y=st['max'], mode='lines',
                             line=dict(color='rgba(255,255,255,0)'), name='Max'), row=1, col=1)
    fig.add_trace(go.Scatter(x=st['f'], y=st['min'], mode='lines',
                             line=dict(color='rgba(255,255,255,0)'), fill='tonexty', name='Min'), row=1, col=1)
    fig.add_trace(go.Scatter(x=st['f'], y=st['avg'], mode='lines',
                             name='Avg'), row=1, col=1)
    # plot hv data of Max,Avg,Min in each figure
    fig.add_trace(go.Scatter(x=st['f'], y=st['max'], mode='lines', marker_color='rgba(255, 151, 255, .8)',
                             name='Max'), row=1, col=2)
    fig.add_trace(go.Scatter(x=st['f'], y=st['avg'], mode='lines', marker_color='rgba(255, 161, 90, .8)',
                             name='Avg'), row=2, col=2)
    fig.add_trace(go.Scatter(x=st['f'], y=st['min'], mode='lines', marker_color='rgba(99, 110, 250, .8)',
                             name='Min'), row=3, col=2)
    # update the figure parameter
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[-0.3, 2])
    fig.update_yaxes(range=[0, 3], tick0=0.00, dtick=0.5)
    fig.update_layout(title=file_path.split("/")[-1].split(".")[-2], showlegend=False)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude")
    filepath, tempfilename = os.path.split(file_path)
    plotly.offline.plot(fig, filename=filepath + '/' + 'html' + '/' + file_path.split("/")[-1].split(".")[-2] + '.html')
