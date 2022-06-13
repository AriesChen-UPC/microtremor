# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/26/2022 1:14 PM
@file: spac_plotly.py
"""

import os
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


def spac_plotly(freq_all, spac_all, ring_name, point_name, period_spac_label, folder_path,
                  is_cps=False, theory_freq=None, theory_spac=None):
    """
    Args:
        freq_all: the list of frequency
        spac_all: the list of spac
        ring_name: the list of ring
        point_name: the list of point name
        period_spac_label: the list of period name

    Returns:

    """
    pio.templates.default = "plotly_white"  # set the plotly templates
    column_num = len(spac_all[0])
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                  '#7f7f7f', '#bcbd22', '#17becf']
    fig = make_subplots(rows=1, cols=column_num, subplot_titles=['ring=%.1fm' % i for i in ring_name])
    # plot the spac
    if is_cps:
        for i in range(column_num):
            fig.add_trace(go.Scatter(x=theory_freq[i]['freq'], y=theory_spac[i]['autoCorrRatio'], name='theory SPAC',
                                     line=dict(color='rgb(132,133,135)', width=2, dash='dash')), row=1, col=i + 1)
            for j in range(len(spac_all)):
                fig.add_trace(go.Scatter(x=freq_all[j][i], y=spac_all[j][i], name=period_spac_label[j],
                                         line=dict(
                                             color=color_list[j]
                                         )), row=1, col=i + 1)
    else:
        for i in range(column_num):
            for j in range(len(spac_all)):
                fig.add_trace(go.Scatter(x=freq_all[j][i], y=spac_all[j][i], name=period_spac_label[j],
                                         line=dict(
                                                 color=color_list[j]
                                         )), row=1, col=i + 1)

    fig.update_xaxes(type="log", range=[np.log10(1), np.log10(100)])
    fig.update_yaxes(range=[-0.6, 1.0], tick0=0.0, dtick=0.2)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Autocorr ratio")
    fig.update_layout(title=point_name)
    fig.update_layout(
        showlegend=False,
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    fig_path = os.path.dirname(folder_path) + '/Compare' + '/spaCompare'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = fig_path + '/' + point_name + '.html'
    plotly.offline.plot(fig, filename=fig_name, auto_open=False)
