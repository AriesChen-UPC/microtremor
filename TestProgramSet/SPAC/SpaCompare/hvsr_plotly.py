# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/26/2022 5:40 PM
@file: hvsr_plotly.py
"""

import os
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


def hvsr_plotly(hvsr_all, point_name, period_hvsr_label, folder_path):
    """
    Args:
        freq_all: the list of frequency
        spac_all: the list of spac
        ring_name: the list of ring
        point_name: the list of point name
        period_hvsr_label: the list of period name

    Returns:

    """
    pio.templates.default = "plotly_white"  # set the plotly templates
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                  '#7f7f7f', '#bcbd22', '#17becf']
    fig = make_subplots(rows=1, cols=1)
    # plot the hvsr
    max_all = 0
    for m in range(len(hvsr_all)):  # FIXME: When len(hvsr_all)=1, the display of the name is not normal
        fig.add_trace(go.Scatter(x=hvsr_all[m]['freq'], y=hvsr_all[m]['avg'], name=period_hvsr_label[m],
                                 line=dict(
                                         color=color_list[m]
                                 )), row=1, col=1)
        hvsr_max = np.max(hvsr_all[m]['avg'].iloc[175:332])  # TODO: the range of hvsr in frequency [2, 30]
        max_all = max_all if max_all > hvsr_max else hvsr_max

    fig.update_xaxes(type="log", range=[np.log10(1), np.log10(100)])
    fig.update_yaxes(range=[0, int(max_all) + 1], tick0=0.0, dtick=int(max_all/5))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="H/V")
    fig.update_layout(title=point_name)
    fig.update_layout(
        showlegend=False,
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    fig_path = os.path.dirname(folder_path) + '/Compare' + '/hvCompare'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = fig_path + '/' + point_name + '.html'
    plotly.offline.plot(fig, filename=fig_name, auto_open=False)
