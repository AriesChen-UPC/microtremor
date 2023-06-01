# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 11/23/2022 1:53 PM
@File     : hv_kmeans_clustering_plotly.py
@Software : PyCharm
"""

import math
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


def hv_kmeans_clustering_plotly(folder_path, fs, hvsr_freq, hvsr_data, data_sel, names, kmeans_clustering, hv_cluster_num):

    pio.templates.default = "plotly_white"  # set the plotly templates
    if hv_cluster_num == 1:
        fig = make_subplots(rows=1, cols=1, subplot_titles="Group 1")
        count = 0
        for k in range(len(fs)):
            count = count + 1
            fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                          row=1, col=1)
    elif hv_cluster_num == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Group 1", "Group 2"))
        count = 0
        for k in range(len(fs)):
            count = count + 1
            if kmeans_clustering.labels_[k] == 0:
                fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                              row=1, col=2)
    elif hv_cluster_num == 3:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Group 1", "Group 2", "Group 3"))
        count = 0
        for k in range(len(fs)):
            count = count + 1
            if kmeans_clustering.labels_[k] == 0:
                fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                              row=1, col=1)
            elif kmeans_clustering.labels_[k] == 1:
                fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                              row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                              row=1, col=3)
    else:
        numRow = 2
        numCol = int(np.ceil(hv_cluster_num / 2))
        subplot_titles = []
        for i in range(hv_cluster_num):
            subplot_titles.append("Group " + str(i + 1))
        fig = make_subplots(rows=numRow, cols=numCol, subplot_titles=subplot_titles)
        # plot the result of classification
        for z in range(hv_cluster_num):  # z is the number of classification
            count = 0
            for k in range(len(fs)):
                if kmeans_clustering.labels_[k] == z:
                    count = count + 1
                    if z < numCol:
                        fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                                      row=1, col=z+1)
                    else:
                        fig.add_trace(go.Scatter(x=hvsr_freq, y=hvsr_data[k], name=names[k]),
                                      row=2, col=z+1-numCol)
    # update the plot frame
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[math.log10(3), math.log10(60)])
    maxYvalue = math.ceil(max(max(data_sel,key=lambda x : max(x)))) + 1
    fig.update_yaxes(range=[0.0, maxYvalue], tick0=0.0, dtick=int(maxYvalue/5))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(title='HVSR cluster', showlegend=False)
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    # print the Classification result of HVSR
    for z in range(hv_cluster_num):
        print('Group {:.0f} '.format(z + 1))
        s = []
        for i in range(len(fs)):
            if kmeans_clustering.labels_[i] == z:
                s.append(names[i])
        print(', '.join(s))
    # save the .html file
    default_html_name = folder_path + '/' + 'hv_kmeans_cluster.html'
    plotly.offline.plot(fig, filename=default_html_name)