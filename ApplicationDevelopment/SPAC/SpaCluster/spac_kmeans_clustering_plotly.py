# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 11:34 AM
@file: spac_kmeans_clustering_plotly.py
"""

import os
import numpy as np
from pandas import DataFrame
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


def spac_kmeans_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac, min_freq, max_freq,
                                  radius, vs_reference, kmeans_clustering, kmeans_clusters):
    """
    Args:
        folder_path: the path of the folder which stores the spac file.
        fs: the number of the spac file.
        freq: the frequency of the spac data.
        spac: the spac of the spac data.
        names: the name of the spac file.
        freq_theory_spac: the frequency of the theoretical spac.
        spac_theory_spac: the theoretical spac.
        min_freq: the minimum frequency of the spac data.
        max_freq: the maximum frequency of the spac data.
        radius: the radius of the spac data.
        vs_reference: the reference of dispersion curve.
        clustering: the clustering result.
        kmeans_clusters: the number of clusters.

    Returns:
        None.

    """
    pio.templates.default = "plotly_white"  # set the plotly templates
    # set the min/max of the Freq
    min_freq_x_line = DataFrame(np.ones((1, 100)) * min_freq).T
    min_freq_y_line = DataFrame(np.arange(-1, 1, 0.02))
    max_freq_x_line = DataFrame(np.ones((1, 100)) * max_freq).T
    max_freq_y_line = DataFrame(np.arange(-1, 1, 0.02))
    if kmeans_clusters == 1:
        fig = make_subplots(rows=1, cols=1, subplot_titles="Group 1")
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        count = 0  # count the cluster
        for k in range(len(fs)):
            fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k]), row=1, col=1)
        # subplot num is 2
    elif kmeans_clusters == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Group 1", "Group 2"))
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=2)
        count = 0  # count the number of each clusters
        for k in range(len(fs)):
            if kmeans_clustering.labels_[k] == 0:
                fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                              row=1, col=2)
        # subplot num is 3
    elif kmeans_clusters == 3:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Group 1", "Group 2", "Group 3"))
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=3)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=3)
        count = 0  # count the number of each clusters
        for k in range(len(fs)):
            if kmeans_clustering.labels_[k] == 0:
                fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                              row=1, col=1)
            elif kmeans_clustering.labels_[k] == 1:
                fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                              row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                              row=1, col=3)
        # subplot num is more than 3
    else:
        numRow = 2
        numCol = int(np.ceil(kmeans_clusters / 2))
        fig = make_subplots(rows=numRow, cols=numCol)
        for z in range(kmeans_clusters):  # z is the number of classification
            if z < numCol:
                fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                         mode='lines', showlegend=False,
                                         line=dict(
                                             color='rgb(132,133,135)',
                                             width=2,
                                             dash='dash')),
                              row=1, col=z + 1)
                fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=z+1)
                fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=z+1)
            else:
                fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=spac_theory_spac['autoCorrRatio'], name='theory SPAC',
                                         mode='lines', showlegend=False,
                                         line=dict(
                                             color='rgb(132,133,135)',
                                             width=2,
                                             dash='dash')),
                              row=2, col=z+1-numCol)
                fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')),
                              row=2, col=z+1-numCol)
                fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')),
                              row=2, col=z+1-numCol)

        for z in range(kmeans_clusters):
            for k in range(len(fs)):
                if kmeans_clustering.labels_[k] == z:
                    if z < numCol:
                        fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                                      row=1, col=z+1)
                    else:
                        fig.add_trace(go.Scatter(x=freq[k], y=spac[k], name=names[k], showlegend=True),
                                      row=2, col=z+1-numCol)
    # print the Classification result of SPAC
    for z in range(kmeans_clusters):
        print('Group {:.0f} '.format(z + 1))
        s = []
        for i in range(len(fs)):
            if kmeans_clustering.labels_[i] == z:
                s.append(names[i])
        print(', '.join(s))

    fig.update_xaxes(type="log", range=[np.log10(1), np.log10(100)])
    fig.update_yaxes(range=[-0.6, 1.0], tick0=0.0, dtick=0.2)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Autocorr ratio")
    fig.update_layout(title='SPAC cluster, ' + 'Ring=' + str(radius) + 'm, ' + 'Vs(nearly 1Hz, depth 57.5m)='
                            + str(vs_reference) + 'm/s')
    fig.update_layout(
        showlegend=False,  #set the legend to be hidden
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    default_html_name = folder_path + '/' + 'spac_kmeans_cluster.html'
    plotly.offline.plot(fig, filename=default_html_name)
    print('\033[0;31mDo you want to rename the result .html file ? (Y/y for Yes, N/n for No)\033[0m')
    nameHtml = input().upper()  # change the char to upper format
    if nameHtml == 'Y':
        rename_html_name = input('Please input the name of the .html file: \n')
        rename_html_name = folder_path + '/' + rename_html_name + '.html'
        if os.path.exists(rename_html_name):
            print('\033[0;31mThe file already exists, The original file will be overwritten!\033[0m')
            os.remove(rename_html_name)
            os.rename(default_html_name, rename_html_name)
        else:
            os.rename(default_html_name, rename_html_name)
    else:
        rename_html_name = default_html_name
    print('\033[0;32m------------------------Done!------------------------\033[0m')