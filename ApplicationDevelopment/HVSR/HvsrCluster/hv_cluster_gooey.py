# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/11/2022 4:46 PM
@file: hv_cluster_gooey.py
"""

import os
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, vq
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import math
from gooey import Gooey, GooeyParser


@Gooey(program_name='HVSR Cluster', required_cols=2, optional_cols=1)
def main():
    parser = GooeyParser(description="This program is used to cluster the HVSR using kmeans method!")
    parser.add_argument('Path', help="Please select the HVSR data path.", widget="DirChooser")
    parser.add_argument('minFreq', help="Please input the min freq.", widget="TextField")
    parser.add_argument('maxFreq', help="Please input the max freq.", widget="TextField")
    parser.add_argument('Number', help="Please input the number of clusters.", widget="TextField")
    parser.add_argument('-Result', help="Rename the result html file.", widget="TextField")
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = main()
    Folderpath = args.Path
    print('----------------------Files Ifo----------------------')
    print('FolderPath:', Folderpath)
    all_files = os.listdir(Folderpath)
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
        print('FileType: This folder has a total of [%s] file %d '
              % (each_type, type_dict[each_type]))  # highlight the filetype
    fs = glob(Folderpath + '/*.hv')
    # define the range of Freq
    freq = np.geomspace(0.1, 100.1, 400)
    # define the min and max Freq you wanted
    min_freq = float(args.minFreq)
    max_freq = float(args.maxFreq)
    # read the HVSR data
    sel = (freq >= min_freq) & (freq <= max_freq)
    freq_sel = freq[sel]
    data = []
    data_sel = np.zeros(shape=(len(fs), len(freq_sel)))
    data2 = np.zeros(shape=(len(fs), len(freq)))
    for i in range(len(fs)):
        # here skiprows=9 is the headers of .hv data
        data.append(pd.read_table(filepath_or_buffer=fs[i], sep='\t', skiprows=9, names=['Freq', 'Aver', 'max', 'min']))
        data2[i] = np.interp(freq, data[i].Freq, data[i].Aver)  # interpolation for the same data dimension
        data_sel[i] = data2[i][sel]
    # change numpy array to pandas dataframe
    data2 = pd.DataFrame(data2).T
    # get the name of dataframe.columns
    hvname = [f'hv{n}' for n in range(len(fs))]
    # change the dataframe.columns name to what you create
    data2.columns = hvname
    # change numpy array to pandas dataframe
    freq = pd.DataFrame(freq)
    # change the dataframe.columns name to what you want
    freq.columns = ['freq']
    # get the .hv file name
    names = []
    for i in range(len(fs)):
        names.append(Path(fs[i]).stem)
    # define the number which you want to sort
    n_clusters = int(args.Number)
    centroid, _ = kmeans(data_sel, n_clusters)
    result1, _ = vq(data_sel, centroid)
    result1 = result1.tolist()
    # print(result1)
    # define the plotly subplot
    pio.templates.default = "plotly_white"  # set the plotly templates
    if n_clusters == 1:
        fig = make_subplots(rows=1, cols=1, subplot_titles="Group 1")
        count = 0
        for k in range(len(fs)):
            count = count + 1
            fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                          row=1, col=1)
    elif n_clusters == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Group 1", "Group 2"))
        count = 0
        for k in range(len(fs)):
            count = count + 1
            if result1[k] == 0:
                fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                              row=1, col=2)
    elif n_clusters == 3:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Group 1", "Group 2", "Group 3"))
        count = 0
        for k in range(len(fs)):
            count = count + 1
            if result1[k] == 0:
                fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                              row=1, col=1)
            elif result1[k] == 1:
                fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                              row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                              row=1, col=3)
    else:
        numRow = 2
        numCol = int(np.ceil(n_clusters / 2))
        subplot_titles = []
        for i in range(n_clusters):
            subplot_titles.append("Group " + str(i + 1))
        fig = make_subplots(rows=numRow, cols=numCol, subplot_titles=subplot_titles)
        # plot the result of classification
        for z in range(n_clusters):  # z is the number of classification
            count = 0
            for k in range(len(fs)):
                if result1[k] == z:
                    count = count + 1
                    if z < numCol:
                        fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                                      row=1, col=z + 1)
                    else:
                        fig.add_trace(go.Scatter(x=freq['freq'], y=data2[hvname[k]], name=names[k]),
                                      row=2, col=z + 1 - numCol)
    # update the plot frame
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[math.log10(3), math.log10(50)])
    maxYvalue = int(max(data2.loc[175:332, :].max())) + 1  # get the max value of y axis with freq.range(2,30)
    fig.update_yaxes(range=[0.0, maxYvalue], tick0=0.0, dtick=int(maxYvalue / 5))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(title='HVSR cluster', showlegend=False)
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    print('------------------------Result------------------------')
    for z in range(n_clusters):
        print('Group {:.0f} '.format(z + 1))
        s = []
        for i in range(len(fs)):
            if result1[i] == z:
                s.append(names[i])
        print(', '.join(s))
    # save the .html file
    if not args.Result:
        default_html_name = Folderpath + '/' + 'spaCluster.html'
        plotly.offline.plot(fig, filename=default_html_name)
    else:
        rename_html_name = args.Result
        rename_html_name = Folderpath + '/' + rename_html_name + '.html'
        plotly.offline.plot(fig, filename=rename_html_name)
    print('-------------------------End-------------------------')
