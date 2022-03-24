# encoding: UTF-8
"""
@author: Aries Chen
@contact: s15010125@s.upc.edu.cn
@time: 05/11/2021
@update:02/25/2022
@file: spac_cluster.py
       This code is designead to sort the SPAC using keans and plot the result with plotly.
"""


import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans, vq
from pandas import DataFrame
from ioGpy import AutocorrTarget
from read_page import read_page
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import time
from tqdm import trange
import math
from scipy.special import *

def spac_cluster():
    # choose the path
    print('Start of the program SPAC clusters :)')
    print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
    print('Please choose the path of the SPAC data:')
    root = tk.Tk()
    root.withdraw()
    Folderpath = filedialog.askdirectory()
    print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
    print('\033[0;31mFolderPath\033[0m:', Folderpath)
    root.destroy()
    # file type ifo in the selected file path
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
        print('\033[0;34mFileType\033[0m: This folder has a total of \033[0;36m[%s]\033[0m file %d '
              % (each_type, type_dict[each_type]))  # highlight the filetype
    print('\033[0;36m-------------------------Ifo-------------------------\033[0m')
    # check the file type
    print('The filetype is .target ? (Y/y for .target, N/n for .page)')
    fileType = input().upper()  # change the char to upper format
    # read the SPAC data from .page or .target
    if fileType == 'Y':
        fs = glob(Folderpath + '/*.target')
        names = []
        targetInstance = AutocorrTarget()
        targetInstance.load(fs[0])
        totalNumRing = len(targetInstance.AutocorrCurves.ModalCurve)
        print('The total number of rings is : ' + str(totalNumRing))
        print('Please select the ring: ' + 'from 1 to ' + str(totalNumRing))
        numRings = int(input()) - 1
        radius = targetInstance.AutocorrCurves.AutocorrRing[numRings]
        freqLen = len(targetInstance.AutocorrCurves.ModalCurve[0].RealStatisticalPoint['mean'])  # set the data dimension
        spac = []
        freq = []
        print('\033[0;31m----------------------Loading...---------------------\033[0m')
        for i in trange(len(fs)):
            time.sleep(0.1)
            names.append(Path(fs[i]).stem)
        # for i in range(len(fs)):
            targetInstance.load(fs[i])
            spac.append([0 if math.isnan(x) else x for x in targetInstance.AutocorrCurves.ModalCurve[numRings].RealStatisticalPoint['mean']])
            freq.append([0 if math.isnan(x) else x for x in targetInstance.AutocorrCurves.ModalCurve[numRings].RealStatisticalPoint['x']])
    else:
        fs = glob(Folderpath + '/*.page')
        names = []
        for i in range(len(fs)):
            names.append(Path(fs[i]).stem)
        totalNumRing = len(read_page(fs[0])[0])
        print('The total number of rings is : ' + str(totalNumRing))
        print('Please select the ring: ' + 'from 1 to ' + str(totalNumRing))
        numRings = int(input()) - 1
        freqLen = 300  # set the data dimension
        radius = int(input('Please input the radius: ' + '\n'))
        spac = []
        freq = []
        for i in range(len(fs)):
            spac.append(read_page(fs[i])[0][numRings]['spac'][-freqLen-1:-1])
            freq.append(read_page(fs[i])[0][numRings]['freq'][-freqLen-1:-1])
    print('\033[0;32m-----------------------Loaded!-----------------------\033[0m')
    # kernal
    k1 = np.array([-0.02306931, -0.1056576, 0.13388212, -0.02341026, -0.0126199,
                   -0.04117879, -0.02753881, 0.08565462, -0.20602849, 0.20614131,
                   -0.14900643, 0.11359798, -0.09662029, 0.05987384, -0.10340957,
                   0.1164462, 0.06197933, -0.06145484, -0.27048752, 0.17245993,
                   0.17806824, -0.03125308, -0.19900371, 0.11171456, 0.09432922,
                   -0.21939862, 0.01255652, -0.32677427, -0.4083756, 0.04717418,
                   0.05470994, 0.29249766, 0.33997664, -0.21587674, -0.01858459,
                   0.13910776, -0.18480915, 0.07169731, 0.17088659, 0.08077492,
                   0.01491521, -0.06912256, -0.06817471, 0.05515498, 0.08658582,
                   0.11424139, -0.0416302, 0.04017343, 0.0067895, -0.052115,
                   0.07161448, 0.03834576, -0.08015742, 0.05258898, -0.00586048,
                   -0.00896439, 0.04149655, 0.00454619, -0.00991248, 0.05561759,
                   -0.02137935, -0.03790385, 0.03550394, 0.04025659, -0.03533922,
                   0.01554128, 0.00957493, -0.0150793, 0.01997719, -0.00605633,
                   -0.01840421, 0.00213903], dtype=np.float32)
    k2 = np.array([0.09681424, 0.0125695, -0.03296668, -0.06259093, 0.02641451,
                   0.02264782, -0.03527348, -0.01100035, -0.00835153, 0.15201114,
                   -0.00604084, -0.06569929, -0.05707217, 0.08184609, -0.03801537,
                   0.02860055, 0.02736357, 0.04610762, 0.02690667, 0.00431063,
                   -0.05841798, 0.01170933, 0.04377702, 0.03132229, 0.01884418,
                   0.00706555, 0.02693376, 0.00865195, -0.08379597, 0.3851489,
                   -0.26176456, -0.12142022, 0.13912982, 0.02746409, -0.2967254,
                   0.18555762, 1.4003265, 0.48990425, -0.61164373, -0.2229173,
                   -0.35163, 0.02921563, -0.14746712, 0.336639, -0.61782324,
                   0.08492271, 0.10232697, -0.2795302, 0.2924005, -0.23487124,
                   -0.11624496, 0.10293019, -0.06085368, -0.15410152, 0.2654883,
                   -0.31394354, 0.05724258, -0.11718074, 0.01887048, 0.06616369,
                   -0.01266902, -0.02204721, -0.06348757, -0.0873571, 0.08890857,
                   -0.014294, 0.02892015, 0.010807, -0.01736347, -0.04279665,
                   -0.02214473, -0.03366138], dtype=np.float32)
    # define the min & max freq
    print('\033[0;36m---------------------Parameter-----------------------\033[0m')
    min_freq = float(input('Please input min freq: from ' + str(round(min(freq[0]))) + 'Hz to ' + str(round(max(freq[0]))) + 'Hz\n'))
    max_freq = float(input('Please input max freq: from ' + str(round(min(freq[0]))) + 'Hz to ' + str(round(max(freq[0]))) + 'Hz\n'))
    freq_scope = np.array(freq[0])
    sel = (freq_scope >= min_freq) & (freq_scope <= max_freq)
    detail = np.zeros(shape=(len(fs), sel.sum()))
    detail1 = np.zeros(shape=(len(fs), freqLen))
    for j in range(len(fs)):
        temp2 = np.convolve(spac[j], k1, 'same')[sel]
        temp3 = np.convolve(spac[j], k2, 'same')[sel]
        temp4 = np.convolve(spac[j], k1, 'same')
        temp5 = np.convolve(spac[j], k2, 'same')
        detail[j:] = temp2 + temp3
        detail1[j:] = temp4 + temp5
    # set the clusters number
    n_clusters = int(input('Please input the number of clusters: 1 <= n_clusters <= ' + str(len(fs)) + '\n'))
    for i in range(99):  # check the clusters number
        if n_clusters <= 0 | n_clusters > len(fs):
            print('Please Re-enter the n_clusters: 1 <= n_clusters <= ' + str(len(fs)) + '\n')
            n_clusters = int(input())
        else:
            break
    centroid, _ = kmeans(detail, n_clusters)
    cluster, _ = vq(detail, centroid)
    cluster = cluster.tolist()
    freq = DataFrame(freq)  # list2DataFrame
    spac = DataFrame(spac)  # list2DataFrame
    # calculate the theory SPAC
    # set the radius
    r = radius  # get the radius from the .target file
    freqSPAC = np.logspace(np.log10(0.1), np.log10(100), num=400)
    # set the Vs0
    Vs0 = int(input('Please input the Vs0 of initialization model: ...m/s \n'))
    print('\033[0;32m-----------------------Setted!-----------------------\033[0m')
    Vs = np.dot(Vs0, [math.pow(f, -0.65) for f in freqSPAC])
    autoCorrRatio = jn(0, np.multiply(r*2*math.pi*freqSPAC, [math.pow(v, -1) for v in Vs]))
    freqSPAC = DataFrame(freqSPAC)
    freqSPAC.columns = ['freq']
    autoCorrRatio = DataFrame(autoCorrRatio)
    autoCorrRatio.columns = ['autoCorrRatio']
    Vs = DataFrame(Vs)
    Vs.columns = ['Vs']
    # plot the initialized model
    pio.templates.default = "plotly_white"  # set the plotly templates
    fig = go.Figure(data=go.Scatter(x=freqSPAC['freq'], y=Vs['Vs'], name='dispersion curve'))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Velocity (m/s)")
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[np.log10(1), np.log10(100)])
    fig.update_yaxes(range=[0, 2000], tick0=0.0, dtick=200)
    plotly.offline.plot(fig, filename=Folderpath + '/' + 'model.html')
    # define the plotly subplot
    pio.templates.default = "plotly_white"  # set the plotly templates
    # subplot num is 1
    if n_clusters == 1:
        fig = make_subplots(rows=1, cols=1, subplot_titles="Group 1")
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        count = 0  # count the cluster
        for k in range(len(fs)):
            count = count + 1
            fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k]),  # todo: legend is not visible
                          row=1, col=1)
    # subplot num is 2
    elif n_clusters == 2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Group 1", "Group 2"))
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=2)
        count = 0  # count the number of each clusters
        for k in range(len(fs)):
            count = count + 1
            if cluster[k] == 0:
                fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                              row=1, col=2)
    # subplot num is 3
    elif n_clusters == 3:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Group 1", "Group 2", "Group 3"))
        # plot the theory SPAC
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                 mode='lines', showlegend=False,
                                 line=dict(
                                     color='rgb(132,133,135)',
                                     width=2,
                                     dash='dash')),
                      row=1, col=3)
        count = 0  # count the number of each clusters
        for k in range(len(fs)):
            count = count + 1
            if cluster[k] == 0:
                fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                              row=1, col=1)
            elif cluster[k] == 1:
                fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                              row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                              row=1, col=3)
    # subplot num is more than 3
    else:
        numRow = 2
        numCol = int(np.ceil(n_clusters / 2))
        subplot_titles = []
        for i in range(n_clusters):
            subplot_titles.append("Group " + str(i + 1))
        fig = make_subplots(rows=numRow, cols=numCol, subplot_titles=subplot_titles)
        for z in range(n_clusters):  # z is the number of classification
            if z < numCol:
                fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                         mode='lines', showlegend=False,
                                         line=dict(
                                             color='rgb(132,133,135)',
                                             width=2,
                                             dash='dash')),
                              row=1, col=z + 1)
            else:
                fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                         mode='lines', showlegend=False,
                                         line=dict(
                                             color='rgb(132,133,135)',
                                             width=2,
                                             dash='dash')),
                              row=2, col=z + 1 - numCol)
            count = 0  # count the number of each clusters
            for k in range(len(fs)):
                if cluster[k] == z:
                    count = count + 1
                    if z < numCol:
                        fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                                      row=1, col=z+1)
                    else:
                        fig.add_trace(go.Scatter(x=freq.iloc[k], y=spac.iloc[k], name=names[k], showlegend=True),
                                      row=2, col=z+1-numCol)
    # update the plot frame
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[np.log10(1), np.log10(100)])
    fig.update_yaxes(range=[-0.6, 1.0], tick0=0.0, dtick=0.2)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Autocorr ratio")
    fig.update_layout(title='SPAC cluster & ' + 'Ring = ' + str(radius) + 'm')
    fig.update_layout(
        showlegend=False,  #set the legend to be hidden
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    # print the Classification result of SPAC
    for z in range(n_clusters):
        print('Group {:.0f} '.format(z + 1))
        s = []
        for i in range(len(fs)):
            if cluster[i] == z:
                s.append(names[i])
        print(', '.join(s))
    # save the .html file
    print('\033[0;31mDo you want to rename the result .html file ? (Y/y for Yes, N/n for No)\033[0m')
    nameHtml = input().upper()  # change the char to upper format
    if nameHtml == 'Y':
        htmlFileName = input('Please input the name of the .html file: \n')
        htmlFileName = htmlFileName + '.html'
    else:
        htmlFileName = Folderpath.split("/")[-1] + '_SPAC_Cluster' + '.html'
    plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
    print('\033[0;32m------------------------Done!------------------------\033[0m')

spac_cluster()  # call the function
while input('Do you want to repeat the program ? (Y/y for Yes, N/n for No) \n').upper() == 'Y':
    spac_cluster()  # restart the program
input('Press Enter to exit … \n')
# end of the program