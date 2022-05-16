# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/11/2022 2:15 PM
@file: spac_cluster_gooey_test.py
"""

import operator
import os
from glob import glob
from pathlib import Path
from pandas import DataFrame
from plotly.subplots import make_subplots
from scipy.cluster.vq import kmeans, vq
import numpy as np
from gooey import Gooey, GooeyParser
from cps_disp import cps_disp
from disp_CPS_gooey import disp_CPS_gooey
from ioGpy import AutocorrTarget
from read_page import read_page
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import math
from scipy.special import *


@Gooey(program_name='SPAC Cluster', required_cols=2, optional_cols=2, default_size=(900, 700))
def main():
    parser = GooeyParser(description="This program is used to cluster the SPAC using kmeans method!")
    parser.add_argument('Path', help="Please select the SPAC data path.", widget="DirChooser")
    parser.add_argument('Target', help="The filetype is .target ? (Y/y for .target, N/n for .page)", widget="Dropdown",
                        choices=['Y', 'N'], default='Y')
    parser.add_argument('minFreq', help="Please input the min freq.", widget="TextField", default=2)
    parser.add_argument('maxFreq', help="Please input the max freq.", widget="TextField", default=20)
    parser.add_argument('Number', help="Please input the number of clusters.", widget="TextField", default=6)
    parser.add_argument('SPAC', help="Please select the method for reference SPAC. If choose eIndex, VS parameters "
                                     "must be provided. If choose CPS, model must be selected.",
                        widget="Dropdown", choices=['eIndex', 'CPS'])
    parser.add_argument('-Radius', help="Please input the radius of SPAC.", widget="TextField")
    parser.add_argument('-VS', help="Please input the reference VS.", widget="TextField")
    parser.add_argument('-Model', help="Please select the model path.", widget="FileChooser")
    parser.add_argument('-Result', help="Rename the result html file.", widget="TextField")
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = main()
    # read data
    print('----------------------Files Ifo----------------------')
    Folderpath = args.Path
    print('FolderPath:', Folderpath)
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
        print('FileType: This folder has a total of [%s] file %d '
              % (each_type, type_dict[each_type]))  # highlight the filetype
    print('----------------------SPAC  Ifo----------------------')
    fileType = args.Target
    if fileType == 'Y':
        fs = glob(Folderpath + '/*.target')
        names = []
        targetInstance = AutocorrTarget()
        spac_data_names = locals()
        freq_data_names = locals()
        spac_ring_data_names = locals()
        for i in range(len(fs)):
            names.append(Path(fs[i]).stem)
            spac_data_names['spac_' + str(i)] = []
            freq_data_names['freq_' + str(i)] = []
            spac_ring_data_names['spac_ring_' + str(i)] = []
            targetInstance.load(fs[i])
            for j in range(len(targetInstance.AutocorrCurves.ModalCurve)):
                spac_ring_data_names['spac_ring_' + str(i)].append(targetInstance.AutocorrCurves.AutocorrRing[j])
                spac_data_names['spac_' + str(i)].append([0 if math.isnan(x) else x for x in targetInstance.AutocorrCurves.ModalCurve[j].RealStatisticalPoint['mean']])
                freq_data_names['freq_' + str(i)].append([0 if math.isnan(x) else x for x in targetInstance.AutocorrCurves.ModalCurve[j].RealStatisticalPoint['x']])
        for i in range(len(fs)-1):
            is_same_length = operator.eq(spac_ring_data_names['spac_ring_' + str(i)], spac_ring_data_names['spac_ring_' + str(i+1)])
            if is_same_length == False:
                break
        max_num_rings = len(spac_ring_data_names['spac_ring_' + str(0)])
        min_num_rings = len(spac_ring_data_names['spac_ring_' + str(0)])
        for i in range(len(fs)):
            if len(spac_ring_data_names['spac_ring_' + str(i)]) > max_num_rings:
                max_num_rings = len(spac_ring_data_names['spac_ring_' + str(i)])
            if len(spac_ring_data_names['spac_ring_' + str(i)]) < min_num_rings:
                min_num_rings = len(spac_ring_data_names['spac_ring_' + str(i)])
        if is_same_length == False:
            print('--------------------Attention------------------------')
            print('The SPAC data is not the same length!')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)]) == min_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                    break
            if not args.Radius:
                radius = spac_ring_data_names['spac_ring_' + str(i)][0]
            else:
                radius = float(args.Radius)
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(spac_data_names['spac_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].index(radius)])
                freq.append(freq_data_names['freq_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].index(radius)])
                freq_min.append(min(freq[i]))
                freq_len.append(len(freq[i]))
            freqMin = max(freq_min)
            for i in range(len(freq)):
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freqLen = len(freq[0])
        else:
            print('The SPAC data is the same length!')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)]) == max_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                    break
            if not args.Radius:
                radius = spac_ring_data_names['spac_ring_' + str(i)][0]
            else:
                radius = float(args.Radius)
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(spac_data_names['spac_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].index(radius)])
                freq.append(freq_data_names['freq_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].index(radius)])
                freq_min.append(min(freq[i]))
                freq_len.append(len(freq[i]))
            freqMin = max(freq_min)
            for i in range(len(freq)):
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freqLen = len(freq[0])
    else:
        fs = glob(Folderpath + '/*.page')
        names = []
        for i in range(len(fs)):
            names.append(Path(fs[i]).stem)
        data_names = locals()
        spac_ring_data_names = locals()
        for i in range(len(fs)):
            data_names['spac_' + str(i)] = []
            spac_ring_data_names['spac_ring_' + str(i)] = []
            data, name, ring_name = read_page(fs[i])
            data_names['spac_' + str(i)].append(data)
            spac_ring_data_names['spac_ring_' + str(i)].append(ring_name)
        for i in range(len(fs)-1):
            is_same_length = operator.eq(spac_ring_data_names['spac_ring_' + str(i)], spac_ring_data_names['spac_ring_' + str(i+1)])
            if is_same_length == False:
                break
        max_num_rings = len(spac_ring_data_names['spac_ring_' + str(0)][0])
        min_num_rings = len(spac_ring_data_names['spac_ring_' + str(0)][0])
        for i in range(len(fs)):
            if len(spac_ring_data_names['spac_ring_' + str(i)][0]) > max_num_rings:
                max_num_rings = len(spac_ring_data_names['spac_ring_' + str(i)][0])
            if len(spac_ring_data_names['spac_ring_' + str(i)][0]) < min_num_rings:
                min_num_rings = len(spac_ring_data_names['spac_ring_' + str(i)][0])
        if is_same_length == False:
            print('--------------------Attention------------------------')
            print('The SPAC data is not the same length!')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)][0]) == min_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)][0]))
                    break
            if not args.Radius:
                radius = spac_ring_data_names['spac_ring_' + str(i)][0]
            else:
                radius = float(args.Radius)
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['spac'])
                freq.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['freq'])
                freq_min.append(min(freq[i]))
                freq_len.append(len(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['spac']))  # set the data dimension
            freqMin = max(freq_min)
            for i in range(len(freq)):
                freq[i] = freq[i].to_list()
                spac[i] = spac[i].to_list()
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freqLen = len(freq[0])
        else:
            print('The SPAC data is the same length!')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)][0]) == max_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)][0]))
                    break
            if not args.Radius:
                radius = spac_ring_data_names['spac_ring_' + str(i)][0]
            else:
                radius = float(args.Radius)
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['spac'])
                freq.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['freq'])
                freq_min.append(min(freq[i]))
                freq_len.append(len(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].index(radius)]['spac']))  # set the data dimension
            freqMin = max(freq_min)
            for i in range(len(freq)):
                freq[i] = freq[i].to_list()
                spac[i] = spac[i].to_list()
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freqLen = len(freq[0])
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
    min_freq = float(args.minFreq)
    max_freq = float(args.maxFreq)
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
    n_clusters = int(args.Number)
    centroid, _ = kmeans(detail, n_clusters)
    cluster, _ = vq(detail, centroid)
    cluster = cluster.tolist()
    freq = DataFrame(freq)  # list2DataFrame
    spac = DataFrame(spac)  # list2DataFrame
    # calculate the theory SPAC
    print('-------------------------SPAC------------------------')
    print('The method to calculate the theory SPAC is %s.' % args.SPAC)
    # fixme: with problem when package gpdc.exe, so the current method 2 is CPS
    spac_method = args.SPAC
    if spac_method == 'eIndex':
        r = radius  # get the radius from the .target file
        freqSPAC = np.logspace(np.log10(0.1), np.log10(100), num=400)
        # set the reference VS
        if not args.VS:
            VS = 1500
        else:
            VS = int(args.VS)
        Vs = np.dot(VS, [math.pow(f, -0.65) for f in freqSPAC])
        autoCorrRatio = jn(0, np.multiply(r * 2 * math.pi * freqSPAC, [math.pow(v, -1) for v in Vs]))
        freqSPAC = DataFrame(freqSPAC)
        freqSPAC.columns = ['freq']
        autoCorrRatio = DataFrame(autoCorrRatio)
        autoCorrRatio.columns = ['autoCorrRatio']
        Vs = DataFrame(Vs)
        Vs.columns = ['Vs']
        print('Dispersion curve and SPAC curve calculated by eIndex were done!')
        # plot the initialized model
        pio.templates.default = "plotly_white"  # set the plotly templates
        fig = go.Figure(data=go.Scatter(x=freqSPAC['freq'], y=Vs['Vs'], name='dispersion curve'))
        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Velocity (m/s)")
        fig.update_xaxes(type="log")
        fig.update_xaxes(range=[np.log10(1), np.log10(100)])
        fig.update_yaxes(range=[0, 3000], tick0=0.0, dtick=500)
        fig.update_layout(title='Vs(nearly 1Hz, depth 57.5m)=' + str(VS) + 'm/s')
        plotly.offline.plot(fig, filename=Folderpath + '/' + 'eindex_disp.html')
    if spac_method == 'CPS':
        r = radius
        model = args.Model
        model_plot, vs_reference, freq_theory_spac, spac_theory_spac, R_matrix_S = disp_CPS_gooey(radius, model)
        cps_disp(Folderpath, model_plot, freq_theory_spac, R_matrix_S)
        freqSPAC = freq_theory_spac
        autoCorrRatio = spac_theory_spac
        VS = vs_reference
        print('Dispersion curve and SPAC curve calculated by CPS were done!')
    # define the plotly subplot
    min_freq_x_line = DataFrame(np.ones((1, 100)) * min_freq).T
    min_freq_y_line = DataFrame(np.arange(-1, 1, 0.02))
    max_freq_x_line = DataFrame(np.ones((1, 100)) * max_freq).T
    max_freq_y_line = DataFrame(np.arange(-1, 1, 0.02))
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
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
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
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
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
        fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                 line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
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
        fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
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
                fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=z + 1)
                fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=1, col=z + 1)
            else:
                fig.add_trace(go.Scatter(x=freqSPAC['freq'], y=autoCorrRatio['autoCorrRatio'], name='theory SPAC',
                                         mode='lines', showlegend=False,
                                         line=dict(
                                             color='rgb(132,133,135)',
                                             width=2,
                                             dash='dash')),
                              row=2, col=z + 1 - numCol)
                fig.add_trace(go.Scatter(x=min_freq_x_line[0], y=min_freq_y_line[0], mode='lines', name='min freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=2, col=z + 1 - numCol)
                fig.add_trace(go.Scatter(x=max_freq_x_line[0], y=max_freq_y_line[0], mode='lines', name='max freq',
                                         line=dict(color='rgb(192,192,192)', width=2, dash='dash')), row=2, col=z + 1 - numCol)
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
    if spac_method == 'eIndex':
        fig.update_layout(title='SPAC cluster, ' + 'Ring=' + str(radius) + 'm, ' +
                                'Vs(nearly 1Hz, depth 57.5m)=' + str(VS) + 'm/s')
    if spac_method == 'CPS':
        fig.update_layout(title='SPAC cluster, ' + 'Ring=' + str(radius) + 'm, ' +
                                'Vs average =' + str(VS) + 'm/s')
    fig.update_layout(
        showlegend=False,  #set the legend to be hidden
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    # print the Classification result of SPAC
    print('------------------------Result------------------------')
    for z in range(n_clusters):
        print('Group {:.0f} '.format(z + 1))
        s = []
        for i in range(len(fs)):
            if cluster[i] == z:
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
