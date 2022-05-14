# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 10:54 AM
@file: read_spac.py
"""

import operator
import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
from pathlib import Path
from ioGpy import AutocorrTarget
from read_page import read_page
from tqdm import trange
import math


def read_spac():
    """
    read_spac: read the spac file and return the data.

    """
    print('Start of the program SPAC clusters :)')
    print("\033[0;36m-------------------------Ifo-------------------------\033[0m")
    print("Please choose the path of the SPAC data:")
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
    # check the file type
    print('The filetype is .target ? (Y/y for .target, N/n for .page)')
    fileType = input().upper()  # change the char to upper format
    # read the SPAC data from .page or .target
    print('\033[0;31m----------------------Loading...---------------------\033[0m')
    if fileType == 'Y':  # todo: confusing data storage formats, optimize the data type
        fs = glob(folder_path + '/*.target')
        names = []
        targetInstance = AutocorrTarget()
        spac_data_names = locals()
        freq_data_names = locals()
        spac_ring_data_names = locals()
        for i in trange(len(fs)):
            names.append(Path(fs[i]).stem)
            spac_data_names['spac_' + str(i)] = []
            freq_data_names['freq_' + str(i)] = []
            spac_ring_data_names['spac_ring_' + str(i)] = []
            targetInstance.load(fs[i])
            for j in range(len(targetInstance.AutocorrCurves.ModalCurve)):
                spac_ring_data_names['spac_ring_' + str(i)].append(targetInstance.AutocorrCurves.AutocorrRing[j])
                spac_data_names['spac_' + str(i)].append([0 if math.isnan(x) else x for x in targetInstance.
                                                         AutocorrCurves.ModalCurve[j].RealStatisticalPoint['mean']])
                freq_data_names['freq_' + str(i)].append([0 if math.isnan(x) else x for x in targetInstance.
                                                         AutocorrCurves.ModalCurve[j].RealStatisticalPoint['x']])
        for i in range(len(fs)-1):
            is_same_length = operator.eq(spac_ring_data_names['spac_ring_' + str(i)],
                                         spac_ring_data_names['spac_ring_' + str(i+1)])
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
            print('\033[0;31m--------------------Attention------------------------\033[0m')
            print('\033[0;31mThe SPAC data is not the same length!\033[0m')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)]) == min_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                    break
            print('Please input the radius of the SPAC data: ')
            radius = float(input())
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(spac_data_names['spac_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].
                            index(radius)])
                freq.append(freq_data_names['freq_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].
                            index(radius)])
                freq_min.append(min(freq[i]))
                freq_len.append(len(freq[i]))
            freqMin = max(freq_min)
            for i in range(len(freq)):
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freq_len = len(freq[0])
        else:
            print('\033[0;32mThe SPAC data is the same length!\033[0m')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)]) == max_num_rings:
                    print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                    break
            print('Please input the radius of the SPAC data: ')
            radius = float(input())
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(spac_data_names['spac_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].
                            index(radius)])
                freq.append(freq_data_names['freq_' + str(i)][spac_ring_data_names['spac_ring_' + str(i)].
                            index(radius)])
                freq_min.append(min(freq[i]))
                freq_len.append(len(freq[i]))
            freqMin = max(freq_min)
            for i in range(len(freq)):
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freq_len = len(freq[0])
    else:
        fs = glob(folder_path + '/*.page')
        names = []
        for i in range(len(fs)):
            names.append(Path(fs[i]).stem)
        data_names = locals()
        spac_ring_data_names = locals()
        for i in trange(len(fs)):
            data_names['spac_' + str(i)] = []
            spac_ring_data_names['spac_ring_' + str(i)] = []
            data, name, ring_name = read_page(fs[i])
            data_names['spac_' + str(i)].append(data)
            spac_ring_data_names['spac_ring_' + str(i)].append(ring_name)
        for i in range(len(fs)-1):
            is_same_length = operator.eq(spac_ring_data_names['spac_ring_' + str(i)],
                                         spac_ring_data_names['spac_ring_' + str(i+1)])
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
            print('\033[0;31m--------------------Attention------------------------\033[0m')
            print('\033[0;31mThe SPAC data is not the same length!\033[0m')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)][0]) == min_num_rings:
                    print('The radius of the rings are available: '
                          + str(spac_ring_data_names['spac_ring_' + str(i)][0]))
                    break
            print('Please input the radius of the SPAC data: ')
            radius = float(input())
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                            index(radius)]['spac'])
                freq.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                            index(radius)]['freq'])
                freq_min.append(min(freq[i]))
                freq_len.append(len(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                                    index(radius)]['spac']))  # set the data dimension
            freqMin = max(freq_min)
            for i in range(len(freq)):
                freq[i] = freq[i].to_list()
                spac[i] = spac[i].to_list()
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freq_len = len(freq[0])
        else:
            print('\033[0;32mThe SPAC data is the same length!\033[0m')
            for i in range(len(fs)):
                if len(spac_ring_data_names['spac_ring_' + str(i)][0]) == max_num_rings:
                    print('The radius of the rings are available: ' +
                          str(spac_ring_data_names['spac_ring_' + str(i)][0]))
                    break
            print('Please input the radius of the SPAC data: ')
            radius = float(input())
            print('The radius is: ' + str(radius) + 'm')
            spac = []
            freq = []
            freq_min = []
            freq_len = []
            for i in range(len(fs)):
                spac.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                            index(radius)]['spac'])
                freq.append(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                            index(radius)]['freq'])
                freq_min.append(min(freq[i]))
                freq_len.append(len(data_names['spac_' + str(i)][0][spac_ring_data_names['spac_ring_' + str(i)][0].
                                    index(radius)]['spac']))  # set the data dimension
            freqMin = max(freq_min)
            for i in range(len(freq)):
                freq[i] = freq[i].to_list()
                spac[i] = spac[i].to_list()
                end_index = freq[i].index(freqMin)
                del freq[i][0:end_index]
                del spac[i][0:end_index]
            freq_len = len(freq[0])
    print('\033[0;32m-----------------------Loaded!-----------------------\033[0m')
    min_freq = float(input('Please input min freq: from ' + str(round(min(freq[0]))) + ' Hz to '
                           + str(round(max(freq[0]))) + ' Hz\n'))
    max_freq = float(input('Please input max freq: from ' + str(round(min(freq[0]))) + ' Hz to '
                           + str(round(max(freq[0]))) + ' Hz\n'))
    # slice the data by min_freq and max_freq
    for freq_value in freq[0]:
        if freq_value >= min_freq:
            freq_min_index = freq[0].index(freq_value)
            break
    for freq_value in freq[0]:
        if freq_value >= max_freq:
            freq_max_index = freq[0].index(freq_value)
            break
    freq_fliter = []
    spac_fliter = []
    for i in range(len(freq)):
        freq_fliter.append(freq[i][freq_min_index:freq_max_index])
        spac_fliter.append(spac[i][freq_min_index:freq_max_index])
    freq_len = len(freq[0])
    print('\033[0;32mSPAC data loaded!\033[0m')
    return folder_path, fs, names, spac, freq, spac_fliter, freq_fliter, radius, min_freq, max_freq, freq_len
