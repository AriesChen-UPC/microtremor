# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 10:54 AM
@file: read_spac.py
"""

import operator
import os
from glob import glob
from pathlib import Path
from ioGpy import AutocorrTarget
from tqdm import trange
import math


def read_spac(folder_path, min_freq, max_freq, radius):
    """
    read_spac: read the spac file and return the data.

    """
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
        print('FileType: This folder has a total of [%s]file %d '
              % (each_type, type_dict[each_type]))
    print('-------------------------Ifo-------------------------')
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
        print('--------------------Attention------------------------')
        print('The SPAC data is not the same length!')
        for i in range(len(fs)):
            if len(spac_ring_data_names['spac_ring_' + str(i)]) == min_num_rings:
                print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                break
        if radius is None:
            radius = float(spac_ring_data_names['spac_ring_' + str(0)][0])
        else:
            radius = float(radius)
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
        print('The SPAC data is the same length!')
        for i in range(len(fs)):
            if len(spac_ring_data_names['spac_ring_' + str(i)]) == max_num_rings:
                print('The radius of the rings are available: ' + str(spac_ring_data_names['spac_ring_' + str(i)]))
                break
        if radius is None:
            radius = float(spac_ring_data_names['spac_ring_' + str(0)][0])
        else:
            radius = float(radius)
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
    print('-----------------------Loaded!-----------------------')
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
    print('SPAC data loaded!')
    return folder_path, fs, names, spac, freq, spac_fliter, freq_fliter, radius, min_freq, max_freq, freq_len