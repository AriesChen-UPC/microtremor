# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 11/23/2022 1:15 PM
@File     : read_hv.py
@Software : PyCharm
"""

import os
from glob import glob
from pathlib import Path
import pandas as pd
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")


def read_hv(folder_path, min_freq, max_freq):

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
        print('FileType: This folder has a total of [%s] file %d '
              % (each_type, type_dict[each_type]))
    print('-------------------------Ifo-------------------------')
    fs = glob(folder_path + '/*.hv')
    hvsr_data = pd.DataFrame()
    for i in trange(len(fs)):
        # here skiprows=9 is the headers of .hv data
        data = pd.read_table(filepath_or_buffer=fs[i], sep='\t', skiprows=9,
                             names=['Freq', 'Aver', 'min_data', 'max_data'])
        hvsr_data = pd.concat([hvsr_data, data.Aver], axis=1)
    hvsr_freq = data['Freq']
    freq_select = [min_freq, max_freq]  # minfeq, maxfreq
    freq_index = (hvsr_freq >= freq_select[0]) & (hvsr_freq <= freq_select[1])
    data_sel = hvsr_data[freq_index].T.values.tolist()
    hvsr_freq = data['Freq'].T.values.tolist()
    hvsr_data = hvsr_data.T.values.tolist()
    print('-----------------------Loaded!-----------------------')
    # # get the .hv file name
    names = []
    for i in range(len(fs)):
        names.append(Path(fs[i]).stem)
    return folder_path, fs, names, hvsr_data, hvsr_freq, data_sel

