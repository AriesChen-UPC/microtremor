# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/26/2022 11:41 AM
@file: read_target_test.py
"""

import math
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from glob import glob
from ioGpy import AutocorrTarget


def read_target_test(spac_list):
    """
    read_target_test: read target file and return a dict, test for multiple data types.

    Args:
        spac_list: the list of spac file(.target) path

    Returns:
        spac: the data of spac in dict format

    """
    target_instance = AutocorrTarget()
    spac = {}
    for i in range(len(spac_list)):
        spac_name = os.path.basename(spac_list[i]).split('.')[0]
        target_instance.load(spac_list[i])
        ring = target_instance.AutocorrCurves.AutocorrRing
        freq_data = []
        spac_data = []
        for j in range(len(ring)):
            freq_data.append([0 if math.isnan(x) else x for x in target_instance.
                             AutocorrCurves.ModalCurve[j].RealStatisticalPoint['x']])
            spac_data.append([0 if math.isnan(x) else x for x in target_instance.
                             AutocorrCurves.ModalCurve[j].RealStatisticalPoint['mean']])
        freq_frame = pd.DataFrame(data=freq_data, index=['ring-' + str(k) for k in ring]).T
        spac_frame = pd.DataFrame(data=spac_data, index=['ring-' + str(k) for k in ring]).T
        spac_dict = {'freq': freq_frame, 'spac': spac_frame, 'ring': ring}
        spac[spac_name] = spac_dict
    return spac


if __name__ == "__main__":
    print('\033[0;31mPlease select the spac file(.target) folder: \033[0m')
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    print('\033[0;32mThe selected folder is %s \033[0m' % folder_path)
    spac_list = glob(folder_path + '/*.target')
    spac = read_target_test(spac_list)
    # take the intersection
    for key in spac.keys():
        ring_list_random = spac[key]['ring']  # random initialization
    for key in spac.keys():
        ring_list = list(set(ring_list_random).intersection(set(spac[key]['ring'])))
    print('\033[0;32mThe intersection of ring is %s \033[0m' % ring_list)
    # select the ring
    print('\033[0;31mPlease select the ring: \033[0m')
    radius = input()
    print('\033[0;32mThe selected ring is %s \033[0m' % radius)
    spac_fliter = {}
    for key in spac.keys():
        freq_temp = pd.DataFrame(data=spac[key]['freq']['ring-' + radius])
        spac_temp = pd.DataFrame(data=spac[key]['spac']['ring-' + radius])
        spac_ = {'freq': freq_temp, 'spac': spac_temp}
        spac_fliter[key] = spac_


#%% read target with xml.dom

from xml.dom import minidom

dom = minidom.parse("D:\\ProjectMaterials\\Chengdu_Line13\\Part_02\\DataProcess\\20220407\\ZDK21+742\\"
                    "grid\\ZDK21+742-Grid[1-1].target")
names = dom.getElementsByTagName("x")  # get the name of the node, 'x', 'imag', 'mean', 'stddev', 'weight', 'valid'
for i in range(len(names)):
    print(names[i].firstChild.data)
