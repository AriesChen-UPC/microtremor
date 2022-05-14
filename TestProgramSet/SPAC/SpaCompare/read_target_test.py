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
