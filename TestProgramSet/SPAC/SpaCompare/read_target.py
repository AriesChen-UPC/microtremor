# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/26/2022 11:41 AM
@file: read_target.py
"""

from ioGpy import AutocorrTarget
import math


def read_target(spac_list):
    """
    read_target: read target file and return a list.

    Args:
        spac_list: the list of spac file path

    Returns:
        ring_all: the list of ring
        freq_all: the list of frequency
        spac_all: the list of spac
        is_ring_same: whether the ring is same

    """
    target_instance = AutocorrTarget()
    ring_all = []
    freq_all = []
    spac_all = []  # format: list
    for i in range(len(spac_list)):
        target_instance.load(spac_list[i])
        ring_data = []
        freq_data = []
        spac_data = []
        for j in range(len(target_instance.AutocorrCurves.ModalCurve)):
            ring_data.append(target_instance.AutocorrCurves.AutocorrRing[j])
            freq_data.append([0 if math.isnan(x) else x for x in target_instance.
                             AutocorrCurves.ModalCurve[j].RealStatisticalPoint['x']])
            spac_data.append([0 if math.isnan(x) else x for x in target_instance.
                             AutocorrCurves.ModalCurve[j].RealStatisticalPoint['mean']])
        ring_all.append(ring_data)
        freq_all.append(freq_data)
        spac_all.append(spac_data)
    is_ring_same = True
    for k in range(len(ring_all) - 1):
        if ring_all[k] != ring_all[k + 1]:
            is_ring_same = False
            print('\033[0;31mThe number of ring is not same!\033[0m')
            break
    return ring_all, freq_all, spac_all, is_ring_same
