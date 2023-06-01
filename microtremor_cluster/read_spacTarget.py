# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 11/8/2022 1:26 PM
@file: read_spacTarget.py
"""

from xml.dom import minidom
import pandas as pd
from glob import glob


def read_spacTarget(folder_path, min_freq, max_freq, radius):

    fs = glob(folder_path + '/*.target')
    # Store the data in a list
    names = []
    spac = []
    freq = []
    spac_filter = []
    freq_filter = []

    for f in fs:
        # Get the file name
        names.append(f.split('\\')[-1].split('.target')[0])
        # Read the data
        xml_file = minidom.parse(f)
        xml_elements = xml_file.documentElement
        sub_ringElements = xml_elements.getElementsByTagName('AutocorrRing')
        sub_curveElement = xml_elements.getElementsByTagName('ModalCurve')  # Child nodes which contains curves
        data_length = len(sub_ringElements)
        ring_data = []
        for i in range(data_length):
            ring_data.append((float(sub_ringElements[i].getElementsByTagName('minRadius')[0].firstChild.data) +
                              float(sub_ringElements[i].getElementsByTagName('maxRadius')[0].firstChild.data)) / 2)

        curve_length = len(sub_curveElement[0].getElementsByTagName('mean'))
        x_data = []
        for j in range(curve_length):
            x_data.append(float(sub_curveElement[0].getElementsByTagName('x')[j].firstChild.data))
        curve_data = [[] for i in range(data_length)]
        for i in range(data_length):
            for j in range(curve_length):
                curve_data[i].append(float(sub_curveElement[i].getElementsByTagName('mean')[j].firstChild.data))
        spac_data = pd.DataFrame(curve_data).T
        spac_data.columns = ['ring-' + str(r) for r in ring_data]
        freq_data = pd.DataFrame(x_data)
        freq_data.columns = ['freq']
        spac_All = pd.concat([freq_data, spac_data], axis=1)
        # Find the index of the radius
        if radius is None:
            radius = ring_data[0]
        radius_index = spac_All.columns.get_loc('ring-' + str(radius))

        spac.append(spac_All.iloc[:, radius_index].values)
        freq.append(spac_All.iloc[:, 0].values)

        spac_slice = spac_All[(spac_All['freq'] >= min_freq) & (spac_All['freq'] <= max_freq)]

        spac_filter.append(spac_slice.iloc[:, radius_index].values)
        freq_filter.append(spac_slice.iloc[:, 0].values)

    print('The radius of the rings are: ', ring_data)

    return folder_path, fs, names, spac, freq, spac_filter, freq_filter, radius, min_freq, max_freq
