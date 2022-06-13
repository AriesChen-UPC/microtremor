# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/16/2022 8:38 AM
@file: inversion_model.py
"""

import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from geopsy_spac import geopsy_spac


def inversion_model(radius):
    print('\033[0;31mPlease select the model from inversion ... \033[0m')
    root = tkinter.Tk()
    root.withdraw()
    model = filedialog.askopenfilename()
    layer = []
    with open(model) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                each_layer = [int(param) for param in line.strip().split(' ') if param != '']
                layer.append(each_layer)

    model_data = pd.DataFrame(layer).dropna(axis=0)  # model format in Geopsy
    # model_data = pd.DataFrame(layer)/1000  # model format in dispa, using CPS
    model_data.columns = ['thickness', 'vp', 'vs', 'density']

    model_plot = pd.DataFrame(index=np.arange(10), columns=np.arange(2))
    model_plot.columns = ['depth', 'vs']
    # set the depth of the model from model_data
    model_plot['depth'][0] = 0
    for i in range(1, len(model_plot)):
        if i % 2 != 0:
            model_plot['depth'][i] = model_plot['depth'][i - 1] + model_data['thickness'][int((i - 1) / 2) + 1]
        else:
            model_plot['depth'][i] = model_plot['depth'][i - 1]
    # set the vs of the model from model_data
    for i in range(len(model_plot)):
        model_plot['vs'][i] = model_data['vs'][int(i / 2) + 1]
    depth_sum = model_data['thickness'].sum()
    vs_reference = 0
    for i in range(1, len(model_data)):
        vs_reference += model_data['vs'][i] * model_data['thickness'][i] / depth_sum
    vs_reference = round(vs_reference, 2)

    R_matrix_F, L_matrix_F, spac_R, spac_L, R_matrix_S = geopsy_spac(radius, model)
    R_matrix_F = R_matrix_F.to_frame()
    R_matrix_F.columns = ['freq']
    spac_R = spac_R.to_frame()
    spac_R.columns = ['autoCorrRatio']
    freq_theory_spac, spac_theory_spac = R_matrix_F, spac_R

    R_matrix_S = R_matrix_S.to_frame()
    R_matrix_S.columns = ['rayleigh']
    print('\033[0;32mDispersion curve and SPAC curve calculated by Geopsy were done!\033[0m')
    return model_plot, vs_reference, freq_theory_spac, spac_theory_spac, R_matrix_S
