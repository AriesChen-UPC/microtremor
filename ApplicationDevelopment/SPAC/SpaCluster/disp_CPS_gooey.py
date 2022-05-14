# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/16/2022 3:12 PM
@file: disp_curve_CPS.py
"""

from pandas import DataFrame
import numpy as np
from disba import PhaseDispersion
from scipy.special import j0


def disp_CPS_gooey(radius, model):
    model = model
    layer = []
    with open(model) as f:  # todo: read the model file with datatable fread
                            # fixme: when read excel file using datatable, xlrd and datatable version is not compatible
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                if '\t' in line:  # model file from Excel, split by tab
                    each_layer = [float(param) for param in line.strip().split('\t')]
                    layer.append(each_layer)
                else:  # model file from Geopsy, split by space
                    each_layer = [float(param) for param in line.strip().split(' ') if param != '']
                    layer.append(each_layer)

    model_data = DataFrame(layer).dropna(axis=0)  # model format in Geopsy, first row is layer number
    model_data.columns = ['thickness', 'vp', 'vs', 'density']

    model_plot = DataFrame(index=np.arange(10), columns=np.arange(2))
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

    model_data.drop(len(model_data), inplace=True)  # drop the last row, half space
    velocity_model = model_data.to_numpy()
    t = np.logspace(-2, 2.0, 400)  # set the time interval [-2, 2.0], equal freq[1, 100], 400 points

    # Compute the foundmental Rayleigh wave modal dispersion curves
    pdisp = PhaseDispersion(*velocity_model.T)
    cpr = pdisp(t, mode=0, wave="rayleigh")

    freq = 1/cpr.period
    disp_velocity = cpr.velocity
    spac_R = j0(radius * 2 * np.pi * freq * 1/disp_velocity)
    freq_theory_spac = DataFrame(freq, columns=['freq'])
    spac_theory_spac = DataFrame(spac_R, columns=['autoCorrRatio'])
    R_matrix_S = DataFrame(1/disp_velocity, columns=['rayleigh'])
    return model_plot, vs_reference, freq_theory_spac, spac_theory_spac, R_matrix_S