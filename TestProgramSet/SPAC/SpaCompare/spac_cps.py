# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/16/2022 3:12 PM
@file: spac_cps.py
"""

from pandas import DataFrame
import numpy as np
from disba import PhaseDispersion
from scipy.special import j0


def spac_cps(model, radius):
    """
    Args:
        model: the ground model, which can be from inversion
        radius: the radius of the spac

    Returns:
        freq: the frequency of the spac
        spac: the spac value which calculated by cps

    """
    layer = []
    with open(model) as f:
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

    model_data = DataFrame(layer).dropna(axis=0)  # model format in Geopsy, first row is layer number and nan
    model_data.columns = ['thickness', 'vp', 'vs', 'density']
    model_data.drop(len(model_data), inplace=True)  # model format in Geopsy, drop the last row, half space
    velocity_model = model_data.to_numpy()
    t = np.logspace(-2, 2.0, 400)  # set the time interval [-2, 2.0], equal freq[1, 100], 400 points
    # Compute the foundmental Rayleigh wave modal dispersion curves
    pdisp = PhaseDispersion(*velocity_model.T)
    cpr = pdisp(t, mode=0, wave="rayleigh")
    freq = 1/cpr.period
    disp_velocity = cpr.velocity
    spac_rayleigh = j0(radius * 2 * np.pi * freq * 1/disp_velocity)
    theory_freq = DataFrame(freq, columns=['freq'])
    theory_spac = DataFrame(spac_rayleigh, columns=['autoCorrRatio'])
    # print('\033[0;32mDispersion curve and SPAC curve calculated by CPS were done!\033[0m')
    return theory_freq, theory_spac
