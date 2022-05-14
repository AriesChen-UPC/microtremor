# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/15/2022 7:40 PM
@file: geopsy_spac.py
"""
import tkinter
from tkinter import filedialog

import numpy as np
import pandas as pd
from scipy.special import j0
from gpdc import gpdc


def geopsy_spac(radius, model):
    # print('\033[0;31mPlease select the gpdc.exe from Geopsy package ... \033[0m')
    # root = tkinter.Tk()
    # root.withdraw()
    # command = filedialog.askopenfilename()
    command = 'D:/MyProject/Geopsy/geopsypack-win64-3.4.2/bin/gpdc.exe'
    R_matrix_F, L_matrix_F, R_matrix_S, L_matrix_S = gpdc(command, model)
    R_matrix_F = pd.DataFrame(R_matrix_F).T
    L_matrix_F = pd.DataFrame(L_matrix_F).T
    R_matrix_S = pd.DataFrame(R_matrix_S).T
    L_matrix_S = pd.DataFrame(L_matrix_S).T
    # the foundmental model of Rayleigh and Love waves
    spac_R = j0(radius * 2 * np.pi * R_matrix_F[0] * R_matrix_S[0])
    spac_L = j0(radius * 2 * np.pi * L_matrix_F[0] * L_matrix_S[0])
    return R_matrix_F[0], L_matrix_F[0], spac_R, spac_L, R_matrix_S[0]
