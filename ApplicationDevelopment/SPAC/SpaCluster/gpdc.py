import os
import numpy as np
import difflib


def gpdc(command, model):
    temp_out = os.popen(command + ' ' + '-L 5 -R 5 -max 200 -n 200' + ' < ' + model).readlines()
    ind_Love = temp_out.index('# 5 Love dispersion mode(s)\n')
    Rayleigh = temp_out[:ind_Love]
    Love = temp_out[ind_Love:]

    mod_num = 5
    R_name = difflib.get_close_matches('Mode', Rayleigh, mod_num, cutoff=0.3)
    L_name = difflib.get_close_matches('Mode', Love, mod_num, cutoff=0.3)
    R_name.reverse()
    L_name.reverse()

    R_ind = np.zeros(mod_num)
    L_ind = np.zeros(mod_num)

    for i in range(mod_num):
        R_ind[i] = Rayleigh.index(R_name[i])
        L_ind[i] = Love.index(L_name[i])

    length_R = []
    length_L = []
    for j in range(1, len(R_ind)):
        length_R.append(R_ind[j] - R_ind[j - 1] - 1)
        length_L.append(L_ind[j] - L_ind[j - 1] - 1)

    max_len_R = np.max(length_R)
    max_len_L = np.max(length_L)

    R_matrix_F = np.zeros(shape=(len(R_ind), int(max_len_R)))
    L_matrix_F = np.zeros(shape=(len(R_ind), int(max_len_L)))

    R_matrix_S = np.zeros(shape=(len(R_ind), int(max_len_R)))
    L_matrix_S = np.zeros(shape=(len(R_ind), int(max_len_L)))

    for i in range(mod_num):
        if i < mod_num-1:
            R_mode = Rayleigh[int(R_ind[i]) + 1:int(R_ind[i + 1])]
            L_mode = Love[int(L_ind[i]) + 1:int(L_ind[i + 1])]
        else:
            R_mode = Rayleigh[int(R_ind[i]) + 1:]
            L_mode = Love[int(L_ind[i]) + 1:]

        R_Freq = np.array([x.split(' ') for x in R_mode], dtype=float)[:, 0]
        R_Slow = np.array([x.split(' ') for x in R_mode], dtype=float)[:, 1]

        L_Freq = np.array([x.split(' ') for x in L_mode], dtype=float)[:, 0]
        L_Slow = np.array([x.split(' ') for x in L_mode], dtype=float)[:, 1]

        R_matrix_F[i][0:len(R_Freq)] = R_Freq
        L_matrix_F[i][0:len(L_Freq)] = L_Freq

        R_matrix_S[i][0:len(R_Freq)] = R_Slow
        L_matrix_S[i][0:len(L_Freq)] = L_Slow

    return R_matrix_F, L_matrix_F, R_matrix_S, L_matrix_S