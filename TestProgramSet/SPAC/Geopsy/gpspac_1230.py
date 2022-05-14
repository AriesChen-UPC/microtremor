import numpy as np
from scipy.special import j0
import os
import matplotlib.pyplot as plt
import difflib
import tkinter as tk
from tkinter import filedialog
import sys

def gpdc(command,model):
    # temp_out = os.popen(command + ' ' + '-L 5 -R 5 -s frequency -max 200 -n 200 ' + ' < ' + model).readlines()
    temp_out = os.popen(command + ' ' + '-L 5 -R 5 -max 200 -n 200 ' + ' < ' + model).readlines()
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

    zero_line_R =np.zeros(int(max_len_R))
    zero_line_L = np.zeros(int(max_len_L))

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

    return R_matrix_F,L_matrix_F,R_matrix_S,L_matrix_S,zero_line_R,zero_line_L


if __name__ == '__main__':
    # choose gpdc.exe
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
    print('filename:', filename)
    root.destroy()

    chk0 = False
    while not chk0:
        # choose test.model
        root1 = tk.Tk()
        root1.withdraw()
        filename1 = filedialog.askopenfilename()
        print('filename1:', filename1)
        root1.destroy()

        command = filename
        model = filename1
        R_matrix_F,L_matrix_F,R_matrix_S,L_matrix_S,zero_line_R,zero_line_L = gpdc(command,model)

        print('Input 半径')
        r = int(input())
        print('Input 最大频率')
        plt_max = float(input())
        print('展示Mode个数')
        mode_fig= int(input())
        print('SPAC X轴坐标点个数(建议能被最大频率除尽）')
        ticks_num = int(input())

        plt.ion()

        plt.figure(1)
        for i in range(0,mode_fig):
            plt.plot(R_matrix_F[i], 1 / R_matrix_S[i])
        plt.xlim(-0.1, plt_max)
        plt.title('Rayleigh Slow')
        plt.xlabel('Freq(HZ)')
        plt.ylabel('Velocity(m/s)')

        plt.figure(2)
        for i in range(0, mode_fig):
            plt.plot(L_matrix_F[i], 1 / L_matrix_S[i])
        plt.xlim(-0.1, plt_max)
        plt.title('Love Slow')
        plt.xlabel('Freq(HZ)')
        plt.ylabel('Velocity(m/s)')

        plt.figure(3)
        spac_R = j0(r * 2 * np.pi * R_matrix_F[0] * R_matrix_S[0])
        plt.xlim(-0.1, plt_max)
        plt.xticks(np.arange(0,plt_max+plt_max/ticks_num,plt_max/ticks_num))
        plt.plot(R_matrix_F[0], spac_R)
        plt.plot(R_matrix_F[0], zero_line_R, linestyle='--', color='red')
        plt.yticks(np.arange(-0.6, 1.2, 0.2))
        plt.ylim(-0.5, 1.1)
        plt.title('Rayleigh SPAC')
        plt.xlabel('Freq(HZ)')
        plt.ylabel('Autocorrelation Spectrum Ratio')

        plt.figure(4)
        spac_L = j0(r * 2 * np.pi * L_matrix_F[0] * L_matrix_S[0])
        plt.xlim(-0.1, plt_max)
        plt.xticks(np.arange(0, plt_max+plt_max/ticks_num, plt_max/ticks_num))
        plt.plot(L_matrix_F[0], spac_L)
        plt.plot(L_matrix_F[0], zero_line_L, linestyle='--', color='red')
        plt.yticks(np.arange(-0.6, 1.2, 0.2))
        plt.ylim(-0.5, 1.1)
        plt.title('Love SPAC')
        plt.xlabel('Freq(HZ)')
        plt.ylabel('Autocorrelation Spectrum Ratio')

        plt.ioff()

        def function1():
            sys.exit()
            root3.quit()


        def function2():
            plt.close('all')
            global chk0
            chk0 = False
            root3.quit()


        root3 = tk.Tk()
        root3.geometry('600x200')
        tk.Label(root3, text="是否结束程序 \n 选择是则退出程序，选择否则重新开始", font='courier 20').pack(pady=10, padx=10)
        B1 = tk.Button(root3, text="是", command=function1, font='courier 20')
        B1.pack(pady=10, padx=50, side='left')
        B2 = tk.Button(root3, text="否", command=function2, font='courier 20')
        B2.pack(pady=10, padx=50, side='right')
        root3.mainloop()
        root3.destroy()