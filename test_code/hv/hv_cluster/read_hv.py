import io
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
filetype = '.hv'


def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.hv' for item in name]
    return final_name


def read_hv(file):
    with open(file, encoding='utf-8') as f0:
        tmp = f0.readlines()
    head = [x for x in tmp if x[0] == '#']
    tab = [x for x in tmp if x[0] != '#']
    keys = ['geo_version', 'num_window', 'f0', 'num_window_f0', 'position', 'category', 'tab_head']
    h0 = dict.fromkeys(keys)
    h0['geo_version'] = float(head[0].split(' ')[-1])
    h0['num_window'] = int(head[1].split('=')[-1])
    h0['f0'] = float(head[2].split('\t')[-1])
    h0['num_window_f0'] = int(head[3].split('=')[-1])
    h0['postion'] = [float(x) for x in head[6].split('\t')[-1].split(' ')]
    h0['category'] = head[7].split('\t')[-1][:-1]
    h0['tab_head'] = head[8][2:].split('\t')
    t0 = np.loadtxt(io.StringIO(''.join(tab)))
    return h0, t0


FileName = get_filename(Folderpath, filetype)
print(FileName)

h, t = read_hv(FileName)


import glob
hv_list = glob.glob(Folderpath + "/*.hv")
print(hv_list)

