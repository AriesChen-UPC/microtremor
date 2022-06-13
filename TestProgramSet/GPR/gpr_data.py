# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 6/9/2022 8:48 AM
@file: gpr_data.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import h5py

# select the directory of the data
print('\033[0;31mPlease select the folder of GPR data: \033[0m')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;34mThe path of the GPR data is: %s.\033[0m' % folder_path)
print('------------------------------')
# initialize the global variables
gpr_path = glob(os.path.join(folder_path, '*.h5'))
# read the GPR data using h5py
f = h5py.File(gpr_path[0], 'r')
print(list(f.keys()))
gpr_data_01 = f['swath01']['data']
