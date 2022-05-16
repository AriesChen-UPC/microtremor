# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/14/2022 1:30 PM
@file: fliter_target.py
"""

import pandas as pd
import os
import shutil
import tkinter as tk
from glob import glob
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
sac_orignal_folder_path = filedialog.askdirectory()
target_name = glob(sac_orignal_folder_path + '/*.target')
fliter_path = sac_orignal_folder_path + '/fliter_target'
os.makedirs(fliter_path)

fliter_name = pd.read_csv("C:/Users/45834583/Desktop/Worksheet.csv")
for i in range(len(target_name)):
    if os.path.basename(target_name[i].split('.')[0]) in fliter_name['Points'].values:
        shutil.move(target_name[i], fliter_path)
