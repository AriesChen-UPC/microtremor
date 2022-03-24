# -*- coding: utf-8 -*-
"""
Created on 05-13-2021
@author: LijiongChen
This code is designed to read the specify FileName.
"""

import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
filetype = '.csv'


def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.csv' for item in name]
    return final_name


FileName = get_filename(Folderpath, filetype)
print(FileName)
# get the specify FileName
SpecifyFileName = []
for i in range(len(FileName)):
    if "gpy_spac[" in FileName[i]:  # "gpy_spac[" is the character
        SpecifyFileName.append(FileName[i])
print(SpecifyFileName)

