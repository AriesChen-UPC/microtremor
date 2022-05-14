# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/2/2022 11:50 AM
@file: fig_select.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import shutil
from PIL import Image
import colorama
colorama.init(autoreset=True)

print('\033[0;31mThe folder path example is :\033[0m')
print('Please select the figure file folder：')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;31mFolderPath\033[0m:', folder_path)
fig_save_path = os.path.abspath(os.path.dirname(folder_path)) + '/Attachment'
hv_fig_save_path = fig_save_path + '/HVSR/Original'
spac_fig_save_path = fig_save_path + '/SPAC/Original'
if not os.path.exists(fig_save_path):
    os.mkdir(fig_save_path)
if not os.path.exists(hv_fig_save_path):
    os.makedirs(hv_fig_save_path)
if not os.path.exists(spac_fig_save_path):
    os.makedirs(spac_fig_save_path)
sub_path_name = os.listdir(folder_path)
sub_path = []
for i in range(len(sub_path_name)):
    sub_path.append(folder_path + '/' + sub_path_name[i])
    try:
        hv_fig = glob(sub_path[i] + '/*hv*.png')
        if len(hv_fig) > 1:
            shutil.copy(hv_fig[0], hv_fig_save_path)
        else:
            shutil.copy(hv_fig[0], hv_fig_save_path)
    except IndexError:
        pass
    try:
        spac_fig = glob(sub_path[i] + '/*spacfig*.png')
        if len(spac_fig) > 1:
            shutil.copy(spac_fig[0], spac_fig_save_path)
        else:
            shutil.copy(spac_fig[0], spac_fig_save_path)
    except IndexError:
        pass

#%% resize the figure


def resize_fig(fig_input, output_dir, width, height):
    img = Image.open(fig_input)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(output_dir, os.path.basename(fig_input)))
    except Exception as e:
        print(e)


resize_hv_fig_path = os.path.abspath(os.path.dirname(hv_fig_save_path)) + '/Resize'
if not os.path.exists(resize_hv_fig_path):
    os.mkdir(resize_hv_fig_path)
for fig in glob(hv_fig_save_path + '/*.png'):
    resize_fig(fig, resize_hv_fig_path, 450, 300)

resize_spac_fig_path = os.path.abspath(os.path.dirname(spac_fig_save_path)) + '/Resize'
if not os.path.exists(resize_spac_fig_path):
    os.mkdir(resize_spac_fig_path)
for fig in glob(spac_fig_save_path + '/*.png'):
    resize_fig(fig, resize_spac_fig_path, 900, 300)

print('\033[0;32m------------------------Done!------------------------\033[0m')

