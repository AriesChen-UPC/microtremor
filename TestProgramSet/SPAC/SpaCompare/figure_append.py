# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/6/2022 9:14 AM
@file: figure_append.py
"""

import tkinter as tk
from tkinter import filedialog
from glob import glob
from PIL import Image

print('\033[0;31mPlease select the folder of the figure: \033[0m')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;34mThe folder is: %s\033[0m' % folder_path)
figure_path = glob(folder_path + '/*.png')
print('\033[0;34mThe number of figure is: %d\033[0m' % len(figure_path))
# append the gpr figure to the background
background = "D:/ProjectMaterials/Chengdu_Line19/RiskInvestigation/GPR/background.png"
back_figure = Image.open(background, 'r')
for figure in figure_path:
    # foreground = "C:/Users/45834583/Desktop/GPR.png"
    gpr_figure = Image.open(figure, 'r').convert("RGBA")  # convert to RGBA
    append_figure = Image.new('RGBA', (1500, 1125), (0, 0, 0, 0))  # create a new image, size 4:3
    append_figure.paste(back_figure, (0, 0))
    append_figure.paste(gpr_figure, (175, 100), mask=gpr_figure)
    append_figure.save(figure.split('.')[0] + '_bg.png')
print('\033[0;32mDone!\033[0m')
