# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 18/10/2021 下午5:09
@file: targetPlot.py
"""
import os
import tkinter as tk
import time
from tqdm import trange
from rich.progress import track
from tkinter import filedialog
from ioGpy import AutocorrTarget
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import matplotlib.pyplot as mp, seaborn

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
filetype = '.target'

# get the FileName
def get_filename(Folderpath, filetype):
    name = []
    final_name = []
    for root, dirs, files in os.walk(Folderpath):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype, ''))
    final_name = [item + '.target' for item in name]
    fileName = [item for item in name]
    return final_name, fileName


# get the specify FileName
FileName, fileName = get_filename(Folderpath, filetype)
# print(FileName)
# print(fileName)
# instance
targetInstance = AutocorrTarget()
# load .target file
# count the numbers of ring
targetInstance.load(Folderpath + '/' + FileName[0])
numRings = len(targetInstance.AutocorrCurves.ModalCurve)
print('\033[0;34mPlease set the number of rings from 1 to ' + str(numRings) + '\033[0m')
# print('Please set the number of rings from 1 to ' + str(numRings))
numRingSelect = int(input())
print('\033[0;33m---------------Loading...---------------\033[0m')
startTime = time.time()
spacData = []
autoCorr = []  # initialization of AutoCorr ratio
for i in trange(len(FileName)):  # todo: try rich
    time.sleep(0.1)
    targetInstance.load(Folderpath + '/' + FileName[i])
    spacData.append(targetInstance.AutocorrCurves.ModalCurve[numRingSelect - 1].RealStatisticalPoint)
    autoCorr.append(spacData[i]['mean'][-251:-1])
    # numRings = len(targetInstance.AutocorrCurves.ModalCurve)
    # for j in range(numRings):
    #     spacData.append(targetInstance.AutocorrCurves.ModalCurve[j].RealStatisticalPoint)

# for j in range(len(spacData)):
#     # autoCorr.append(spacData[j]['mean'][-201:-1])  # read the Autocorr ratio
#     autoCorr.append(spacData[j]['mean'][-251:-1])  # read the Autocorr ratio
autoCorr = DataFrame(autoCorr)  # dataFormat: list to dataframe
autoCorr = autoCorr.T
autoCorr.columns = fileName
autoCorr.to_excel(Folderpath + '/' + 'SPAC_Data.xlsx')  # save to excel
# pandas.DataFrame().corr()
autoCorr_corr = autoCorr.corr()
mask = np.zeros_like(autoCorr_corr)
mask[np.triu_indices_from(mask)] = True
with seaborn.axes_style("white"):
    seaborn.heatmap(autoCorr_corr, mask=mask, annot=True, vmax=1.0, cmap="RdBu_r")
endTime = time.time()
print('\033[0;36m------------------Done!-----------------\n'
      'The time spent is ' + str(endTime - startTime) + '\033[0m')
# seaborn.heatmap(autoCorr_corr, center=0, annot=True, cmap='YlGnBu', vmax=1.0, vmin=0.9)
# seaborn.heatmap(autoCorr_corr, square=True, cmap='RdBu_r')
# mp.show()

# for k in range(len(spacData)):
#     plt.plot(autoCorr.iloc[k])
# plt.xlim([0, 399])
# plt.ylim([-0.6, 1.0])
# plt.grid()
    # plt.plot(autoCorr.iloc[k], label=k)
    # plt.legend()
