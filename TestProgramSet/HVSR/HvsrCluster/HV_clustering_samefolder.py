import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, vq
from glob import glob
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


# Load HV
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
root.destroy()

fs = glob(Folderpath + '/*.hv')

freq = np.geomspace(0.1,100.1,400)

min_freq = float(input('Min_freq:\n'))
max_freq = float(input('Max_freq:\n'))

sel = (freq >= min_freq) & (freq <= max_freq)
freq_sel =freq[sel]

data = []
data_sel = np.zeros(shape=(len(fs), len(freq_sel)))
data2 = np.zeros(shape=(len(fs), len(freq)))
for i in range(len(fs)):
    data.append( pd.read_table(filepath_or_buffer= fs[i],sep='\t',skiprows= 9,names=['Freq','Aver','max','min']))
    data2[i] = np.interp(freq, data[i].Freq, data[i].Aver)
    data_sel[i] = data2[i][sel]

names = []
for i in range(len(fs)):
    names.append(Path(fs[i]).stem)

n_clusters = int(input('Please input the number of clusters:\n'))
centroid, _ = kmeans(data_sel, n_clusters)
result1, _ = vq(data_sel, centroid)
result1 = result1.tolist()
print(result1)


for z in range(n_clusters):
    count = 0
    plt.subplot(2, int(np.ceil(n_clusters / 2)), int(z + 1))
    plt.grid()
    plt.semilogx()
    plt.ylim(0, data_sel.max()+1)
    plt.xlim(1, 20.1)
    # plt.xticks([2, 5, 10, 15, 20,40,60,100], [2, 5, 10, 15, 20,40,60,100])
    plt.xticks([2, 5, 10, 15, 20], [2, 5, 10, 15, 20])
    for k in range(len(fs)):
        if result1[k] == z:
            count = count + 1
            plt.plot(freq, data2[k])
    plt.axvline(min_freq, color='k')
    plt.axvline(max_freq, color='k')
    plt.title('Group {:.0f} || {:.0f} points'.format(z + 1, count))
plt.suptitle('HV cluster')

for z in range(n_clusters):
    print('Group {:.0f} '.format(z + 1))
    for i in range(len(fs)):
        if result1[i] == z:
            print(names[i])
plt.show()
