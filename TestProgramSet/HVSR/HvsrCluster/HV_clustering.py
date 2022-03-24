from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster.vq import kmeans, vq

# load parameters
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
root.destroy()

p = Path(Folderpath)
pdir = [x for x in p.iterdir() if x.is_dir()]

fs = []
for idir in pdir:
    fs.append(list(idir.glob('*.hv')))

data = list()
freq = list()
for i in range(len(pdir)):
    data0 = list()
    freq0 = list()
    for fi in fs[i]:
        data0.append(pd.read_table(filepath_or_buffer=fi, sep='\t', skiprows=9,
                                   names=['Freq', 'Aver', 'min_data', 'max_data']).Aver)
        freq0.append(pd.read_table(filepath_or_buffer=fi, sep='\t', skiprows=9,
                                   names=['Freq', 'Aver', 'min_data', 'max_data']).Freq)
    data.append(data0)
    freq.append(freq0[0])

freq_std = np.geomspace(0.1, 20.1, 200)  # todo: range of frequency

min_freq = float(input('Min_freq:\n'))
max_freq = float(input('Max_freq:\n'))

sel = (freq_std >= min_freq) & (freq_std <= max_freq)
freq_sel = freq_std[sel]

mean_data = list()
mean_data1 = np.zeros(shape=(len(pdir), len(freq_std)))
mean_data_sel = np.zeros(shape=(len(pdir), len(freq_sel)))
for i in range(len(pdir)):
    mean_data.append(np.mean(data[i], 0))
    mean_data1[i] = np.interp(freq_std, freq[i], mean_data[i])
    mean_data_sel[i] = mean_data1[i][sel]

n_clusters = int(input('Please input the number of clusters:\n'))
centroid, _ = kmeans(mean_data_sel, n_clusters)
result1, _ = vq(mean_data_sel, centroid)
result1 = result1.tolist()
print(result1)

for z in range(n_clusters):
    count = 0
    plt.subplot(2, int(np.ceil(n_clusters / 2)), int(z + 1))
    plt.grid()
    plt.semilogx()
    plt.ylim(0, mean_data_sel.max() + 1)
    plt.xlim(1, 20.1)  # todo: range of xlim
    # plt.xticks([2, 5, 10, 15, 20, 40, 60, 100], [2, 5, 10, 15, 20, 40, 60, 100])
    plt.xticks([1, 2, 5, 10, 15, 20], [1, 2, 5, 10, 15, 20])
    for k in range(len(fs)):
        if result1[k] == z:
            count = count + 1
            plt.plot(freq_std, mean_data1[k])
    plt.axvline(min_freq, color='k')
    plt.axvline(max_freq, color='k')
    plt.title('Group {:.0f} || {:.0f} points'.format(z + 1, count))
plt.suptitle('HV cluster')


names = []
for i in range(len(pdir)):
    names.append(Path(pdir[i]).stem)

for z in range(n_clusters):
    print('Group {:.0f} '.format(z + 1))
    for i in range(len(fs)):
        if result1[i] == z:
            print(names[i])
plt.show()
