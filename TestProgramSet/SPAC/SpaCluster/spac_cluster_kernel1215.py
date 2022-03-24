from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, vq
from read_page import read_page
from glob import glob
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Load SPAC
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
root.destroy()

fs = glob(Folderpath + '/*.page')
freq = np.logspace(np.log10(0.5), np.log10(100.1), 306)

# kernal
k1 = np.array([-0.02306931, -0.1056576, 0.13388212, -0.02341026, -0.0126199,
               -0.04117879, -0.02753881, 0.08565462, -0.20602849, 0.20614131,
               -0.14900643, 0.11359798, -0.09662029, 0.05987384, -0.10340957,
               0.1164462, 0.06197933, -0.06145484, -0.27048752, 0.17245993,
               0.17806824, -0.03125308, -0.19900371, 0.11171456, 0.09432922,
               -0.21939862, 0.01255652, -0.32677427, -0.4083756, 0.04717418,
               0.05470994, 0.29249766, 0.33997664, -0.21587674, -0.01858459,
               0.13910776, -0.18480915, 0.07169731, 0.17088659, 0.08077492,
               0.01491521, -0.06912256, -0.06817471, 0.05515498, 0.08658582,
               0.11424139, -0.0416302, 0.04017343, 0.0067895, -0.052115,
               0.07161448, 0.03834576, -0.08015742, 0.05258898, -0.00586048,
               -0.00896439, 0.04149655, 0.00454619, -0.00991248, 0.05561759,
               -0.02137935, -0.03790385, 0.03550394, 0.04025659, -0.03533922,
               0.01554128, 0.00957493, -0.0150793, 0.01997719, -0.00605633,
               -0.01840421, 0.00213903], dtype=np.float32)

k2 = np.array([0.09681424, 0.0125695, -0.03296668, -0.06259093, 0.02641451,
               0.02264782, -0.03527348, -0.01100035, -0.00835153, 0.15201114,
               -0.00604084, -0.06569929, -0.05707217, 0.08184609, -0.03801537,
               0.02860055, 0.02736357, 0.04610762, 0.02690667, 0.00431063,
               -0.05841798, 0.01170933, 0.04377702, 0.03132229, 0.01884418,
               0.00706555, 0.02693376, 0.00865195, -0.08379597, 0.3851489,
               -0.26176456, -0.12142022, 0.13912982, 0.02746409, -0.2967254,
               0.18555762, 1.4003265, 0.48990425, -0.61164373, -0.2229173,
               -0.35163, 0.02921563, -0.14746712, 0.336639, -0.61782324,
               0.08492271, 0.10232697, -0.2795302, 0.2924005, -0.23487124,
               -0.11624496, 0.10293019, -0.06085368, -0.15410152, 0.2654883,
               -0.31394354, 0.05724258, -0.11718074, 0.01887048, 0.06616369,
               -0.01266902, -0.02204721, -0.06348757, -0.0873571, 0.08890857,
               -0.014294, 0.02892015, 0.010807, -0.01736347, -0.04279665,
               -0.02214473, -0.03366138], dtype=np.float32)

names = []
for i in range(len(fs)):
    names.append(Path(fs[i]).stem)

spac = np.zeros(shape=(len(fs), len(freq)))
for i in range(len(fs)):
    print(fs[i])
    temp1 = read_page(fs[i])[0][-1]
    spac[i, :] = np.interp(freq, temp1.freq, temp1.spac)

print('主要关注范围建议选择 2HZ-8HZ， 次要关注范围建议选择 10HZ-15HZ')
min_freq = float(input('Min_freq:\n'))
max_freq = float(input('Max_freq:\n'))

sel = (freq >= min_freq) & (freq <= max_freq)

detail = np.zeros(shape=(len(fs), sel.sum()))
detail1 = np.zeros(shape=(len(fs), len(freq)))
for j in range(len(fs)):
    temp2 = np.convolve(spac[j], k1, 'same')[sel]
    temp3 = np.convolve(spac[j], k2, 'same')[sel]
    temp4 = np.convolve(spac[j], k1, 'same')
    temp5 = np.convolve(spac[j], k2, 'same')
    detail[j:] = temp2 + temp3
    detail1[j:] = temp4 + temp5

n_clusters = int(input('Please input the number of clusters:\n'))
centroid, _ = kmeans(detail, n_clusters)
cluster, _ = vq(detail, centroid)
cluster = cluster.tolist()
print(cluster)

for z in range(n_clusters):
    #print('\n')
    print('Group {:.0f} '.format(z + 1))
    for i in range(len(fs)):
        if cluster[i] == z :
            print(names[i])

plt.figure(1, figsize=(9, 5))
for z in range(n_clusters):
    count = 0
    plt.subplot(2, int(np.ceil(n_clusters / 2)), int(z + 1))
    plt.grid()
    plt.semilogx()
    plt.xlim(2, 100.1)
    plt.xticks([2, 5, 10, 15, 20,40,60,100], [2, 5, 10, 15, 20,40,60,100])
    for k in range(len(fs)):
        if cluster[k] == z:
            count = count + 1
            plt.plot(freq, spac[k])
    plt.axvline(min_freq, color='k')
    plt.axvline(max_freq, color='k')
    plt.title('Group {:.0f} || {:.0f} points'.format(z + 1, count))
plt.suptitle('SPAC cluster')

plt.figure(2, figsize=(9, 5))
for z in range(n_clusters):
    count = 0
    plt.subplot(2, int(np.ceil(n_clusters / 2)), int(z + 1))
    plt.grid()
    plt.semilogx()
    plt.xlim(2,100.1)
    plt.xticks([2, 5, 10, 15, 20,40,60,100], [2, 5, 10, 15, 20,40,60,100])
    for k in range(len(fs)):
        if cluster[k] == z:
            count = count + 1
            plt.plot(freq, detail1[k])
    plt.axvline(min_freq, color='k')
    plt.axvline(max_freq, color='k')
    plt.title('Group {:.0f} || {:.0f} points'.format(z + 1, count))
plt.suptitle('Detail cluster')

plt.show()
