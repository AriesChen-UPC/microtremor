# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 8:14 AM
@file: spac_clustering.py
"""

from read_spac import read_spac
from theory_spac import theory_spac
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from spac_clustering_plotly import spac_clustering_plotly


def spac_cluster():
    folder_path, fs, names, spac, freq, spac_fliter, freq_fliter, radius, min_freq, max_freq, freqLen = read_spac()
    vs_reference = int(input('\033[0;36mPlease input the reference Vs(nearly 1Hz, depth 57.5m) of initialization model: '
                             '...m/s \n''\033[0m'))
    freq_theory_spac, spac_theory_spac = theory_spac(folder_path, radius, vs_reference)
    n_clusters = int(input('\033[0;36mPlease input the number of clusters: 1<=n_clusters<=%s: \n''\033[0m' % str(len(fs))))
    for i in range(99):  # check the clusters number
        if n_clusters <= 0 | n_clusters > len(fs):
            print('\033[0;32mPlease input the number of clusters: 1<=n_clusters<=%s: \n''\033[0m' % str(len(fs)))
            n_clusters = int(input())
        else:
            break
    # kmeans clustering
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(spac_fliter)
    # DBSCAN clustering
    # clustering = DBSCAN().fit(spac_fliter)  # fixme: DBSCAN clustering
    spac_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac, min_freq, max_freq,
                           clustering, n_clusters)


while True:
    spac_cluster()
    if input('\033[0;31mDo you want to continue? (y/n) \n''\033[0m') == 'y':
        continue
    else:
        break
input('Press Enter to exit … \n')
