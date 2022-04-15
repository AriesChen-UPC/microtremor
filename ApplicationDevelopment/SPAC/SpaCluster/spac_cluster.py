# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/12/2022 10:23 AM
@file: spac_cluster.py
"""

from sklearn.cluster import KMeans
from read_spac import read_spac
from theory_spac import theory_spac
from spac_kmeans_clustering_plotly import spac_kmeans_clustering_plotly


def spac_cluster():
    """
    spac_cluster: cluster the spac data by kmeans, dbscan and other methods.

    Returns:
        None.

    """
    folder_path, fs, names, spac, freq, spac_fliter, freq_fliter, radius, min_freq, max_freq, freq_len = read_spac()
    vs_reference = int(input('\033[0;36mPlease input the reference Vs(nearly 1Hz, depth 57.5m) of '
                             'initialization model: ...m/s \n''\033[0m'))
    freq_theory_spac, spac_theory_spac = theory_spac(folder_path, radius, vs_reference)
    n_clusters = int(input('\033[0;32mPlease input the number of clusters: 1<=n_clusters<=%s: \n''\033[0m'
                           % str(len(fs))))
    for i in range(99):  # check the clusters number
        if n_clusters <= 0 | n_clusters > len(fs):
            print('\033[0;32mPlease re-input the number of clusters: 1<=n_clusters<=%s: \n''\033[0m' % str(len(fs)))
            n_clusters = int(input())
        else:
            break
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clustering = kmeans.fit(spac_fliter)

    spac_kmeans_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac, min_freq, max_freq,
                           radius, vs_reference, kmeans_clustering, n_clusters)


while True:
    spac_cluster()
    if input('\033[0;31mDo you want to continue? (y/n) \n''\033[0m').upper() == 'Y':
        continue
    else:
        break
input('Press Enter to exit … \n')
