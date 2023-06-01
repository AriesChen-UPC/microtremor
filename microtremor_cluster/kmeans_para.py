# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 11/23/2022 10:47 AM
@File     : kmeans_para.py
@Software : PyCharm
"""

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore")


def kmeans_para(spac_filter, folder_path):

    file_num = len(spac_filter)
    if file_num < 9:
        max_cluster_num = file_num
    else:
        max_cluster_num = 10

    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(spac_filter) for k in range(1, max_cluster_num)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    x = np.arange(1, max_cluster_num)
    y = np.array(inertias)
    kn = KneeLocator(x, y, curve="convex", direction="decreasing")
    cluster_num_k = kn.knee
    if cluster_num_k is None:
        cluster_num_k = 3
    else:
        plt.figure()
        plt.plot(x, y, 'bo-')
        plt.plot(cluster_num_k, kn.knee_y, 'ro--')
        # kn.plot_knee()
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.grid(linestyle='--')
        plt.savefig(folder_path + '/cluster_num_k.png')

    # silhouette score
    silhouette_scores = [silhouette_score(spac_filter, model.labels_) for model in kmeans_per_k[1:]]
    cluster_num_s = 2 + silhouette_scores.index(max(silhouette_scores))
    plt.figure()
    plt.plot(range(2, max_cluster_num), silhouette_scores, "bo-")
    plt.plot(cluster_num_s, max(silhouette_scores), 'ro--')
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel("Silhouette score")
    plt.grid(linestyle='--')
    plt.savefig(folder_path + '/cluster_num_s.png')

    cluster_num = max(cluster_num_k, cluster_num_s)
    print('The cluster number is: ', cluster_num)
    return cluster_num


