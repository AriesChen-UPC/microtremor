# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 8:14 AM
@file: spac_cluster_methods.py
"""

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from cps_disp import cps_disp
from disp_curve_CPS import disp_curve_CPS
from geopsy_disp import geopsy_disp
from inversion_model import inversion_model
from read_spac import read_spac
from theory_spac import theory_spac
from spac_kmeans_clustering_plotly import spac_kmeans_clustering_plotly
from spac_dbscan_clustering_plotly import spac_dbscan_clustering_plotly
from spac_gm_clustering_plotly import spac_gm_clustering_plotly
from spac_clf_clustering_plotly import spac_clf_clustering_plotly
from spac_sc_clustering_plotly import spac_sc_clustering_plotly
from plot_dendrogram import plot_dendrogram

#%% prepare the spac data

folder_path, fs, names, spac, freq, spac_fliter, freq_fliter, radius, min_freq, max_freq, freq_len = read_spac()

#%% calculate the theory spac

spac_method = int(input('\033[0;31mPlease choose the method to calculate the spac: [1] eIndex [2] Geopsy'
                        ' [3] CPS \n\033[0m'))
if spac_method == 1:
    # using the method of eIndex function
    vs_reference = int(input('\033[0;36mPlease input the reference Vs(nearly 1Hz, depth 57.5m) of '
                             'initialization model: ...m/s \n''\033[0m'))
    freq_theory_spac, spac_theory_spac = theory_spac(folder_path, radius, vs_reference)
if spac_method == 2:
    # using the result of Geopsy
    model_plot, vs_reference, freq_theory_spac, spac_theory_spac, R_matrix_S = inversion_model(radius)
    geopsy_disp(folder_path, model_plot, freq_theory_spac, R_matrix_S)
if spac_method == 3:
    model_plot, vs_reference, freq_theory_spac, spac_theory_spac, R_matrix_S = disp_curve_CPS(radius)
    cps_disp(folder_path, model_plot, freq_theory_spac, R_matrix_S)

#%% kmeans clustering

kmeans_clusters = int(input('\033[0;32mPlease input the number of clusters: 1 <= clusters <= %s: \n''\033[0m'
                            % str(len(fs))))
for i in range(99):  # check the clusters number
    if kmeans_clusters <= 0 | kmeans_clusters > len(fs):
        print('\033[0;32mPlease re-input the number of clusters: 1 <= clusters <= %s: \n''\033[0m' % str(len(fs)))
        kmeans_clusters = int(input())
    else:
        break
kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
kmeans_clustering = kmeans.fit(spac_fliter)

spac_kmeans_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac,
                              min_freq, max_freq, radius, vs_reference, kmeans_clustering, kmeans_clusters)

#%% kmeans parameters test

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(spac_fliter) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]  # finding the optimal number of clusters

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.title("The Elbow Method showing the optimal k", fontsize=14)
plt.show()

# silhouette score
silhouette_score(spac_fliter, kmeans.labels_)
silhouette_scores = [silhouette_score(spac_fliter, model.labels_) for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.show()

# silhouette analysis plot
plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(spac_fliter, y_pred)

    padding = len(spac_fliter) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")

    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

plt.show()

#%% DBSCAN clustering

dbscan = DBSCAN(eps=0.20, min_samples=5)  # todo: DBSCAN clustering parameters, eps and min_samples
dbscan_clustering = dbscan.fit(spac_fliter)

spac_dbscan_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac,
                              min_freq, max_freq, radius, vs_reference, dbscan_clustering)

#%% GaussianMixture clustering

gm_components = int(input('\033[0;32mPlease input the number of components: 1 <= components <= %s: \n''\033[0m'
                          % str(len(fs))))
for i in range(99):  # check the clusters number
    if gm_components <= 0 | gm_components > len(fs):
        print('\033[0;32mPlease re-input the number of clusters: 1 <= components <= %s: \n''\033[0m' % str(len(fs)))
        gm_components = int(input())
    else:
        break
gm = GaussianMixture(n_components=gm_components, random_state=42).fit(spac_fliter)
gm_clustering = gm.predict(spac_fliter)

spac_gm_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac, min_freq, max_freq,
                          radius, vs_reference, gm_clustering, gm_components)

#%% IsolationForest

clf = IsolationForest(random_state=42).fit(spac_fliter)
clf_clustering = clf.predict(spac_fliter)

spac_clf_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac, min_freq, max_freq,
                           radius, vs_reference, clf_clustering)

#%% Hierarchical Clustering

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(spac_fliter)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#%% Spectral Clustering

sc_clusters = int(input('\033[0;32mPlease input the number of clusters: 1 <= clusters <= %s: \n''\033[0m'
                       % str(len(fs))))
for i in range(99):  # check the clusters number
    if sc_clusters <= 0 | sc_clusters > len(fs):
        print('\033[0;32mPlease re-input the number of clusters: 1 <= clusters <= %s: \n''\033[0m' % str(len(fs)))
        sc_clusters = int(input())
    else:
        break
sc_clustering = SpectralClustering(n_clusters=sc_clusters, assign_labels='discretize', random_state=42).fit(spac_fliter)

spac_sc_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac,
                              min_freq, max_freq, radius, vs_reference, sc_clustering, sc_clusters)
