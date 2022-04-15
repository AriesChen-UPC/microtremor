# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/15/2022 2:17 PM
@file: create_colormap.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


# def plot_examples(colormaps):
#     """
#     Helper function to plot data with associated colormap.
#     """
#     np.random.seed(19680801)
#     data = np.random.randn(30, 30)
#     n = len(colormaps)
#     fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
#                             constrained_layout=True, squeeze=False)
#     for [ax, cmap] in zip(axs.flat, colormaps):
#         psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
#         fig.colorbar(psm, ax=ax)
#     plt.show()


jet = cm.get_cmap('jet', 256)
viridis_big = cm.get_cmap('jet', 512)
newcmp = ListedColormap(viridis_big(np.linspace(0.15, 0.85, 256)))
# plot_examples([jet, newcmp])
