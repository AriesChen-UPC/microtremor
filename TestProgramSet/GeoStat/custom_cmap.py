# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/12/2022 6:56 PM
@file: custom_cmap.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from matplotlib.colors import ListedColormap

#%% Custom colormap

np.random.seed(19680801)
# spectrum full bright colormap
# color_dict = ["blue", "royalblue", "cyan", "yellow", "orange", "red"]
color_dict = ["blue", "cyan", "yellow", "red"]
spectrum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', color_dict, N=512)
spectrum_reference_big = cm.get_cmap(spectrum_cmap, 512)
spectrum_cmap_ = ListedColormap(spectrum_reference_big(np.linspace(0.0, 1.0, 256)))
# jet colormap
jet_reference_big = cm.get_cmap('jet', 512)
jet_cmap = ListedColormap(jet_reference_big(np.linspace(0.10, 0.90, 256)))

data = np.random.random((100, 100))
fig, ax = plt.subplots()
plt.subplot(121)
plt.imshow(data, cmap=spectrum_cmap_)
cb_s = plt.colorbar(shrink=0.6, aspect=20, pad=0.05, fraction=0.1)
cb_s.outline.set_visible(False)
plt.subplot(122)
plt.imshow(data, cmap=jet_cmap)
cb_j = plt.colorbar(shrink=0.6, aspect=20, pad=0.05, fraction=0.1)
cb_j.outline.set_visible(False)
plt.show()
fig.savefig('D:/ProjectMaterials/Chengdu_Line8/DataAnalysis/Line08/Test/compareColor.png', format='png',
            bbox_inches='tight', dpi=300, transparent=True)

#%% plot colormap


def custom_bar():
    fig, ax = plt.subplots(figsize=(0.5, 6))
    norm = mpl.colors.Normalize(vmin=200, vmax=800)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=spectrum_cmap_),
        cax=ax,
        orientation="vertical",
        ticks=[200, 300, 400, 500, 600, 700, 800]
    )
    plt.show()


custom_bar()
