# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 6/10/2022 3:10 PM
@file: model2table.py
"""

import os
import time
import tkinter
from glob import glob
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def read_model(model, position):
    """
    Read the model from inversion
    Args:
        model: the model from inversion
        position: the position of the model

    Returns:
        model_data: the data of model

    """
    layer = []
    with open(model) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                each_layer = [float(param) for param in line.strip().split(' ') if param != '']
                layer.append(each_layer)

    model_data = pd.DataFrame(layer).dropna(axis=0)  # model format in Geopsy
    model_data.columns = ['thickness', 'vp', 'vs', 'density']
    model_data.loc[0] = [0, 350, 150, 2000]  # set the first layer
    model_data.sort_index(inplace=True)
    model_data.drop([len(model_data)-1], inplace=True)
    model_data['depth'] = model_data['thickness'].cumsum()
    model_data['position'] = position
    return model_data


print('\033[0;31mPlease select the model from inversion ... \033[0m')
root = tkinter.Tk()
root.withdraw()
file_path = filedialog.askdirectory()
print('\033[0;34mThe path of the model is: %s.\033[0m' % file_path)
model_file = glob(file_path + "/*.txt")
print('\033[0;34mThe number of the model is: %d.\033[0m' % len(model_file))
# read the model from inversion
data = pd.DataFrame(columns=['thickness', 'vp', 'vs', 'density', 'depth', 'position'])
for model in model_file:
    model_data = read_model(model, float(os.path.basename(model).split('.')[0]))
    data = data.append(model_data, ignore_index=True)

x = data['position'].to_numpy()
y = data['depth'].to_numpy()
vs = data['vs'].to_numpy()
sns.histplot(data=data, x="vs", kde=True)
plt.show()

#%% Kriging

time_start = time.time()
# build the target grid
gridx = np.arange(x.min(), x.max(), 0.1)
gridy = np.arange(y.min(), y.max(), 0.1)
# Ordinary Kriging
print("\033[0;36mThe current method is: Ordinary Kriging.\033[0m")
OK = OrdinaryKriging(x, y, vs, variogram_model="spherical", nlags=30, anisotropy_scaling=3.0, enable_plotting=True,
                     enable_statistics=True, exact_values=False, pseudo_inv=True)
vs_, ss = OK.execute("grid", gridx, gridy)

color_dict = ["blue", "cyan", "yellow", "red"]
spectrum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', color_dict, N=512)
spectrum_reference_big = cm.get_cmap(spectrum_cmap, 512)
spectrum_cmap_ = ListedColormap(spectrum_reference_big(np.linspace(0.0, 1.0, 256)))

fig, axes = plt.subplots(figsize=(len(gridx) / 10, len(gridy) / 10))
plt.imshow(vs_, cmap=spectrum_cmap_, extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()], origin='lower')
plt.xticks(fontsize=60, fontweight="bold")
plt.yticks(fontsize=60, fontweight="bold")
plt.gca().invert_yaxis()
cb = plt.colorbar(shrink=0.8)
cb.ax.invert_yaxis()
cb.ax.tick_params(labelsize=60)
plt.show()
fig_save_name = file_path + '/' + os.path.split(file_path)[1].split('.')[0] + '.png'
fig.savefig(fig_save_name, format='png', bbox_inches='tight', dpi=96, transparent=True)
