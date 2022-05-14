# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/10/2022 3:00 PM
@file: gstools_example.py
"""

import tkinter
from tkinter import filedialog
import numpy as np
import gstools as gs
import pandas as pd
from matplotlib import pyplot as plt

print('\033[0;31mPlease select the data(.xls, .xlsx) for estimation ...\033[0m')
root = tkinter.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print('\033[0;32mFile path is: %s.\033[0m' % file_path)
data = pd.read_excel(file_path)  # data stored in title 'x',''y',''vs'. 'x','y' are coordinates, 'vs' is the value
# data.dropna(subset=['vs'], inplace=True)
data_plot = data.dropna(subset=['vs'])  # TODO: drop the rows with NaN in 'vs' column
x = data_plot['x'].to_numpy()
y = data_plot['y'].to_numpy()
vs = data_plot['vs'].to_numpy()

# estimate the variogram of the field
bins = np.arange(40)
bin_center, gamma = gs.vario_estimate((x, y), vs, bins)
models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    "Stable": gs.Stable,
    "Rational": gs.Rational,
    "Cubic": gs.Cubic,
    "Linear": gs.Linear,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
    "HyperSpherical": gs.HyperSpherical,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
    "TPLGaussian": gs.TPLGaussian,
    "TPLExponential": gs.TPLExponential,
    "TPLStable": gs.TPLStable,
    "TPLSimple": gs.TPLSimple
}
scores = {}
# plot the estimated variogram
plt.scatter(bin_center, gamma, color="k")
ax = plt.gca()

# fit all models to the estimated variogram
for model in models:
    fit_model = models[model](dim=2)
    try:  # FIXME: RuntimeError: Optimal parameters not found: The maximum number of function evaluations is exceeded.
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, max_eval=10000)
        ax.plot(bin_center, fit_model.variogram(bin_center), label=model)
        scores[model] = r2
        print('\033[0;32mFitted %s model.\033[0m' % model)
    except RuntimeError:
        print('\033[0;31m%s model failed.\033[0m' % model)
ax.legend(ncol=2)
ax.set_xlabel('Distance')
ax.set_ylabel('Variogram')
plt.show()

ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
print("RANKING by Pseudo-r2 score")
for i, (model, score) in enumerate(ranking, 1):
    print(f"{i:>6}. {model:>15}: {score:.5}")
