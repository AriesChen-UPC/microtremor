# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/17/2022 4:55 PM
@file: cmap_select_default.py: This script is used to select the custom colormap.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import palettes
from PIL import ImageColor


def grayscale_cmap(colors):
    """Return a grayscale version of the given colormap"""

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list('gray', colors, N=512)


if __name__ == '__main__':

    # color bar from MetBrewer
    colors_metbrew = palettes.met_brew(name="Homer1", n=512, brew_type="continuous")
    colors = []
    for color in colors_metbrew:
        colors.append(ImageColor.getcolor(color, "RGBA"))
    colors_rgb = np.array(colors) / 255  # or convert to [0, 1], np.array(colors) / 255.
    colors_gray = grayscale_cmap(np.array(colors) / 255)
    grayscale = colors_gray(np.arange(512))

    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors_rgb], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.show()
