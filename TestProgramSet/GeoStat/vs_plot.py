# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 9/15/2022 1:30 PM
@file: vs_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import cm, patches, colors
from matplotlib.colors import ListedColormap
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import warnings
mpl.rc('font', family='SimHei', weight='bold')
warnings.filterwarnings("ignore")


def vs_plot(data, nlags=6, weight=True, anisotropy_scaling=1.0, anisotropy_angle=0.0, min_color=0.0, max_color=1.0,
            file_path=os.getcwd()):
    """
    Args:
        data: vs data in Dataframe format, with 'position','depth','vs' columns
        nlags: Number of average bins for the semivariogram, default value is 6
        weight: The weigh scheme for the semivariogram, default is True
        anisotropy_scaling: The scale of the anisotropy, default value is 1.0
        anisotropy_angle: The angle of the anisotropy, default value is 0.0
        min_color: The minimum color limit of the colormap, default value is 0.0
        max_color: The maximum color limit of the colormap, default value is 1.0
        file_path: The path for storing the figure

    Returns:
        None

    """
    # load the vs data, check if the x distance(position) is too large
    is_interpolation = False
    data_check = data[['position', 'depth', 'vs']].dropna(subset=['vs'])
    x_check = pd.DataFrame(data_check['position'].drop_duplicates().reset_index(drop=True))
    x_distance = np.diff(data_check['position'].drop_duplicates())
    x_distance_value = []
    x_index = []
    x_value = []
    x_inter_value = []
    for i in range(len(x_distance)):
        if x_distance[i] >= 20:
            is_interpolation = True
            x_distance_value.append(x_distance[i])
            x_index.append(i)
            x_value.append(x_check.loc[i]['position'])
    if is_interpolation:
        x_insert = pd.DataFrame(columns=['position', 'depth', 'vs'])
        for j in range(len(x_distance_value)):
            x_temp = data_check[data_check['position'] == x_value[j]]
            x_temp['position'].replace(x_value[j], x_value[j] + x_distance_value[j] / 2, inplace=True)
            x_inter_value.append(x_value[j] + x_distance_value[j] / 2)
            x_insert = x_insert.append(x_temp)
        data_insert = pd.concat([data_check, x_insert], axis=0, ignore_index=True)
        data_plot = data_insert
        print('The max distance between two points is: %d' % max(x_distance))
        print('Interpolation has been performed, %d points were added.' % len(x_distance_value))
    else:
        data_plot = data[['position', 'depth', 'vs']].dropna(subset=['vs'])
        print('No interpolation has been performed automatically.')
    x = data_plot['position'].to_numpy()
    y = data_plot['depth'].to_numpy()
    vs = data_plot['vs'].to_numpy()
    # build the grid, with space 0.1m
    gridx = np.arange(x.min(), x.max(), 0.1)
    gridy = np.arange(y.min(), y.max(), 0.1)
    # Ordinary Kriging
    print("The current method is: Ordinary Kriging.")
    OK = OrdinaryKriging(x, y, vs, variogram_model="spherical", nlags=nlags, weight=weight,
                         anisotropy_scaling=float(anisotropy_scaling), anisotropy_angle=float(anisotropy_angle),
                         exact_values=False, pseudo_inv=True)
    vs_, ss = OK.execute("grid", gridx, gridy)
    # plot the result of Kriging
    fig, axes = plt.subplots(1, 1, figsize=(len(gridx) / 10, len(gridy) / 10))  # define the fig size
    fig_font_size = int((len(gridx) / 10 + len(gridy) / 10) / 2) + 20
    # manually define the color scale range
    min_color_index = min_color
    max_color_index = max_color
    # define my own colormap
    color_dict = ["blue", "cyan", "yellow", "red"]
    spectrum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', color_dict, N=512)
    spectrum_reference_big = cm.get_cmap(spectrum_cmap, 512)
    spectrum_cmap_ = ListedColormap(spectrum_reference_big(np.linspace(0.0, 1.0, 256)))
    mycmap = ListedColormap(spectrum_reference_big(np.linspace(min_color_index, max_color_index, 256)))
    # imshow
    plt.imshow(vs_, cmap=mycmap, zorder=20)
    # set the x, y ticks information
    if 0 < x.max() - x.min() <= 50:
        if len(gridx) % 50 == 0:
            x_ticks = [i for i in range(0, len(gridx) + 50, 50)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 5, 5)]
            x_labels = [i[-3:] for i in x_label_mile]
        else:
            x_ticks = [i for i in range(0, len(gridx), 50)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 5)]
            x_labels = [i[-3:] for i in x_label_mile]
    elif 50 < x.max() - x.min() <= 150:
        if len(gridx) % 100 == 0:
            x_ticks = [i for i in range(0, len(gridx) + 100, 100)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 10, 10)]
            x_labels = [i[-3:] for i in x_label_mile]
        else:
            x_ticks = [i for i in range(0, len(gridx), 100)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 10)]
            x_labels = [i[-3:] for i in x_label_mile]
    else:
        if len(gridx) % 200 == 0:
            x_ticks = [i for i in range(0, len(gridx) + 200, 200)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10) + 20, 20)]
            x_labels = [i[-3:] for i in x_label_mile]
        else:
            x_ticks = [i for i in range(0, len(gridx), 200)]
            x_label_mile = ['%03d' % (i + x.min() % 1000) for i in range(0, int(len(gridx) / 10), 20)]
            x_labels = [i[-3:] for i in x_label_mile]
    # set the x label information
    try:
        line_number = data['里程编号'][data['里程编号'].notnull()].unique()[0]
    except KeyError:
        line_number = 'Line'
    kilo_meter = np.unique(x // 1000)
    if len(kilo_meter) == 1:
        line_kilo_num = line_number + str(int(kilo_meter[0])) + '+'
    elif len(kilo_meter) == 2:
        line_kilo_num = line_number + str(int(kilo_meter[0])) + '/' + str(int(kilo_meter[1])) + '+'
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_labels, fontsize=fig_font_size)
    axes.tick_params(axis='x', which='major', pad=25, length=20, width=2)
    axes.xaxis.set_ticks_position("bottom")
    axes.set_xlabel('里程区间' + '(' + line_kilo_num + ')', fontsize=fig_font_size, labelpad=20, fontweight="bold")
    y_ticks = [i for i in range(0, len(gridy) + 10, 50)]
    y_labels = [int(i / 10) for i in y_ticks]
    y_labels.sort(reverse=True)
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_labels, fontsize=fig_font_size)
    axes.tick_params(axis='y', direction='in', length=20, width=2)
    axes.set_ylabel('深度(米)', fontsize=fig_font_size, fontweight="bold")
    # set the colorbar using my own colormap spectrum_cmap_
    norm = mpl.colors.Normalize(vmin=200, vmax=800)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=spectrum_cmap_),
                      ax=axes, shrink=0.7, pad=0.005, aspect=20)
    cb.outline.set_visible(False)
    cb.set_label('横波速度(米/秒)', fontsize=fig_font_size, fontweight="bold")
    cb.ax.tick_params(labelsize=fig_font_size, rotation=90, length=20, width=2)
    cb.ax.invert_yaxis()
    # plot the metro lines
    try:
        metro_x = (data['隧道里程'].dropna() - x.min()) / 0.1
        metro_y_up = len(gridy) - data['隧道顶面'].dropna() / 0.1
        metro_y_down = len(gridy) - (data['隧道顶面'].dropna() + data['隧道范围'].dropna()[0]) / 0.1
        plt.plot(metro_x, metro_y_up, metro_x, metro_y_down, color='black', linewidth=5.0, zorder=30)
    except KeyError:
        pass
    # plot the microtremor points
    points_label = data['position'].dropna().drop_duplicates().reset_index(drop=True)
    try:
        is_real = data['isReal'].dropna().reset_index(drop=True)
        if 0 in is_real:
            is_man_interpolation = True
        points_all = pd.concat([points_label, is_real], axis=1)
        points_real = points_all[points_all['isReal'] == 1].reset_index(drop=True)
        points_unreal = points_all[points_all['isReal'] == 0].reset_index(drop=True)
        point_x_real = points_real['position'].values
    except KeyError:
        is_man_interpolation = False
        point_x_real = points_label.values
    points_x = (point_x_real - point_x_real.min()) / 0.1
    points_y = np.ones(len(points_x)) * (len(gridy) + 5)
    label_size = len(gridx) if len(gridx) > len(gridy) else len(gridy)  # set the size of the label
    plt.scatter(points_x, points_y, marker='v', c='c', s=label_size, zorder=10)
    # plot mask for interpolation points
    if is_interpolation:
        if is_man_interpolation:
            for k in range(len(points_unreal)):
                x_inter_value.append(points_unreal['position'].values[k])
    else:
        if is_man_interpolation:
            for k in range(len(points_unreal)):
                x_inter_value.append(points_unreal['position'].values[k])
    if x_inter_value:
        for value in x_inter_value:
            mask_value = (value - x.min()) / 0.1 - 30
            axes.add_patch(patches.Rectangle((mask_value, 0), 60, len(gridy), color='white', alpha=0.3, zorder=40))
    # set the border of the figure
    axes.set_xlim((-5, len(gridx) + 5))
    axes.set_ylim((0, len(gridy) + 10))
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    plt.show()
    fig_save_name = file_path + '/' + os.path.split(file_path)[1].split('.')[0] + '.png'
    fig.savefig(fig_save_name, format='png', bbox_inches='tight', dpi=96, transparent=True)
    print('Done!\n')
