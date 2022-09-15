# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 9/15/2022 1:30 PM
@file: model_pre.py
"""

import os
from glob import glob
import pandas as pd


def model_pre(file_path):
    """
    Prepare model data for kriging VS plot
    Args:
        file_path: the model files' path from inversion

    Returns:
        data: all the model files' data, storing in DataFrame format

    """
    def read_model(model, position):
        """
        Read the model file from inversion in Geopsy format
        Args:
            model: the model file path
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

        model_data = pd.DataFrame(layer).dropna(axis=0)
        model_data.columns = ['thickness', 'vp', 'vs', 'density']
        model_data.loc[0] = [0, 350, 150, 2000]  # set the first layer
        model_data.sort_index(inplace=True)
        model_data.drop([len(model_data) - 1], inplace=True)
        model_data['depth'] = model_data['thickness'].cumsum()
        model_data['position'] = position
        return model_data

    model_file = glob(file_path + "/*.txt")
    data = pd.DataFrame(columns=['thickness', 'vp', 'vs', 'density', 'depth', 'position'])
    for model in model_file:
        model_data = read_model(model, float(os.path.basename(model).split('.')[0]))
        data = data.append(model_data, ignore_index=True)
    data['depth'] = data['depth'] * (-1)
    return data
