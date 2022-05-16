# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/15/2022 11:26 AM
@file: vs_plot_version_0.0.py
"""

from glob import glob
from gooey import Gooey, GooeyParser
from pykrige_func import pykrige_func


@Gooey(program_name='VS Plot', required_cols=1, optional_cols=2, default_size=(700, 550))
def main():
    parser = GooeyParser(description="This program is used to plot the VS profile using Kriging method!")
    parser.add_argument('Filepath', help="Please select the data(.xls, .xlsx) path.", widget="DirChooser")
    parser.add_argument('-Filename', help="Please select the data(.xls, .xlsx) for plotting.", widget="FileChooser")
    parser.add_argument('-Loop', help="Whether a loop operation is required.", widget="Dropdown", choices=['Yes', 'No'],
                        default='No')
    parser.add_argument('-minColor', help="Whether to input the min color limit.", widget="TextField")
    parser.add_argument('-maxColor', help="Whether to input the input the max color limit.", widget="TextField")

    args_ = parser.parse_args()
    return args_

#%% pykrige_func


if __name__ == '__main__':
    args = main()

    folder_path = args.Filepath
    file_path = args.Filename
    is_loop = args.Loop
    min_color = args.minColor
    max_color = args.maxColor

    if not file_path:
        filePath = glob(folder_path + '/*.xls*')
        if is_loop == 'Yes':
            for file_path in filePath:
                pykrige_func(file_path, min_color, max_color)
        else:
            print('No program execution, please check the operation!')
    else:
        pykrige_func(file_path, min_color, max_color)
