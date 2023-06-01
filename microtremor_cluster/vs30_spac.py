# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-03-22 12:56 PM
@File     : vs30_spac.py
@Software : PyCharm
"""

import glob
import re
import numpy as np
from tqdm import trange
from read_spacTarget import read_spacTarget
import seaborn as sns
import matplotlib.pyplot as plt


def vs30_spac(files_path):

    def find_split_point(data, X):
        for i in range(1, len(data)):
            left_data = data[:i]
            right_data = data[i:]
            left_mean = np.mean(left_data)
            right_data_all_less_than_X = all([value < X for value in right_data])

            if left_mean > X and right_data_all_less_than_X:
                break
        return i

    files = glob.glob(files_path + r"\*.target")

    vs30_all = []
    for i in trange(len(files)):
        target = read_spacTarget(files[i])
        columns = target.columns[3]
        ring = float(re.findall(r"\d+\.?\d*", columns)[0])
        # According the ring value to set the X and b value
        if ring >= 5 and ring < 7.5:
            X = 0.9
            b = 9.934
        elif ring >= 7.5 and ring < 10.5:
            X = 0.8
            b = 6.655
        elif ring >= 10.5 and ring < 13:
            X = 0.7
            b = 5.439
        elif ring >= 13 and ring < 15.5:
            X = 0.6
            b = 4.728
        elif ring >= 15.5 and ring < 17.5:
            X = 0.5
            b = 4.189
        elif ring >= 17.5 and ring < 19.5:
            X = 0.4
            b = 3.738
        elif ring >= 19.5 and ring < 23.5:
            X = 0.3
            b = 3.360
        elif ring >= 23.5 and ring < 25:
            X = 0.2
            b = 3.051
        else:
            X = 0
            b = 0
        if X == 0:
            continue
        else:
            target_slice = target[(target['freq'] >= 2)]  # Slice the data by frequency [2, 100]
            freq = np.array(target_slice['freq'])
            spac = np.array(target_slice[columns])

            split_point = find_split_point(spac, X)
            freq_split = freq[split_point]
            vs30 = (0.47 * b * X * ring + 20.43) * freq_split
            vs30_all.append(vs30)

    if len(vs30_all) == 0:
        print("The vs30 is empty!")
    else:
        print("The valid number of vs30 is: %d" % len(vs30_all))
        print("The max value of vs30 is: %.2f m/s" % max(vs30_all))
        print("The min value of vs30 is: %.2f m/s" % min(vs30_all))
        print("The mean of vs30 is: %.2f m/s" % np.mean(vs30_all))

        vs30_all = [value for value in vs30_all if value < 2000]
        sns.set()
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
        sns.displot(vs30_all, bins=20, kde=True, rug=False)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        plt.xlabel("vs30(m/s)")
        plt.title("vs30 distribution")
        plt.savefig(files_path + r"\vs30_distribution.png")
