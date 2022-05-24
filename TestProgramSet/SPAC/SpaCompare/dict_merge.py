# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/6/2022 9:14 AM
@file: dict_merge.py: combine two dicts into one dict, the value of the same key will be the sum of the two values,
                    the value of the different keys will be the value of the dict
"""

dict_01 = {"A": [['A1'], 1], "B": [['B1'], 1], "C": [['C1'], 1]}
dict_02 = {"A": [['A2'], 1], "B": [['B2'], 1], "D": [['D1'], 1]}

dict_all = {"dict_01": dict_01, "dict_02": dict_02}

dict_combine = {}
for key in dict_all:
    for k, v in dict_all[key].items():
        if k in dict_combine:
            dict_combine[k][0].extend(v[0])
            dict_combine[k][1] += v[1]
        else:
            dict_combine[k] = v

print(dict_combine)
