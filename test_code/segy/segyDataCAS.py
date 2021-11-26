# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 2021/10/21 10:42
@file: segyDataCAS.py
"""

from obspy import read
import matplotlib.pyplot as plt
import numpy as np

st = read("F:/CAS/Data/AdditionalData/z_component/l5_d2_119_1096.2021.09.29.04.18.10.000.z.segy")
plt.plot(st[0])
