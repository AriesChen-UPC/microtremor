# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 11/24/2021 8:56 AM
@file: theorySPAC.py
       The code is designed to calculate the theory SPAC curve
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import *

# set the radius
r = 3.0
d = 0.01
# freq = np.arange(0.1, 100, d)
freq = np.logspace(np.log10(0.1), np.log10(100), num=400)
Vs = np.dot(1500, [math.pow(f, -0.65) for f in freq])  # todo:only size-1 arrays can be converted to Python scalars
autoCorrRatio = jn(0, np.multiply(r*2*math.pi*freq, [math.pow(v, -1) for v in Vs]))
plt.plot(freq, autoCorrRatio)
plt.grid()


