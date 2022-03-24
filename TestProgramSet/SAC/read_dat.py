
import numpy as np
import matplotlib.pyplot as plt

c = np.fromfile('D:\\项目资料\\资料收集\\吉赛特相关\\Convert_Python\\1334\\126807034940410001.dat', dtype=int)
plt.plot(c[20000:30000])
plt.show()