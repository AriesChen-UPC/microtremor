import obspy
import matplotlib.pyplot as plt
from obspy.signal.detrend import polynomial
import numpy as np
from scipy.fftpack import fft,ifft,fftfreq



a1 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/1_Z.sac')
a2 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/2_Z.sac')
a3 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/3_Z.sac')
a4 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/4_Z.sac')
a5 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/5_Z.sac')
a6 = obspy.read('D:/项目资料/西安/微动数据/20200527/all/6_Z.sac')


polynomial(a1[0].data[0:2000], order=3, plot=True)
polynomial(a2[0].data[0:2000], order=3, plot=True)

#plt.plot(a1[0].data[0:2000])
#plt.plot(a2[0].data[0:2000])
#plt.show()

f = np.linspace(0,500,2000)
a11 = a1[0].data[0:2000]
a111 = fft(a11)
a22 = a2[0].data[0:2000]
a222 = fft(a22)
b111 = a111 - a222
plt.plot(f,a111)
plt.plot(f,a222)
plt.xlim(0.5,40)
plt.show()
