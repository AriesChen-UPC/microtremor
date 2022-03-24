from read_page import read_page
import glob
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# 数据所在文件夹
# path = '/Users/tenqei/Desktop/MQtech2020/page_data/'
# 指定输出文件的标记
# note = '11'


def load_data3(path, freq, plot=False, output=True, note=None):
    fs = glob.glob(path + '*.page')
    d = [[],[],[]]
    e = [[],[],[]]
    n = []

    for fi in fs:
        data, n0 = read_page(fi)
        for i in range(len(data)):
            data[i].fillna(0)
        for ri in range(3):
            model = interp1d(x=list(data[ri].freq), y=list(data[ri].spac), kind='linear', bounds_error=False,
                             fill_value=(float(data[ri].spac.head(1)), float(data[ri].spac.tail(1))))
            modelerr = interp1d(x=list(data[ri].freq), y=list(data[ri].stderr), kind='linear', bounds_error=False,
                                fill_value=(float(data[ri].stderr.head(1)), float(data[ri].stderr.tail(1))))
            out = model(freq)
            outerr = modelerr(freq)
            d[ri].append(out)
            e[ri].append(outerr)
        n.append(n0)
        print('\rCollecting {:.0f} out of {:.0f} ...'.format(fs.index(fi) + 1, fs.__len__()))

    print('Scanned files:')
    print(' '.join(n))

    # output matrix
    m3 = [np.array(d[0]),np.array(d[1]),np.array(d[2])]
    me3 = [np.array(e[0]),np.array(e[1]),np.array(e[2])]
    n = np.array(n)

    if plot:
        plt.figure()
        for ri in range(3):
            plt.subplot(1, 3, ri+1)
            plt.plot(freq, m3[ri].transpose())
            plt.xlim(0, 20)
            plt.ylim(-0.4,1.05)
            plt.grid()
            plt.title('Ring {:.0f}'.format(ri+1))
        plt.suptitle('SPAC curves from {:.0f} points'.format(fs.__len__()))

    if output:
        dname = 'data'
        ename = 'stderr'
        fname = 'filename'
        if note:
            dname = dname + note
            ename = ename + note
            fname = fname + note
        for ri in range(3):
            np.savetxt(dname+'_r'+str(ri+1)+'.csv', m3[ri], delimiter=',')
            np.savetxt(ename+'_r'+str(ri+1)+'.csv', me3[ri], delimiter=',')
        np.savetxt(fname+'csv', n, delimiter=',', fmt='%s')
    return m3, me3, n

# outputs 2 matrices: data & stderr
# outputs 1 list: file names
