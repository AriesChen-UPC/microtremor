import struct
import pylab
import numpy as np
from scipy.fftpack import fft, ifft


class sacfile_wave:
    def read(self, sFile):
        sFile='D:\\MyProject\\Python\\PycharmProjects\\studyTest\\1.sac'

        f = open(sFile, 'rb')
        hdrBin = f.read(632)

        sfmt = 'f' * 70 + 'I ' * 40 + '8s ' * 22 + '16s';
        hdrFmt = struct.Struct(sfmt)
        self.m_header = hdrFmt.unpack(hdrBin)
        self.dt = self.m_header[0]

        npts = int(self.m_header[79])
        fmt_data = 'f' * npts
        dataFmt = struct.Struct(fmt_data)
        dataBin = f.read(4 * npts)
        f.close()
        self.m_data = dataFmt.unpack(dataBin)

    def draw(self, sImageFile):
        npts = len(self.m_data)
        xd = range(1, npts + 1)
        dt = self.dt
        NFFT = 250
        Fs = int(1.0 / dt)
        x0 = np.arange(0, npts / Fs, np.round(dt, 2))

        pylab.subplot(311)
        pylab.plot(x0, self.m_data, linewidth=.6)
        # pylab.xlabel('Time / second')
        # pylab.xlabel('时间 / 秒', fontproperties = 'FangSong', fontsize = 20)
        pylab.subplot(312)
        Pxx, freqs, bins, imm = pylab.specgram(self.m_data, NFFT=NFFT, Fs=Fs,
                                               noverlap=NFFT * 19 / 20, cmap=pylab.get_cmap('jet'))

        pylab.subplot(313)  # FFT快速傅丽叶变换，显示频谱曲线
        n = len(self.m_data)
        k = np.arange(n)
        T = n / Fs
        frq = k / T
        frq1 = frq[range(int(n / 2))]
        YY = np.fft.fft(self.m_data)
        Y = np.fft.fft(self.m_data) / n
        Y1 = Y[range(int(n / 2))]
        Y1[0] = Y1[1]
        pylab.plot(frq1, abs(Y1))

        pylab.savefig(sImageFile, dpi=1200)
        pylab.show()

    def exportAsc(self, sAscFile):
        f2 = open(sAscFile, "wt")
        sdataAsc = [str(x) for x in self.m_data]
        sDataAsc = '\n'.join(sdataAsc)
        f2.writelines(sDataAsc)
        f2.close()


if __name__ == "__main__":
    print('输入文件名必须为1.sac')
    print('输出文件名为1.png')
    sacfile = '1.sac'
    sac = sacfile_wave()
    sac.read(sacfile)
    sac.draw("1.png")