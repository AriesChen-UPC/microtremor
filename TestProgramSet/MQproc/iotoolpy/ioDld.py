import os
import re
import sqlite3
import struct
import warnings
import wave
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from tkinter import Tk, Frame, Scrollbar, Button
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showinfo, showerror
from tkinter.ttk import Treeview, Progressbar

import numpy as np

from iotoolpy.ioSac import write_sac, dict_to_header_arrays


def readhead(tmphead):
    """Read the head of DLD file (binary)

    Parameters
    ----------
    tmphead: bytes

    Returns
    -------
    dict
        The header dictionary
    """
    # Python decode() 方法以 encoding 指定的编码格式解码字符串。默认编码为字符串编码
    head = {'ID0': tmphead[0:16].decode('utf-8').strip('\x00'), 'ID1': tmphead[32:48].decode('utf-8').strip('\x00'),
            'ID2': tmphead[48:64].decode('utf-8').strip('\x00'), 'ID3': tmphead[64:90].decode('utf-8').strip('\x00'),
            'Ver0': tmphead[112:128].decode('utf-8').strip('\x00'),
            'Ver1': tmphead[128:144].decode('utf-8').strip('\x00'),
            'Time0': tmphead[144:160].decode('utf-8').strip('\x00'),
            'Date0': tmphead[160:176].decode('utf-8').strip('\x00'),
            'Time1': tmphead[176:192].decode('utf-8').strip('\x00'),
            'Date1': tmphead[192:208].decode('utf-8').strip('\x00'),
            '#0': tmphead[208:216], '#1': tmphead[216:224], 'Mode': tmphead[320:336].decode('utf-8').strip('\x00'),
            'Lon': struct.unpack('<d', tmphead[400:408])[0], 'Lat': struct.unpack('<d', tmphead[408:416])[0]}
    return head


def scandld(directory):
    """Scan the directory and return the DLD file database

    Parameters
    ----------
    directory: str

    Returns
    -------
    np.recarray
        Full Filename, instrument ID, file size, number of points, start timeframe, end timeframe
    """
    fs = list(Path(directory).rglob('*.DLD'))
    # fs = sorted([x for x in os.listdir(directory) if x.split('.')[-1].lower() == 'dld'])
    db = np.recarray(shape=len(fs), dtype=[('filename', 'U80'), ('instrument', 'U9'), ('filesize', 'i4'),
                                           ('npnts', 'i4'), ('starttime', 'U23'), ('endtime', 'U23')])
    for i in range(len(fs)):
        fn = fs[i]  # os.path.join(directory, fs[i])
        print(f'Loading {fn.stem} [{i+1}/{len(fs)}]')
        with open(fn, 'rb') as f:
            head = readhead(f.read(512))  # 读取文件
        instname = head['ID1']
        filesize = os.path.getsize(fn)
        if (filesize - 512) % 3072 != 0:
            raise Exception('Filesize incorrect!')
        nblock = (filesize - 512) // 3072
        npnts = nblock * 1000 + 1
        if head['Mode'] in ['jiaoben1', '2ms_0dB', '2ms_0dB11', '2_ms_0dB']:
            dt = 2e-3
        else:
            warnings.warn('Sampling rate not 2ms! Output sampling rate might not be valid!')
            try:
                dt = int(re.sub('[^0-9]', '', head['Mode'].split('_')[0])) * 1e-3
            except ValueError:
                warnings.warn('Failed reading sampling rate! Setting sampling rate to 2ms.')
                dt = 2e-3
        st = datetime(int(head['Date0'][0:4]), int(head['Date0'][4:6]), int(head['Date0'][6:8]),
                      int(head['Time0'][0:2]), int(head['Time0'][2:4]), int(head['Time0'][4:6]),
                      int(float(head['Time0'][6:]) * 1e3)) + timedelta(seconds=dt * 500)
        et = datetime(int(head['Date1'][0:4]), int(head['Date1'][4:6]), int(head['Date1'][6:8]),
                      int(head['Time1'][0:2]), int(head['Time1'][2:4]), int(head['Time1'][4:6]),
                      int(float(head['Time1'][6:]) * 1e3)) + timedelta(seconds=dt * 500)
        db[i] = (fn, instname, filesize, npnts, st, et)
    return db


def readdld(dldfile):
    """Read DLD file content

    Parameters
    ----------
    dldfile: str

    Returns
    -------
    (np.ndarray, dict, float, datetime, datetime, str, int)
        dout: Data output,
        head: Header dictionary,
        dt: sampling period,
        t0: Start time as datetime class,
        t1: End time as datetime class,
        channel: Channel ('N', 'E', 'Z'),
        nblock: Number of blocks in data file
    """
    channel = os.path.split(dldfile)[-1].split('.')[-2][-1]
    # 把xyz换成ENZ
    if channel.lower() == 'x':
        channel = 'E'
    elif channel.lower() == 'y':
        channel = 'N'
    elif channel.lower() == 'z':
        channel = 'Z'
    else:
        warnings.warn('Channel detection failed!')
    filesize = os.path.getsize(dldfile)
    if (filesize - 512) % 3072 != 0:
        raise Exception('Filesize incorrect!')
    nblock = (filesize - 512) // 3072
    file = open(dldfile, 'rb')
    head = readhead(file.read(512))
    if head['Mode'] in ['jiaoben1', '2ms_0dB', '2ms_0dB11', '2_ms_0dB']:
        dt = 2e-3
    else:
        warnings.warn('Sampling rate not 2ms! Output sampling rate might not be valid!')
        try:
            dt = int(re.sub('[^0-9]', '', head['Mode'].split('_')[0])) * 1e-3
        except ValueError:
            dt = 2e-3
    # TODO: Why do I add this timedelta here? Anyway it doesn't matter since every values has it...
    t0 = datetime(int(head['Date0'][0:4]), int(head['Date0'][4:6]), int(head['Date0'][6:8]),
                  int(head['Time0'][0:2]), int(head['Time0'][2:4]), int(head['Time0'][4:6]),
                  int(float(head['Time0'][6:]) * 1e3)) + timedelta(seconds=dt * 500)
    t1 = datetime(int(head['Date1'][0:4]), int(head['Date1'][4:6]), int(head['Date1'][6:8]),
                  int(head['Time1'][0:2]), int(head['Time1'][2:4]), int(head['Time1'][4:6]),
                  int(float(head['Time1'][6:]) * 1e3)) + timedelta(seconds=dt * 500)
    timeSteps = [t0 + timedelta(seconds=its) for its in np.arange(nblock) * dt * 1000]
    if not (t0 == timeSteps[0] and t1 == timeSteps[-1]):
        warnings.warn('Startime/Endtime and timeStamp no match')
    data = np.zeros(shape=(nblock, 1000), dtype=float)
    for i in range(nblock):
        tmpblock = list()
        for j in range(1000):
            tmpblock.append(int.from_bytes(file.read(3), byteorder='little', signed=True))
        tmp = file.read(72)
        blockHead = {'#0': tmp[4:12], '#1': tmp[56:64].decode('utf-8').strip('\x00'),
                     'Time': tmp[12:21].decode('utf-8').strip('\x00'),
                     'Date': tmp[23:31].decode('utf-8').strip('\x00'),
                     'Lat': struct.unpack('<d', tmp[32:40])[0], 'Lon': struct.unpack('<d', tmp[48:56])[0]}
        ti = datetime(int(blockHead['Date'][0:4]), int(blockHead['Date'][4:6]), int(blockHead['Date'][6:8]),
                      int(blockHead['Time'][0:2]), int(blockHead['Time'][2:4]), int(blockHead['Time'][4:6]),
                      int(float(blockHead['Time'][6:]) * 1e3)) + timedelta(seconds=dt * 500)
        if type(timeSteps.index(ti)) is not int:
            warnings.warn('Time block matching error!')
            pass
        else:
            data[timeSteps.index(ti)] = np.array(tmpblock) / (2 ** 23 - 1) * 2500
    dout = np.append(data.flatten(), 0)
    file.close()
    return dout, head, dt, t0, t1, channel, nblock


def dld2sac(dldfile, outpath, wavefile=None):
    """Convert one DLD file to SAC

    Parameters
    ----------
    dldfile: str
    outpath: str
    wavefile: str
    """
    data, head, dt, t0, t1, channel, nblock = readdld(dldfile)
    if wavefile is not None:
        data = wavedeconvolve(wavefile, data, dt)
    sacHead = {'delta': dt, 'scale': 1.0, 'odelta': dt, 'b': 0.0, 'e': (nblock * 1000 + 1) * dt, 'o': 0.0,
               'stla': head['Lat'], 'stlo': head['Lon'], 'stel': 0.0,
               'nzyear': t0.year, 'nzjday': int(t0.strftime('%j')),
               'nzhour': t0.hour, 'nzmin': t0.minute, 'nzsec': t0.second, 'nzmsec': int(t0.microsecond / 1e3),
               'nvhdr': 6, 'npts': nblock * 1000 + 1, 'iftype': 1, 'iztype': 11, 'leven': 1,
               'kinst': head['ID1'][:-7], 'kstnm': head['ID1'][-7:], 'kcmpnm': 'DP' + channel}
    sacHeadArray = dict_to_header_arrays(sacHead)
    if wavefile is not None:
        dtccname = '.'.join([head['ID1'], '2', t0.strftime('%Y.%m.%d.%H.%M.%S.%f')[:-3], channel, 'sac'])
    else:
        dtccname = '.'.join([head['ID1'], '1', t0.strftime('%Y.%m.%d.%H.%M.%S.%f')[:-3], channel, 'sac'])
    outfile = os.path.join(outpath, dtccname)
    write_sac(outfile, sacHeadArray[0], sacHeadArray[1], sacHeadArray[2], data)


def readdb(dbfile):
    """[Possibly depreiciated] Read the DLD directory DB file

    Parameters
    ----------
    dbfile: str

    Returns
    -------
    (list, list[int], list[int], list[datetime], list[datetime], list)
        Filename, number of points, file size, file start timeframe, sampling delta, project name
    """
    # 该 API 打开一个到 SQLite 数据库文件 database 的链接。您可以使用 ":memory:" 来在 RAM 中打开一个到 database 的数据库连接
    # 而不是在磁盘上打开。如果数据库成功打开，则返回一个连接对象。
    con = sqlite3.connect(dbfile)
    # fetchAll()方法获取结果集中的所有行。
    fname = [x[0] for x in con.execute("SELECT fname FROM tbl_main").fetchall()]
    # 用于执行返回多个结果集、多个更新计数或二者组合的语句。
    # execute方法应该仅在语句能返回多个ResultSet对象、多个更新计数或ResultSet对象与更新计数的组合时使用。
    fsize = [int(x[0]) for x in con.execute("SELECT filesize FROM tbl_main").fetchall()]
    ftime = [datetime.fromisoformat(x[0]) for x in con.execute("SELECT fdatetime FROM tbl_main").fetchall()]
    dltime = [datetime.fromisoformat(x[0]) for x in con.execute("SELECT dl_datetime FROM tbl_main").fetchall()]
    proj = [x[0] for x in con.execute("SELECT project_name FROM tbl_main").fetchall()]
    npnts = list()
    for i in range(len(fname)):
        if fname[i].split('.')[-1] == 'dld' or fname[i].split('.')[-1] == 'DLD':
            if (fsize[i] - 512) % 3072 == 0:
                npnts.append((fsize[i] - 512) // 3072)
            else:
                npnts.append('WrongSize')
        else:
            npnts.append('NotApplicable')
    con.close()
    return fname, npnts, fsize, ftime, dltime, proj


def sign_extend(bits, value):
    """ Sign-extend from http://stackoverflow.com/questions/32030412/twos-complement-sign-extension-python """
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def get_samples(w):
    # 文件指针的位置倒回文件开头
    w.rewind()
    num_samples = w.getnframes()
    sample_depth = w.getsampwidth()
    raw_data = w.readframes(num_samples)
    for i in range(0, num_samples * sample_depth, sample_depth):
        yield sign_extend(
            sample_depth * 8,
            reduce(lambda x, y: x << 8 | y, reversed(raw_data[i:i + sample_depth])))


def wavedeconvolve(wavefile, dat, dt):  # TODO: should we read the .WAV file and do deconvolution?
    """ Alpha funciton that uses a WAV file to deconvolve the instrument response

    testing...
    import matplotlib.pyplot as plt
    from iotoolpy.ioDld import *
    wavefile='/Users/tenqei/Desktop/590000029/PULSE_Z.WAV'
    wav = np.array(list(get_samples(wave.open(wavefile, 'r'))))
    dat, head, dt, t0, t1, chn, n = readdld('/Users/tenqei/Desktop/590000029/seis049Z.DLD')
    # a=np.fromfile('/Users/tenqei/Downloads/Export06/Project4/Project4.GPZ')
    """
    tmpwav = wave.open(wavefile, 'r')
    wav = np.array(list(get_samples(tmpwav)))
    n0 = len(wav)
    n1 = len(dat)
    rate = tmpwav.getframerate()
    # wav1 = np.interp(np.linspace(0,n0/rate0, int(n0/rate0*rate1)),np.linspace(0,n0/rate0,n0),wav)
    fwav = np.fft.fftshift(np.fft.fft(wav))
    fdat = np.fft.fftshift(np.fft.fft(dat))
    fw = np.interp(np.fft.fftshift(np.fft.fftfreq(n1, dt)), np.fft.fftshift(np.fft.fftfreq(n0, 1/rate)), fwav)
    dout = np.fft.ifft(np.fft.ifftshift(fdat / fw)).real  # TODO: verify if normalization is needed
    return dout


if __name__ == '__main__':
    try:
        root = Tk()
        # 设置窗口大小
        winWidth = 800
        winHeight = 600
        # 获取屏幕分辨率
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()
        x = int((screenWidth - winWidth) / 2)
        y = int((screenHeight - winHeight) / 2)
        # 设置窗口初始位置在屏幕居中
        root.geometry("%sx%s+%s+%s" % (winWidth, winHeight, x, y))
        # 设置主窗口标题
        root.title("DTCC数据预览")
        # 设置窗口图标
        # root.iconbitmap("./image/icon.ico")
        # 设置表格布局
        frame = Frame(root)
        frame.pack(expand=True, fill='both')
        columns = ('ID', 'filename', 'starttime', 'endtime')
        tree = Treeview(frame, show='headings', columns=columns, selectmode='extended')
        tree.pack(expand=True, side='left', fill='both')
        sb = Scrollbar(frame, orient='vertical')
        sb.pack(side='right', fill='y')
        tree.config(yscrollcommand=sb.set)
        sb.config(command=tree.yview)
        # 设置表格文字居中
        tree.column("ID", anchor="center", width=35)
        tree.column("filename", anchor="center", width=350)
        tree.column("starttime", anchor="center", width=200)
        tree.column("endtime", anchor="center", width=200)
        # 设置表格头部标题
        tree.heading("ID", text="ID")
        tree.heading("filename", text="文件目录"'')
        tree.heading("starttime", text="开始时间 (UTC)")
        tree.heading("endtime", text="结束时间 (UTC)")

        # 扫描目录
        path = askdirectory(title='请选择仪器目录')
        tab = scandld(path)
        for i in range(len(tab)):
            tree.insert('', i, values=(i, tab['filename'][i], tab['starttime'][i], tab['endtime'][i]))


        def convert_selected():
            print('Selected item(s): {}'.format(tree.selection()))
            ind = [tree.item(isel, 'values')[0] for isel in tree.selection()]
            pb.start()
            for i in range(len(ind)):
                # pvar.set((i+1)/len(ind))
                print('Converting {} ({}/{})'.format(tab['filename'][int(ind[i])], i + 1, len(ind)))
                dld2sac(tab['filename'][int(ind[i])], str(Path(tab['filename'][int(ind[i])]).parent))
            pb.stop()
            showinfo(title='DTCC数据转换', message='转换完成！\n数据已输出至{}'.format(path))


        def convert_response():
            print('Selected item(s): {}'.format(tree.selection()))
            ind = [tree.item(isel, 'values')[0] for isel in tree.selection()]
            pb.start()
            for i in range(len(ind)):
                # pvar.set((i+1)/len(ind))
                print('Converting {} ({}/{})'.format(tab['filename'][int(ind[i])], i + 1, len(ind)))
                tmpfile = tab['filename'][int(ind[i])]
                wavchn = 'PULSE_{}.WAV'.format(tmpfile[-5])
                dld2sac(tab['filename'][int(ind[i])], path, wavefile=os.path.join(os.path.dirname(tmpfile), wavchn))
            pb.stop()
            showinfo(title='DTCC数据转换', message='转换+去响应完成！\n数据已输出至{}'.format(path))


        b1 = Button(root, text='转换格式', command=convert_selected)
        b1.pack(after=frame, side='left')
        b2 = Button(root, text='去响应转换', command=None)
        b2.pack(after=b1, side='left')
        pb = Progressbar(root, mode='indeterminate', orient='horizontal')
        pb.pack(after=b2, side='left', fill='x', expand=True)

        root.mainloop()
    except AssertionError or Exception as e:
        showerror(type(e).__name__, str(e))
        raise e
