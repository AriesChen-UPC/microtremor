import os
import re
import sqlite3
import struct
import warnings
from datetime import datetime, timedelta

import numpy as np

from iotoolpy.ioSac import write_sac, dict_to_header_arrays


def scandld(directory):
    """Scan the directory and return the DLD file database

    Parameters
    ----------
    directory: str

    Returns
    -------
    np.recarray
        Filename, instrument ID, file size, number of points, start timeframe, end timeframe
    """
    fs = sorted([x for x in os.listdir(directory) if x.split('.')[-1].lower() == 'dld'])
    db = np.recarray(shape=len(fs),
                     dtype=[('filename', 'U80'), ('instrument', 'U9'), ('filesize', 'i4'), ('npnts', 'i4'),
                            ('starttime', 'U23'), ('endtime', 'U23')])
    for i in range(len(fs)):
        fn = os.path.join(directory, fs[i])
        with open(fn, 'rb') as f:
            tmphead = f.read(512)
        instname = tmphead[32:48].decode('utf-8').strip('\x00')
        filesize = os.path.getsize(fn)
        if (filesize - 512) % 3072 != 0:
            raise Exception('Filesize incorrect!')
        nblock = (filesize - 512) // 3072
        npnts = nblock * 1000 + 1
        mode = tmphead[320:336].decode('utf-8').strip('\x00')
        if '2ms' in mode:
            dt = 2e-3
        else:
            warnings.warn('Sampling rate not 2ms! Output sampling rate might not be valid!')
            try:
                dt = int(re.sub('[^0-9]', '', mode.split('_')[0])) * 1e-3
            except ValueError:
                dt = 2e-3
        tmptime = {'Time0': tmphead[144:160].decode('utf-8').strip('\x00'),
                   'Date0': tmphead[160:176].decode('utf-8').strip('\x00'),
                   'Time1': tmphead[176:192].decode('utf-8').strip('\x00'),
                   'Date1': tmphead[192:208].decode('utf-8').strip('\x00')}
        st = datetime(int(tmptime['Date0'][0:4]), int(tmptime['Date0'][4:6]), int(tmptime['Date0'][6:8]),
                      int(tmptime['Time0'][0:2]), int(tmptime['Time0'][2:4]), int(tmptime['Time0'][4:6]),
                      int(float(tmptime['Time0'][6:]) * 1e3)) + timedelta(dt * 500)
        et = datetime(int(tmptime['Date1'][0:4]), int(tmptime['Date1'][4:6]), int(tmptime['Date1'][6:8]),
                      int(tmptime['Time1'][0:2]), int(tmptime['Time1'][2:4]), int(tmptime['Time1'][4:6]),
                      int(float(tmptime['Time1'][6:]) * 1e3)) + timedelta(dt * 500)
        db[i] = (fn, instname, filesize, npnts, st, et)
    return db


def readdb(dbfile):
    """Read the DLD directory DB file

    Parameters
    ----------
    dbfile: str

    Returns
    -------

        Filename, number of points, file size, file start timeframe, sampling delta, microtremor name
    """
    con = sqlite3.connect(dbfile)
    fname = [x[0] for x in con.execute("SELECT fname FROM tbl_main").fetchall()]
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


def dld2sac(dldfile, outpath):
    """[CORE] Convert one DLD file to SAC

    Parameters
    ----------
    dldfile: str
    outpath: str
    """
    channel = os.path.split(dldfile)[-1].split('.')[-2][-1]
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
    tmphead = file.read(512)
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
    if '2ms' in head['Mode']:
        dt = 2e-3
    else:
        warnings.warn('Sampling rate not 2ms! Output sampling rate might not be valid!')
        try:
            dt = int(re.sub('[^0-9]', '', head['Mode'].split('_')[0])) * 1e-3
        except ValueError:
            dt = 2e-3

    t0 = datetime(int(head['Date0'][0:4]), int(head['Date0'][4:6]), int(head['Date0'][6:8]),
                  int(head['Time0'][0:2]), int(head['Time0'][2:4]), int(head['Time0'][4:6]),
                  int(float(head['Time0'][6:]) * 1e3)) + timedelta(dt * 500)
    t1 = datetime(int(head['Date1'][0:4]), int(head['Date1'][4:6]), int(head['Date1'][6:8]),
                  int(head['Time1'][0:2]), int(head['Time1'][2:4]), int(head['Time1'][4:6]),
                  int(float(head['Time1'][6:]) * 1e3)) + timedelta(dt * 500)
    timeSteps = [t0 + timedelta(seconds=its) for its in np.arange(nblock) * dt * 1000]
    assert t0 == timeSteps[0] and t1 == timeSteps[-1]
    data = np.zeros(shape=(nblock, 1000), dtype=float)
    for i in range(nblock):
        tmpblock = list()
        for j in range(1000):
            tmpblock.append(int.from_bytes(file.read(3), byteorder='little', signed=True))
        tmp = file.read(72)
        blockHeead = {'#0': tmp[4:12], '#1': tmp[56:64].decode('utf-8').strip('\x00'),
                      'Time': tmp[12:21].decode('utf-8').strip('\x00'),
                      'Date': tmp[23:31].decode('utf-8').strip('\x00'),
                      'Lat': struct.unpack('<d', tmp[32:40])[0], 'Lon': struct.unpack('<d', tmp[48:56])[0]}
        ti = datetime(int(blockHeead['Date'][0:4]), int(blockHeead['Date'][4:6]), int(blockHeead['Date'][6:8]),
                      int(blockHeead['Time'][0:2]), int(blockHeead['Time'][2:4]), int(blockHeead['Time'][4:6]),
                      int(float(blockHeead['Time'][6:]) * 1e3)) + timedelta(dt * 500)
        if type(timeSteps.index(ti)) is not int:
            warnings.warn('Time block matching error!')
            pass
        else:
            data[timeSteps.index(ti)] = np.array(tmpblock) / (2 ** 23 - 1) * 2500
    d = np.append(data.flatten(), 0)
    file.close()
    sacHead = {'delta': dt, 'scale': 1.0, 'odelta': dt, 'b': 0.0, 'e': (nblock * 1000 + 1) * dt, 'o': 0.0,
               'stla': head['Lat'], 'stlo': head['Lon'], 'stel': 0.0,
               'nzyear': t0.year, 'nzjday': int(t0.strftime('%j')),
               'nzhour': t0.hour, 'nzmin': t0.minute, 'nzsec': t0.second, 'nzmsec': int(t0.microsecond / 1e3),
               'nvhdr': 6, 'npts': nblock * 1000 + 1, 'iftype': 1, 'iztype': 11, 'leven': 1,
               'kinst': head['ID1'][:-7], 'kstnm': head['ID1'][-7:], 'kcmpnm': 'DP' + channel}
    sacHeadArray = dict_to_header_arrays(sacHead)
    dtccname = '.'.join([head['ID1'], '1', t0.strftime('%Y.%m.%d.%H.%M.%S.%f')[:-3], channel, 'sac'])
    outfile = os.path.join(outpath, dtccname)
    write_sac(outfile, sacHeadArray[0], sacHeadArray[1], sacHeadArray[2], d)
