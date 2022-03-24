import os
import obspy.core.stream as st
import obspy.core.trace as tr
from obspy import UTCDateTime
import numpy as np


def verify_filename(filename):
    subname = filename.split('/')
    filename = subname[-1]
    subname = filename.split('.')
    file_id = subname[0]
    t_raw = int(file_id[:-5])
    status = file_id[-5:]
    check = bool(status[-1])
    sampling_rate = int(status[:-1])
    sec0 = t_raw / sampling_rate
    return sec0, check, sampling_rate


def read_JST_int(file, cut=False, starttime=None, endtime=None,
                 sta='', net='', chn='BHZ', loc=''):
    """
    function to read one specified JST raw data file
    ( data format: int32 )
    :type file: str
    :type cut: bool
    :type starttime: UTCDateTime
    :type endtime: UTCDateTime
    :type sta: str
    :type net: str
    :type chn: str
    :type loc: str
    """
    t0, chk, sampling_rate = verify_filename(file)
    if not chk:
        print('File {} with bad record\nReturning starting time...'.format(file))
        return t0
    else:
        t0 = int(file[-22:-9]) / 1000  # unit: s
        stats = tr.Stats()
        stats.starttime = UTCDateTime('1980-01-06T00:00:00') + t0
        stats.sampling_rate = sampling_rate
        stats.station = sta
        stats.network = net
        stats.channel = chn
        stats.location = loc
        stats.npts = os.path.getsize(file) / 4
        with open(file, 'rb') as f:
            sig = tr.Trace(data=np.fromfile(f, dtype=np.int32) * 1e-6,
                           header=stats)  # unit: mV
        if cut:
            if not starttime:
                starttime = sig.stats.starttime
            if not endtime:
                endtime = sig.stats.endtime
            sig.trim(starttime, endtime)
        return sig


def read_JST_station(path, net='', chn='BHZ', loc='',
                     starttime=None, endtime=None):
    """
    function that reads all station in dir during specified time
    :type path: str
    :type net: str
    :type chn: str
    :type loc: str
    :type starttime: UTCDateTime
    :type endtime: UTCDateTime
    """
    stations = os.listdir(path)
    sig = st.Stream()
    for sta in stations:
        print('Reading station: {}...'.format(sta))
        fs = os.listdir('/'.join([path, sta]))
        fs.sort(key=lambda file: file[-22:-9])
        # filtering starttime and endtime
        t_min = int(fs[0][-22:-9]) / 1000
        t_max = (int(fs[-1][-22:-9]) + os.path.getsize('/'.join([path, sta, fs[-1]])) / 4) / 1000
        if starttime:
            t0 = max(starttime - UTCDateTime('1980-01-06T00:00:00'), t_min)
        else:
            t0 = t_min
        if endtime:
            t1 = min(endtime - UTCDateTime('1980-01-06T00:00:00'), t_max)
        else:
            t1 = t_max
        fs_sel = list()
        for fn in fs:
            tn = int(fn[-22:-9]) / 1000  # unit: ms
            if t0 - pow(2, 19) / 1000 < tn <= t1 + pow(2, 19) / 1000:
                fs_sel.append(fn)
        # acquiring signal from selected files
        sta_sig = read_JST_int(file='/'.join([path, sta, fs_sel[0]]),
                               sta=sta, net=net, chn=chn, loc=loc)
        for fn in fs_sel[1:-1]:
            sta_sig += read_JST_int(file='/'.join([path, sta, fn]),
                                    sta=sta, net=net, chn=chn, loc=loc)
        sta_sig.trim(starttime=UTCDateTime('1980-01-06T00:00:00') + t0,
                     endtime=UTCDateTime('1980-01-06T00:00:00') + t1,
                     fill_value=0)
        sig.append(sta_sig)
    print('Done reading {} stations!'.format(len(stations)))
    return sig


# todo: add units,

# setup time to read
year = 2020
month = 12
day = 14
t_from = 1130
t_to = 1230
date = UTCDateTime(year, month, day)

#filepath = '/Users/tenqei/Desktop/MQtech2020/test20200426'
#st_time = UTCDateTime('2020-04-26T04:15:00')
#ed_time = UTCDateTime('2020-04-26T04:45:00')

#test_sig = read_JST_station(filepath, 'JST', 'BHZ', 'ANT0', st_time, ed_time)

#for tr in test_sig:
#    fn = os.path.join(filepath,'_'.join([tr.stats.network,tr.stats.station,tr.stats.channel])+'.sac')
#    tr.write(fn,'sac')