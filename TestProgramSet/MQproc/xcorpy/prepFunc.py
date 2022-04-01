import configparser
import fnmatch
import re
import warnings
from pathlib import Path
from configparser import ConfigParser

import numpy as np
from obspy import read, Stream, UTCDateTime
from progressbar import ProgressBar, Bar, Percentage

from iotoolpy.ioGpy import SignalDatabase
from xcorpy.coreFunc import update_prog, update_logs, update_init, print_line, geopsy_component, stalta


def prep_atrig(sig0, dt, sta, lta):
    """Prepare the STA/LTA anti-triggering value, fixing the atrig maxtrix with the same length as the input signal

    Parameters
    ----------
    sig0: Stream
        Signal to calculate sta/lta
    dt: float
        Sampling period in sec
    sta: float
        Short-time-average length in sec
    lta: float
        Long-time-average length in sec
    """
    # calc sta/lta
    _atrig = np.zeros([len(sig0), int(np.floor(sig0[0].stats.npts * dt / sta))])
    for k in range(len(sig0)):
        _atrig[k, :] = stalta(sig0[k].data, dt, sta, lta)
    atrig = np.ones([len(sig0), sig0[0].stats.npts])
    xatrig = atrig.shape[1] // _atrig.shape[1]
    yatrig = atrig.shape[1] % _atrig.shape[1]
    atrig[:, yatrig // 2:atrig.shape[1] - (yatrig - yatrig // 2)] = \
        _atrig.repeat(xatrig).reshape(len(sig0), xatrig * _atrig.shape[1])
    return atrig


def prep_location(sig0=None, coord=None):
    """Prepare the station coordinates to ring infomation.
    If sig0 is provided, will use SAC header user7/user8 as coordinates.
    Otherwise, a coordinate matrix is needed

    Parameters
    ----------
    sig0: Stream
    coord: np.ndarray
    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
        5 ring information is returned.
        [ring_mat]: inter-station distance matrix;
        [ring_num]: number of ring groups;
        [ring_grp]: radius of each ring group;
        [ring_nps]: number of pairs per ring group;
        [ring_azs]: azimuth of each pair
    """
    if sig0:
        coord = np.array([[tr.stats.sac.user7, tr.stats.sac.user8] for tr in sig0])
    cx = coord[:, 0]  # np.array([sig0[i].stats.stlo for i in range(len(sig0))])
    cy = coord[:, 1]  # np.array([sig0[i].stats.stla for i in range(len(sig0))])
    ring_mat = np.round(np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2), 1)
    ring_grp = np.sort(np.unique(ring_mat))[1:]
    ring_num = len(ring_grp)
    ring_nps = np.array([int((ring_mat == igrp).sum() / 2) for igrp in ring_grp])
    ring_azs = np.rad2deg(np.arctan2(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :]))
    return coord, ring_mat, ring_num, ring_grp, ring_nps, ring_azs


def find_files(path, glob_pat, ignore_case=False):
    """Recursively find files of certain pattern in given path

    Parameters
    ----------
    path: str or Path
        Path to search
    glob_pat: str
        File pattern, supports regular expression
    ignore_case: bool
        If pattern case is ignored
    Returns
    -------
    list[Path]
    """
    rule = re.compile(fnmatch.translate(glob_pat), re.IGNORECASE) if ignore_case \
        else re.compile(fnmatch.translate(glob_pat))
    return [n for n in sorted(Path(path).rglob('*')) if rule.match(str(n))]


def find_sac(path, name=None, ftype='', chn='Z', verbose=False, update=False, fs0=None):
    """Find SAC file(s) by name, type and channel.
        Principle for name selection:
            1. n0 (first split by '.' from name) is any of parent directory
            2. name is any of the parent directory
            3. name with no '.' is in the filename

    Parameters
    ----------
    path: str or Path
        Path to search for SAC files
    name: str
        Station/instrument name, have no larger than 1 separator '.'
    ftype: str
        Station/instrumnet type ['DTCC', 'CGE', 'JST']
    chn: str
        Channel ['E', 'N', 'Z'], 'X' and 'Y' are supported but not recommended
    verbose: bool
    update: bool
    fs0: list
        List of pre-selected files (optional)

    Returns
    -------
    list[Path]
    """
    if fs0 is not None:
        fs = fs0
    else:
        fs = find_files(path, '*.sac', ignore_case=True)
    # find filetype
    if ftype.lower() == 'dtcc':
        fs = [x for x in fs if 'dtcc' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if any(['bdfsac' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]) and \
                any(['sac' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]):
            fs = [x for x in fs if 'bdfsac' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if chn not in ['E', 'N', 'Z']:
            if chn == 'X':
                chn = 'E'
            elif chn == 'Y':
                chn = 'N'
            else:
                if update:
                    update_logs('PythonException', 'error', 'Illegal channel for DTCC!')
                raise Exception('Illegal channel for DTCC!')
    elif ftype.lower() == 'cge':
        fs = [x for x in fs if 'cge' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if any(['sac' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]) and \
                any(['raw' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]):
            fs = [x for x in fs if 'sac' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if chn not in ['X', 'Y', 'Z']:
            if chn == 'E':
                chn = 'X'
            elif chn == 'N':
                chn = 'Y'
            else:
                if update:
                    update_logs('PythonException', 'error', 'Illegal channel for CGE!')
                raise Exception('Illegal channel for CGE!')
    elif ftype.lower() == 'jst':
        fs = [x for x in fs if 'jst' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if any(['sac' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]) and \
                any(['raw' in [xp.lower() for xp in x.relative_to(Path(path)).parts] for x in fs]):
            fs = [x for x in fs if 'sac' in [xp.lower() for xp in x.relative_to(Path(path)).parts]]
        if chn != 'Z':
            if update:
                update_logs('PythonException', 'error', 'Illegal channel for JST!')
            raise Exception('Illegal channel for JST!')
    else:
        if update:
            update_logs('PythonWarning', 'warning', 'Instrument type not filtered!')
        else:
            warnings.warn('Instrument type not filtered!')
    # find filename
    if name:
        if len(name.split('.')) == 2:
            n0, n1 = name.split('.')
        elif len(name.split('_')) == 2:  # TODO: changed name separator from '_' to '.', verify rest of the code
            name = name.replace('_', '.')
            n0, n1 = name.split('.')
            warnings.warn('Name separator is "_" not ".", this separator is planned to depreciate in the future')
            if update:
                update_logs('PythonWarning', 'warning', 'Old name separator detected!')
        else:
            n0 = name
            n1 = None
        fs = [x for x in fs if n0 in x.relative_to(Path(path)).parts or name.replace('.', '') == x.stem]
        if n1:
            fs = [x for x in fs if n1 in x.relative_to(Path(path)).parts]
    # find channel
    if any(chn in re.split('[._]', x.stem) for x in fs):
        fs = [x for x in fs if chn in re.split('[._]', x.stem)]
    else:
        print('Channel {} no found!'.format(chn)) if verbose else None
        if update:
            update_logs('Channel verification', 'warning', 'Instrument type not filtered!')

    if len(fs) == 0:
        print('Station {} no found!'.format(name)) if verbose else None
        if update:
            update_logs('PythonWarning', 'warning', 'No file selected!')

    return fs


# reading project settings
def prep_project(data_path, record, param, channel=None, update=False, geopsy=False):
    """Scan data files in path, then prepare by the record

    # TODO: The loading sequence now tries to save Z coordinates in gpy.ReceiverZ/sac.user9,
        make sure other modules' compatibility!
    # TODO: loading sequance too slow! need to refine details (parallel computation?)

    This function supports 2 input methods and 2 output methods.
    The input methods are either polygon record or coordinate record.
    The output methods are either Geopsy database or Obspy signals.

    Parameters
    ----------
    data_path: str or Path
    record: np.recarray
    param: configparser.ConfigParser
    channel: list[str]
    update: bool
    geopsy: bool

    Returns
    -------
    list[Stream] or list[SignalDatabase]
    """
    fs = find_files(data_path, '*.sac', ignore_case=True)
    if channel is None:
        channel = ['Z']
    date0 = getCP(param, 'basic', 'date', 'str')  # UTCDateTime(param.get('basic', 'date'))
    group = np.unique(record.location).tolist()
    siglst = []
    if update:
        status = 0
        update_prog('Loading SAC', [0, len(group)], status, 'Start')
    else:
        pbar = ProgressBar(widgets=['Loading: ', Percentage(), Bar('█')], maxval=len(group)).start()
    for i in range(len(group)):
        if update:
            update_prog('Loading SAC', [(i + 1) * 0.99, len(group)], status, '[{}]-{}'.format(i + 1, group[i]))
        else:
            pbar.update(i + 1)
        sub_record = record[record.location == group[i]]
        if geopsy:
            sigtab = np.recarray(shape=0, dtype=[('filename', '<U200'), ('format', '<U20'), ('original', '<U20'),
                                                 ('size', int), ('crc32', int),
                                                 ('ID', '<U20'), ('Name', '<U20'), ('Component', '<U20'),
                                                 ('StartTime', float), ('SamplingPeriod', float), ('Type', '<U20'),
                                                 ('NSamples', int), ('CountPerVolt', int), ('VoltPerUnit', int),
                                                 ('AmplitudeUnit', '<U20'), ('NumberInFile', int),
                                                 ('ReceiverX', float), ('ReceiverY', float), ('ReceiverZ', float),
                                                 ('TimeRange0', float), ('TimeRange1', float),
                                                 ('UtmZone', '<U20'), ('OffsetInFile', int), ('ByteIncrement', int)])
        else:
            sub_sig = Stream()
        sub_time0 = list()
        sub_time1 = list()
        if len(sub_record) == 1 and 'poly' in sub_record.xpoly_n[0]:
            n_loop = int(sub_record.xpoly_n[0].split('_')[-1]) + 1
            rhead = True
        else:
            try:
                [float(x) for x in sub_record.xpoly_n]
            except:
                if update:
                    update_logs('PythonException', 'Illegal coordinate input!')
                raise Exception('Illegal coordinate input!')
            n_loop = len(sub_record)
            rhead = False
        for j in range(n_loop):
            for k in range(len(channel)):
                if rhead:
                    fname = sub_record.station[0] + '.' + str(j + 1)
                    ftype = sub_record.network[0]
                else:
                    fname = sub_record.station[j]
                    ftype = sub_record.network[j]
                fn = find_sac(data_path, fname, ftype, channel[k], update=update, fs0=fs)
                if len(fn) == 0:
                    continue
                if geopsy:
                    tmpsig = SignalDatabase()
                    tmpsig.fromFile([str(x) for x in fn])
                    tmptab = tmpsig.toTab()
                else:
                    temp_sig = Stream()
                    for fi in fn:
                        temp_sig.append(read(str(fi), fsize=False)[0])
                    temp_sig.merge(fill_value='interpolate')
                # calibrating stats
                if geopsy:
                    try:
                        tmptab['SamplingPeriod'] = getCP(param, 'freq', 'dt', 'float')
                    except configparser.NoOptionError:
                        tmptab['SamplingPeriod'] = round(tmptab['SamplingPeriod'][0], 3)
                    tmptab['Name'] = fname
                    tmptab['ID'] = str(j + 1)
                    tmptab['Component'] = geopsy_component(channel[k])
                    # verifying if startime & sampling rate conflict
                    for itab in range(len(tmptab)):
                        tmptab['StartTime'][itab] = tmptab['StartTime'][itab] // 100 * 100 + \
                                                    round(tmptab['StartTime'][itab] % 100 /
                                                          tmptab['SamplingPeriod'][itab]) \
                                                    * tmptab['SamplingPeriod'][itab]
                else:
                    try:
                        temp_sig[0].stats.delta = getCP(param, 'freq', 'dt', 'float')
                    except configparser.NoOptionError:
                        temp_sig[0].stats.sampling_rate = round(temp_sig[0].stats.sampling_rate)
                    temp_sig[0].stats.network = ftype
                    temp_sig[0].stats.station = fname
                    temp_sig[0].stats.location = group[i]
                    temp_sig[0].stats.channel = channel[k]
                    # verifying if startime & sampling rate conflict
                    temp_sig[0].stats.starttime = round(temp_sig[0].stats.starttime.timestamp *
                                                        temp_sig[0].stats.sampling_rate) \
                                                  / temp_sig[0].stats.sampling_rate
                # calibrating location
                if geopsy:
                    if rhead:
                        if j == 0:
                            tmptab['ReceiverX'] = 0.0
                            tmptab['ReceiverY'] = 0.0
                            if 'z' in sub_record.dtype.names:
                                tmptab['ReceiverZ'] = float(sub_record.z[0])
                            else:
                                tmptab['ReceiverZ'] = 0.0
                                if update:
                                    update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                                else:
                                    warnings.warn('Record file depreciated! No Z coordinate availbale')
                        else:
                            tmptab['ReceiverX'] = np.sin(2 * np.pi * (j - 1) / (n_loop - 1)) \
                                                  * float(sub_record.yradius[0])
                            tmptab['ReceiverY'] = np.cos(2 * np.pi * (j - 1) / (n_loop - 1)) \
                                                  * float(sub_record.yradius[0])
                            if 'z' in sub_record.dtype.names:
                                tmptab['ReceiverZ'] = float(sub_record.z[0])
                            else:
                                tmptab['ReceiverZ'] = 0.0
                                if update:
                                    update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                                else:
                                    warnings.warn('Record file depreciated! No Z coordinate availbale')
                    else:
                        tmptab['ReceiverX'] = float(sub_record.xpoly_n[j])
                        tmptab['ReceiverY'] = float(sub_record.yradius[j])
                        if 'z' in sub_record.dtype.names:
                            tmptab['ReceiverZ'] = float(sub_record.z[j])
                        else:
                            tmptab['ReceiverZ'] = 0.0
                            if update:
                                update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                            else:
                                warnings.warn('Record file depreciated! No Z coordinate availbale')

                else:
                    # station X, Y, Z coordinates are stored in SAC header user7/user8/user9
                    if rhead:
                        if j == 0:
                            temp_sig[0].stats.sac.user7 = 0.0
                            temp_sig[0].stats.sac.user8 = 0.0
                            if 'z' in sub_record.dtype.names:
                                temp_sig[0].stats.sac.user9 = float(sub_record.z[0])
                            else:
                                temp_sig[0].stats.sac.user9 = 0.0
                                if update:
                                    update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                                else:
                                    warnings.warn('Record file depreciated! No Z coordinate availbale')
                        else:
                            temp_sig[0].stats.sac.user7 = np.sin(2 * np.pi * (j - 1) / (n_loop - 1)) \
                                                          * float(sub_record.yradius[0])
                            temp_sig[0].stats.sac.user8 = np.cos(2 * np.pi * (j - 1) / (n_loop - 1)) \
                                                          * float(sub_record.yradius[0])
                            if 'z' in sub_record.dtype.names:
                                temp_sig[0].stats.sac.user9 = float(sub_record.z[0])
                            else:
                                temp_sig[0].stats.sac.user9 = 0.0
                                if update:
                                    update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                                else:
                                    warnings.warn('Record file depreciated! No Z coordinate availbale')
                    else:
                        temp_sig[0].stats.sac.user7 = float(sub_record.xpoly_n[j])
                        temp_sig[0].stats.sac.user8 = float(sub_record.yradius[j])
                        if 'z' in sub_record.dtype.names:
                            temp_sig[0].stats.sac.user9 = float(sub_record.z[j])
                        else:
                            temp_sig[0].stats.sac.user9 = 0.0
                            if update:
                                update_logs('Record file depreciated', 'warning', 'No Z coordinate availbale')
                            else:
                                warnings.warn('Record file depreciated! No Z coordinate availbale')
                # get the SAC data starttime/endtime
                if geopsy:
                    timeval0 = format(min(tmptab['TimeRange0']), '.3f')
                    timeval1 = format(max(tmptab['TimeRange1']), '.3f')
                    sactime0 = UTCDateTime('T'.join([timeval0[:8], timeval0[8:]]))
                    sactime1 = UTCDateTime('T'.join([timeval1[:8], timeval1[8:]]))
                else:
                    sactime0 = temp_sig[0].stats.starttime
                    sactime1 = temp_sig[0].stats.endtime
                if (sactime1 - sactime0) / 3600 >= 24:
                    if update:
                        update_logs('Time record', 'warning', 'The record time is longer than 24 hours')
                    else:
                        warnings.warn('Total data longer than 24 hours! Might cause problems...')
                # if no record time set, use SAC data time
                if sub_record.starttime[0] == '' or sub_record.endtime[0] == '':
                    temp_time0 = sactime0
                    temp_time1 = sactime1
                # if record time available, calibrate within data time range
                else:
                    if rhead:
                        ist = ':'.join('{:0>2d}'.format(int(x)) for x in sub_record.starttime[0].split(':'))
                        iet = ':'.join('{:0>2d}'.format(int(x)) for x in sub_record.endtime[0].split(':'))
                    else:
                        ist = ':'.join('{:0>2d}'.format(int(x)) for x in sub_record.starttime[j].split(':'))
                        iet = ':'.join('{:0>2d}'.format(int(x)) for x in sub_record.endtime[j].split(':'))

                    temp_time0 = UTCDateTime(date0 + 'T' + ist + '+08')
                    temp_time1 = UTCDateTime(date0 + 'T' + iet + '+08')

                    if temp_time0 > temp_time1:
                        temp_time1 += 24 * 3600

                    tolerence = 0
                    while not (temp_time0 >= sactime0) & (temp_time0 <= sactime1) \
                              & (temp_time1 >= sactime0) & (temp_time1 <= sactime1):
                        if (temp_time0 > sactime1) & (temp_time1 > sactime1):
                            temp_time0 -= 24 * 3600
                            temp_time1 -= 24 * 3600
                        elif (temp_time0 < sactime0) & (temp_time1 < sactime0):
                            temp_time0 += 24 * 3600
                            temp_time1 += 24 * 3600
                        else:
                            if update:
                                update_logs('PythonException', 'error', 'Time record & data no match!')
                            raise Exception('Time record & data no match!')
                        tolerence += 1
                        if tolerence > 10:
                            if update:
                                update_logs('PythonException', 'error',
                                            'Time calibration failed! Please check your time record!')
                            raise Exception('Time calibration failed! Please check your time record!')
                # combining array time & signal
                sub_time0.append(temp_time0)
                sub_time1.append(temp_time1)
                if geopsy:
                    sigtab = np.concatenate([sigtab, tmptab])
                else:
                    sub_sig.append(temp_sig[0])
        # preprocessing signal
        if len(sub_time0) == 0 or len(sub_time1) == 0:
            if update:
                update_logs('PythonException', 'No file read! Record & data directory no match!')
            raise Exception('No file read! Record & data directory no match!')
        if geopsy:
            sigtab['TimeRange0'] = max(sub_time0).datetime.strftime('%Y%m%d%H%M%S.%f')
            sigtab['TimeRange1'] = min(sub_time1).datetime.strftime('%Y%m%d%H%M%S.%f')
            sub_sig = SignalDatabase()
            sub_sig.fromTab(sigtab)
            sub_sig.MasterGroup = group[i]
            siglst.append(sub_sig)
        else:
            '''
            # TODO: try fixing edge problems --> move to main program later
            sub_sig.trim(max(sub_time0), min(sub_time1), pad=True, fill_value='interpolate')
            sta_temp = [stalta(sub_sig[i], 1 / 500, sta0, lta0) for i in range(len(sub_sig))]
            for j in range(len(sub_sig)):
                try:
                    sub_time0[j] += (np.argwhere(sta_temp[j][1][:round(len(sta_temp[j][1]) / 2)]
                                                 > np.median(sta_temp[j][1]) * 1e3)).max() * sta0 + lta0 / 2
                    sub_time1[j] -= (np.argwhere(sta_temp[j][1][round(len(sta_temp[j][1]) / 2):]
                                                 > np.median(sta_temp[j][1]) * 1e3)).min() * sta0 + lta0 / 2
                except:
                    continue
            '''
            sub_sig.trim(max(sub_time0), min(sub_time1), pad=True, fill_value='interpolate')
            sub_sig.detrend('demean')
            sub_sig.normalize(global_max=True)  # TODO: is nromalize necessary?
            siglst.append(sub_sig)
    if update:
        update_prog('Loading SAC', [len(group), len(group)], 1, 'Complete')
    else:
        pbar.finish()
        print('Project SAC loaded')
    return siglst


def load_param(param, verbose=False):
    """Load calculation parameters via ConfigParser

    Parameters
    ----------
    param: configparser.ConfigParser
    verbose: bool
    """
    # window settings
    if getCP(param, 'freq', 'log', 'bool'):
        freq = np.geomspace(getCP(param, 'freq', 'freq_from', 'float'), getCP(param, 'freq', 'freq_to', 'float'),
                            getCP(param, 'freq', 'freq_n', 'int'))
        # np.geomspace(param.getfloat('freq', 'freq_from'), param.getfloat('freq', 'freq_to'),
        #              param.getint('freq', 'freq_n'))
    else:
        freq = np.linspace(getCP(param, 'freq', 'freq_from', 'float'), getCP(param, 'freq', 'freq_to', 'float'),
                           getCP(param, 'freq', 'freq_n', 'int'))
        # np.linspace(param.getfloat('freq', 'freq_from'), param.getfloat('freq', 'freq_to'),
        #             param.getint('freq', 'freq_n'))

    try:
        dt = getCP(param, 'freq', 'dt', 'float')  # param.getfloat('freq', 'dt')
    except configparser.NoOptionError:
        dt = None  # 1/round(siglst[0][0].stats.sampling_rate)
    bandwidth = getCP(param, 'xcor', 'bandwidth', 'float')  # param.getfloat('spac', 'bandwidth')
    nf = getCP(param, 'xcor', 'nf', 'int')  # param.getint('spac', 'nf')
    win_size = getCP(param, 'xcor', 'win_size', 'float')  # param.getfloat('spac', 'win_size')
    overlap = getCP(param, 'xcor', 'overlap', 'float')  # param.getfloat('spac', 'overlap')
    if verbose:
        print('Sampling period: {} sec'.format(dt))
        print('Central frequency: {:.1f} - {:.1f} Hz (n = {:.0f}, log: {})'
              .format(freq[0], freq[-1], len(freq), getCP(param, 'freq', 'log', 'bool')))
        # param.getboolean('freq', 'log')))
        print('Frequency bandwidth: {:.1f}%'.format(bandwidth * 100))
        print('Frequency points: {:.0f}'.format(nf))
        print('Window size: {:.0f} T'.format(win_size))
        print('Window overlap: {:.1f}%'.format(overlap * 100))
    # sta/lta settings
    sta = getCP(param, 'anti-trigger', 'sta', 'float')  # param.getfloat('anti-trigger', 'sta')
    lta = getCP(param, 'anti-trigger', 'lta', 'float')  # param.getfloat('anti-trigger', 'lta')
    atrig_range = [getCP(param, 'anti-trigger', 'atrig_from', 'float'),
                   getCP(param, 'anti-trigger', 'atrig_to', 'float')]
    # [param.getfloat('anti-trigger', 'atrig_from'), param.getfloat('anti-trigger', 'atrig_to')]
    if verbose:
        print('STA: {:.1f} sec'.format(sta))
        print('LTA: {:.1f} sec'.format(lta))
        print('STA/LTA accept range: {:.1f} - {:.1f}'.format(atrig_range[0], atrig_range[1]))
    return dt, freq, bandwidth, win_size, overlap, nf, sta, lta, atrig_range


def getCP(cp, section, option, valtype=None):
    """Get the config detail via ConfigParser

    Parameters
    ----------
    cp: configparser.ConfigParser
    section: str
    option: str
    valtype: str
    """
    if valtype:
        try:
            if valtype == 'int':
                val = cp.getint(section, option)
            if valtype == 'float':
                val = cp.getfloat(section, option)
            if valtype == 'bool':
                val = cp.getboolean(section, option)
            if valtype == 'str':
                val = cp.get(section, option)
        except ValueError:
            ival = cp.get(section, option)
            if '"' in ival:
                ival = ival.strip('"')
            try:
                if valtype == 'int':
                    val = int(ival)
                if valtype == 'float':
                    val = float(ival)
                if valtype == 'bool':
                    if ival.lower() == 'true':
                        val = True
                    elif ival.lower() == 'false':
                        val = False
                    else:
                        val = bool(int(ival))
            except ValueError:
                val = None
    else:
        val = cp.get(section, option)
    if type(val) is str and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
    return val


def prep_init(proj, data_root, para_root, prep_root, resl_root, update=False):
    # path setup
    if update:
        update_init()
        update_prog('Initialisation', [0, 6], 0, 'Preparing project')
    else:
        print_line('Preparing project')
    # verifying input paths
    data_path = [x for x in data_root.iterdir() if proj in str(x)]
    para_path = [x for x in para_root.iterdir() if proj in str(x)]
    if len(data_path) > 1 or len(para_path) > 1:  # TODO: Verify stability
        data_path = [x for x in data_root.iterdir() if proj == x.stem]
        para_path = [x for x in para_root.iterdir() if proj == x.stem]
        if len(data_path) != 1 or len(para_path) != 1:
            if update:
                update_logs('PythonException', 'error', 'Project name duplicated!')
            raise Exception('Project name ambiguous!')
    elif len(data_path) == 0 or len(para_path) == 0:
        if update:
            update_logs('PythonException', 'Project setup insufficient!')
        raise Exception('Project setup insufficient!')
    data_path = data_path[0]
    para_path = para_path[0]
    assert data_path.name == para_path.name
    proj = para_path.name
    proj_date = proj[:8]
    # verifying output paths
    prep_path = [x for x in prep_root.iterdir() if proj in str(x)]
    resl_path = [x for x in resl_root.iterdir() if proj in str(x)]
    if len(resl_path) == 0:
        resl_path = resl_root.joinpath(proj)
        resl_path.mkdir()
    else:
        resl_path = resl_path[0]
    if update:
        update_prog('Initialisation', [1, 6], 0, 'Project ready: {}'.format(proj))
    else:
        print('Project ready: {}'.format(proj))

    # default setup
    assert para_root.joinpath('defaults').exists()
    default_para = ConfigParser()
    default_para.read(str(para_root.joinpath('defaults').joinpath('param.ini')))
    if update:
        update_prog('Initialisation', [2, 6], 0, 'Initialsing project setup')
    else:
        print('Default setup complete')

    # project setup
    proj_para = ConfigParser()
    proj_para.read(str(para_path.joinpath('param.ini')))
    for i in range(len(default_para.sections())):
        if not default_para.sections()[i] in proj_para.sections():
            proj_para.read_dict({default_para.sections()[i]: dict(default_para.items(default_para.sections()[i]))})
    if not proj_para.has_option('basic', 'date'):
        proj_para.set('basic', 'date', proj_date)
    assert len(list(para_path.glob('*record.csv'))) == 1
    proj_recd = np.recfromcsv(str(list(para_path.glob('*record.csv'))[0]),
                              encoding='UTF-8-sig', dtype=','.join(['U20'] * 8))
    if getCP(proj_para, 'basic', 'save_sac', 'bool'):
        if len(prep_path) == 0:
            prep_path = prep_root.joinpath(proj)
            prep_path.mkdir()
        else:
            prep_path = prep_path[0]
    if update:
        update_prog('Initialisation', [3, 6], 0, 'Loading parameters')
    else:
        print('Project setup complete')
    return data_path, prep_path, resl_path, proj_recd, proj_para
