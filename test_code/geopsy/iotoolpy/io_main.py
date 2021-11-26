import fnmatch
import re
import configparser
import numpy as np
from pathlib import Path
from progressbar import ProgressBar, Bar, Percentage
from obspy import read, Stream, UTCDateTime

from xcorpy.base_func import stalta, update_msg


def find_path(path, key):
    assert type(path) == str
    assert type(key) == str or type(key) == list
    p = [Path(path)]
    if type(key) == str:
        key = [key]
    for ikey in key:
        if len(p) == 1:
            p = list(p[0].rglob(ikey))
        else:
            p = [p0 for p0 in p if ikey in p0.parts]
    if len(p) == 1:
        p_out = str(p[0])
    else:
        p_out = None
    return p_out


def find_files(path: str, glob_pat: str, ignore_case: bool = False):
    rule = re.compile(fnmatch.translate(glob_pat), re.IGNORECASE) if ignore_case \
        else re.compile(fnmatch.translate(glob_pat))
    return [n for n in Path(path).rglob('*') if rule.match(str(n))]


def find_sac(path, name=None, ftype=None, chn='Z', verbal=False):
    plen = len(str(path))
    if ftype is None or ftype == '':
        fs = list(path.glob('*.sac'))
    else:
        fs = list(path.rglob('*.sac'))
        if ftype == 'DTCC' or ftype == 'dtcc':
            fs = [x for x in fs if 'dtcc' in str(x.parent)[plen:].lower()]
            if any('bdfsac' in str(x.parent)[plen:].lower() for x in fs) and \
                    any('sac' in str(x.parent)[plen:].lower() for x in fs):
                fs = [x for x in fs if 'bdfsac' in str(x.parent)[plen:].lower()]
            if chn not in ['E', 'N', 'Z']:
                if chn == 'X':
                    chn = 'E'
                elif chn == 'Y':
                    chn = 'N'
                else:
                    raise Exception('Illegal channel for DTCC!')
        elif ftype == 'CGE' or ftype == 'cge':
            fs = [x for x in fs if 'cge' in str(x.parent)[plen:].lower()]
            if any('sac' in str(x.parent)[plen:].lower() for x in fs) and \
                    any('raw' in str(x.parent)[plen:].lower() for x in fs):
                fs = [x for x in fs if 'sac' in str(x.parent)[plen:].lower()]
            if chn not in ['X', 'Y', 'Z']:
                if chn == 'E':
                    chn = 'X'
                elif chn == 'N':
                    chn = 'Y'
                else:
                    raise Exception('Illegal channel for CGE!')
        elif ftype == 'JST' or ftype == 'jst':
            fs = [x for x in fs if 'jst' in str(x.parent)[plen:].lower()]
            if any('sac' in str(x.parent)[plen:].lower() for x in fs) and \
                    any('raw' in str(x.parent)[plen:].lower() for x in fs):
                fs = [x for x in fs if 'sac' in str(x.parent)[plen:].lower()]
            if chn != 'Z':
                raise Exception('Illegal channel for JST!')
        else:
            raise Exception('Instrument type not recognized!')
    if name:
        if '_' in name:
            assert len(name.split('_')) == 2
            n0 = name.split('_')[0]
            n1 = name.split('_')[1]
        else:
            n0 = name
            n1 = None
        _name = [x.stem for x in fs]
        fn = [x for x in fs if n0 in x.parts or n0 in _name]
        fn = fs if len(fn) == 0 else fn
        fn = [x for x in fn if n1 in x.parts]
    else:
        fn = fs
    if any(chn in re.split('[._]', x.stem) for x in fn):
        fn = [x for x in fn if chn in re.split('[._]', x.stem)]
    else:
        print('Channel {} no found!'.format(chn)) if verbal else None
    if len(fn) == 0:
        print('Station {} no found!'.format(name)) if verbal else None
    return fn


# reading microtremor settings
# todo: refine loading sequence
def prep_project(data_path, record, param, channel=None, update=False):
    if channel is None:
        channel = list('Z')
    sta0 = float(param.get('anti-trigger', 'sta'))
    lta0 = float(param.get('anti-trigger', 'lta'))
    date0 = UTCDateTime(param.get('basic', 'date'))
    group = np.unique(record.location).tolist()
    siglst = []
    if update:
        status = 0
        update_msg('Loading sac: Start', [0, len(group)], status, 'do push job start')
    else:
        pbar = ProgressBar(widgets=['Loading: ', Percentage(), Bar('█')], maxval=len(group)).start()
    for i in range(len(group)):
        if update:
            update_msg('Loading sac: preping {}'.format(i), [i, len(group)], status, 'do push job process')
        else:
            pbar.update(i+1)
        sub_record = record[record.location == group[i]]
        sub_sig = Stream()
        sub_time0 = list()
        sub_time1 = list()
        if len(sub_record) == 1:
            assert 'poly' in sub_record.xpoly_n[0]
            n_loop = int(sub_record.xpoly_n[0].split('_')[-1]) + 1
            rhead = True
        else:
            try:
                [float(x) for x in sub_record.xpoly_n]
            except:
                raise Exception('Illegal coordinate input!')
            n_loop = len(sub_record)
            rhead = False
        for j in range(n_loop):
            for k in range(len(channel)):
                if rhead:
                    fname = sub_record.station[0] + '_' + str(j + 1)
                    ftype = sub_record.network[0]
                else:
                    fname = sub_record.station[j]
                    ftype = sub_record.network[j]
                fn = find_sac(data_path, fname, ftype, channel[k])
                if len(fn) == 0:
                    continue
                temp_sig = Stream()
                for fi in fn:
                    temp_sig.append(read(str(fi), 'sac')[0])
                temp_sig.merge(fill_value='interpolate')
                # calibrating stats
                try:
                    temp_sig[0].stats.delta = param.getfloat('freq', 'dt')
                except configparser.NoOptionError:
                    temp_sig[0].stats.sampling_rate = round(temp_sig[0].stats.sampling_rate)
                temp_sig[0].stats.network = ftype
                temp_sig[0].stats.station = fname
                temp_sig[0].stats.location = group[i]
                temp_sig[0].stats.channel = channel[k]
                # verifying if startime & sampling rate conflict
                temp_sig[0].stats.starttime = round(temp_sig[0].stats.starttime.timestamp *
                                                    temp_sig[0].stats.sampling_rate) / temp_sig[0].stats.sampling_rate
                # calibrating location
                if rhead:
                    if j == 0:
                        temp_sig[0].stats.stlo = 0
                        temp_sig[0].stats.stla = 0
                    else:
                        temp_sig[0].stats.stlo = np.sin(2 * np.pi * (j - 1) / (n_loop - 1)) * float(
                            sub_record.yradius[0])
                        temp_sig[0].stats.stla = np.cos(2 * np.pi * (j - 1) / (n_loop - 1)) * float(
                            sub_record.yradius[0])
                else:
                    temp_sig[0].stats.stlo = float(sub_record.xpoly_n[j])
                    temp_sig[0].stats.stla = float(sub_record.yradius[j])
                temp_sig[0].stats.sac.user7 = temp_sig[0].stats.stlo
                temp_sig[0].stats.sac.user8 = temp_sig[0].stats.stla
                # calibrating time
                if rhead:
                    ist = sub_record.starttime[0]
                    iet = sub_record.endtime[0]
                else:
                    ist = sub_record.starttime[j]
                    iet = sub_record.endtime[j]
                temp_time0 = UTCDateTime(date0.isoformat()[:date0.isoformat().index('T') + 1] + ist + '+08')
                temp_time1 = UTCDateTime(date0.isoformat()[:date0.isoformat().index('T') + 1] + iet + '+08')
                while not (temp_time0 >= temp_sig[0].stats.starttime) & (temp_time0 <= temp_sig[0].stats.endtime) \
                          & (temp_time1 >= temp_sig[0].stats.starttime) & (temp_time1 <= temp_sig[0].stats.endtime):
                    if (temp_time0 > temp_sig[0].stats.endtime) & (temp_time1 > temp_sig[0].stats.endtime):
                        temp_time0 -= 24 * 3600
                        temp_time1 -= 24 * 3600
                    elif (temp_time0 < temp_sig[0].stats.starttime) & (temp_time1 < temp_sig[0].stats.starttime):
                        temp_time0 += 24 * 3600
                        temp_time1 += 24 * 3600
                    else:
                        raise Exception('Time record & data no match!')
                # combining array time & signal
                sub_time0.append(temp_time0)
                sub_time1.append(temp_time1)
                sub_sig.append(temp_sig[0])
        # preprocessing signal
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
        sub_sig.trim(max(sub_time0), min(sub_time1), pad=True, fill_value='interpolate')
        sub_sig.detrend('demean')
        sub_sig.normalize(global_max=True)
        siglst.append(sub_sig)
    if update:
        status = 1
        update_msg('Loading sac: Complete', [len(group), len(group)], status, 'do push job success')
    else:
        print('\nProject sac loaded!')
    return siglst


# importing calculation settings
def load_param(param, verbose=False):
    # window settings
    if param.getboolean('freq', 'log'):
        freq = np.geomspace(param.getfloat('freq', 'freq_from'), param.getfloat('freq', 'freq_to'),
                            param.getint('freq', 'freq_n'))
    else:
        freq = np.linspace(param.getfloat('freq', 'freq_from'), param.getfloat('freq', 'freq_to'),
                           param.getint('freq', 'freq_n'))

    try:
        dt = param.getfloat('freq', 'dt')
    except configparser.NoOptionError:
        dt = None  # 1/round(siglst[0][0].stats.sampling_rate)
    bandwidth = param.getfloat('spac', 'bandwidth')
    nf = param.getint('spac', 'nf')
    win_size = param.getfloat('spac', 'win_size')
    overlap = param.getfloat('spac', 'overlap')
    if verbose:
        print('Sampling period: {} sec'.format(dt))
        print('Central frequency: {:.1f} - {:.1f} Hz (n = {:.0f}, log: {})'
              .format(freq[0], freq[-1], len(freq), param.getboolean('freq', 'log')))
        print('Frequency bandwidth: {:.1f}%'.format(bandwidth * 100))
        print('Frequency points: {:.0f}'.format(nf))
        print('Window size: {:.0f} T'.format(win_size))
        print('Window overlap: {:.1f}%'.format(overlap * 100))
    # sta/lta settings
    sta = param.getfloat('anti-trigger', 'sta')
    lta = param.getfloat('anti-trigger', 'lta')
    atrig_range = [param.getfloat('anti-trigger', 'atrig_from'), param.getfloat('anti-trigger', 'atrig_to')]
    if verbose:
        print('STA: {:.1f} sec'.format(sta))
        print('LTA: {:.1f} sec'.format(lta))
        print('STA/LTA accept range: {:.1f} - {:.1f}'.format(atrig_range[0], atrig_range[1]))
    return dt, freq, bandwidth, win_size, overlap, nf, sta, lta, atrig_range
