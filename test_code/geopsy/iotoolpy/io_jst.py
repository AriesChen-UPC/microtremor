import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
from obspy import UTCDateTime

from iotoolpy.io_array import dict_to_header_arrays, write_sac


def verify_filename(filename):
    fname = os.path.basename(filename)
    file_id = fname.split('.')[0]
    t_raw = int(file_id[:-5])  # unit: ms
    status = file_id[-5:]
    check = bool(status[-1])
    sampling_rate = int(status[:-1])
    return t_raw, check, sampling_rate


def read_JST_int0(file, sta='', net='', chn='BHZ'):
    t0, chk, sampling_rate = verify_filename(file)
    if not chk:
        print('File {} with bad record\nReturning starting time...'.format(file))
        return t0
    else:
        st = UTCDateTime(1980, 1, 6) + t0 / sampling_rate
        dt = 1 / sampling_rate
        n = int(os.path.getsize(file) / 4)
        stat0 = {'npts': n, 'delta': dt, 'b': 0, 'e': (n - 1) * dt, 'st': st,
                 'kstnm': sta, 'knetwk': net, 'kcmpnm': chn,
                 'nzyear': st.year, 'nzjday': st.julday, 'nzhour': st.hour, 'nzmin': st.minute,
                 'nzsec': st.second, 'nzmsec': int(st.microsecond / 1e3)}
        with open(file, 'rb') as f:
            data0 = np.fromfile(f, dtype=np.int32) * 1e-6  # unit: mV
        return stat0, data0  # stat0 cannot be converted to header arrays


def get_dat_filename(path, filetype):
    name = []
    for _, _, files in os.walk(path):
        for fi in files:
            if filetype in fi:
                name.append(fi.replace(filetype, ''))
    final_name = [item + '.dat' for item in name]
    return final_name


def countdown(t):
    t = int(t)
    for it in range(t):
        print('Counting down [{:.0f}]'.format(t - it))


# initiate tkinter
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
filetype_dat = '.dat'
# get the files
FileName_dat = get_dat_filename(Folderpath, filetype_dat)
FileName_dat.sort()
print('The names of Raw data are:', '\n', FileName_dat)

# read raw files
num = len(FileName_dat)
stat = []
data = []
for i in range(num):
    temp = read_JST_int0(os.path.join(Folderpath, FileName_dat[i]))
    stat.append(temp[0])
    data.append(temp[1])

# combining data
# default sampling rate 1000
data_combined = np.zeros(int((stat[-1]['st'] - stat[0]['st'] + stat[-1]['e']) * 1e3 + 1))
for i in range(num):
    st_ind = int((stat[i]['st'] - stat[0]['st']) * 1e3)
    ed_ind = st_ind + stat[i]['npts']
    data_combined[st_ind:ed_ind] = data[i]

stat_combined = {'npts': len(data_combined), 'delta': stat[0]['delta'],
                 'b': 0, 'e': (len(data_combined) - 1) * stat[0]['delta'],
                 'nvhdr': 6, 'iztype': 9, 'idep': 4,  # 'scale': 1000,
                 'iftype': 1, 'leven': 1, 'lpspol': 1, 'lovrok': 1, 'lcalda': 0,
                 'kstnm': stat[0]['kstnm'], 'knetwk': stat[0]['knetwk'], 'kcmpnm': stat[0]['kcmpnm'],
                 'nzyear': stat[0]['nzyear'], 'nzjday': stat[0]['nzjday'], 'nzhour': stat[0]['nzhour'],
                 'nzmin': stat[0]['nzmin'], 'nzsec': stat[0]['nzsec'], 'nzmsec': stat[0]['nzmsec']}

# name input
fold_name = os.path.basename(Folderpath)
if stat_combined['kstnm'] == '':
    name_combined = input('Please input station name: (default: {})\n'.format(fold_name))
    if name_combined == '':
        name_combined = fold_name
else:
    name_combined = input('Please input station name: (default: {})\n'.format(stat_combined['kstnm']))
    if name_combined == '':
        name_combined = stat_combined['kstnm']
filename_combined = name_combined + '.sac'
stat_combined['kstnm'] = name_combined
filepath_combined = os.path.join(os.path.dirname(Folderpath), filename_combined)
hf, hi, hs = dict_to_header_arrays(stat_combined)
write_sac(filepath_combined, hf, hi, hs, data_combined)
print('The exported file is:\n', filepath_combined)
countdown(5)
