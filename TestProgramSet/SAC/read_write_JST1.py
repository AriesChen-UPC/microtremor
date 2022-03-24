
import obspy.core.trace as tr
from obspy import UTCDateTime
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os


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

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:',Folderpath)
filetype_dat = '.dat'

def get_dat_filename(Folderpath,filetype_dat):
    name = []
    final_name = []
    for root,dirs,files in os.walk(Folderpath):
        for i in files:
            if filetype_dat in i:
                 name.append(i.replace(filetype_dat,''))
    final_name = [item + '.dat' for item in name]
    return final_name

FileName_dat = get_dat_filename(Folderpath,filetype_dat)
print('The names of Raw data are:','\n',FileName_dat)

num = len(FileName_dat)

for i in range(num):
    names_dat = locals()
    names_dat['st%s' % i] = read_JST_int(Folderpath + '/' + FileName_dat[i])

if num == 1:
    st = names_dat['st%s' % 0];
else:
    st = names_dat['st%s' % 0];
    for i in range(1,num):
        st += names_dat['st%s' % i];

if isinstance(st.data, np.ma.masked_array):
    st.data = st.data.filled()
st.data[st.data >= 1e+20] = 0

Data_merge_name = input('Please input the Synthetic data name:')
Data_merge_name = Data_merge_name + '.SAC'
print('The Synthetic data name is:','\n',Data_merge_name)
file_path_merge = Folderpath + '/' + Data_merge_name
st.write(file_path_merge, format='sac')
