# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 3/12/2022 10:13 AM
@file: sac_file_process.py
"""

import os
from glob import glob
import numpy as np
from obspy import read
from obspy import UTCDateTime
import tkinter as tk
from tkinter import filedialog
from scipy import signal
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from itertools import combinations


def sac_file_detrend():
    """sac file detrend: detrend the sac file using obspy"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    file_name = glob(folder_path + '/*.SAC')
    # make a new folder to store the processed data
    sac_detrend = folder_path + '/sac_detrend'
    if not os.path.exists(sac_detrend):
        os.mkdir(folder_path + '/sac_detrend')
    # read and process the sac files
    for i in trange(len(file_name)):
        tr = read(file_name[i])
        tr.detrend("spline", order=3, dspline=500)
        tr.write(folder_path + '/sac_detrend/' + os.path.basename(file_name[i]).split('.')[7] + '_'
                 + os.path.basename(file_name[i]).split('.')[9] + '.SAC', format='SAC')
    print('sac data process done!')


def sac_file_fft():
    """sac file fft: fft the sac file using scipy"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    tr = read(file_path)
    signal = tr.traces[0].data
    # define the parameters
    length = len(signal)
    fs = int(1/tr.traces[0].meta.delta) + 1  # sampling frequency from the sac file in Stream format
    fft_signal = fft(signal)
    freq = fs * np.linspace(0, length, num=length) / length
    abs_fft_signal = np.abs(fft_signal)
    angle_fft_signal = np.angle(fft_signal)  # phase angle from fft
    # normalize the fft
    normalized_fft_signal = abs_fft_signal / length
    # half the fft
    half_freq = freq[0:int(length / 2)]
    normalized_half_fft_signal = normalized_fft_signal[0:int(length / 2)]
    plt.figure()
    plt.plot(half_freq, normalized_half_fft_signal)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Normalized FFT')
    plt.grid()
    plt.show()


def sac_file_slice():
    """sac file slice: slice the sac file using obspy"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    tr = read(file_path)
    print('The start time is:', tr.traces[0].stats.starttime)
    print('The end time is:', tr.traces[0].stats.endtime)
    start_time = UTCDateTime(input('Please input the start time, format in "2022-03-11T15:11:00" \n'))
    end_time = UTCDateTime(input('Please input the end time, format in "2022-03-11T15:11:00" \n'))
    tr_slice = tr.slice(start_time, end_time)
    return tr_slice


def sac_file_down_sample():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    tr = read(file_path)
    tr_new = tr.copy()
    tr_new.decimate(factor=4, strict_length=False)
    return tr_new


#%% get the destination file path in folder and subfolder

def get_list_files(folder_path):
    all_file_path = []
    for root, dirs, files in os.walk(folder_path):
        for files_path in files:
            all_file_path.append(os.path.join(root, files_path))
    return all_file_path


root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
all_file_path = get_list_files(folder_path)
z_component_path = []  # Z component sac file path for example
for i in all_file_path:
    if "Z.SAC" in i or "Z.Q.SAC" in i:  # sac file name format for CGS
        z_component_path.append(i)
print(z_component_path)

#%% slice the sac file

start_time = UTCDateTime(input('Please input the start time, format in yyyy-mm-ddThh:mm:ss \n'))
end_time = UTCDateTime(input('Please input the end time, format in yyyy-mm-ddThh:mm:ss \n'))
tr_slice_data = []
for i in trange(len(z_component_path)):
    tr = read(z_component_path[i])
    tr_slice = tr.slice(start_time, end_time)
    tr_slice.write(z_component_path[i].split('.')[0] + '_slice.SAC', format='SAC')
    tr_slice_data.append(tr_slice)

#%% edit the sac file and calculate the coherence between two traces

print('\033[0;31mPlease select the sac data of DTCC, time in UTC+0 : \033[0m')
tr_dtcc = sac_file_slice()
# tr_dtcc.write(sac_file_test + '/' + 'dtcc.SAC', format='SAC')
fs_dtcc = int(1/tr_dtcc.traces[0].meta.delta) + 1
print('\033[0;31mPlease select the sac data of Other, time in UTC+8 : \033[0m')
tr_other = sac_file_slice()
tr_other.decimate(factor=4, strict_length=False)  # downsample the sac file
# tr_other.write(sac_file_test + '/' + 'other.SAC', format='SAC')
fs_other = int(1/tr_other.traces[0].meta.delta) + 1

if fs_dtcc == fs_other:
    print('\033[0;32mThe sampling frequency is the same!\033[0m')
    print('The sampling frequency is:', fs_dtcc)
    f, Cxy = signal.coherence(tr_dtcc.traces[0].data, tr_other.traces[0].data, fs_dtcc, nperseg=1024)
    # plt.semilogy(f, Cxy)
    plt.plot(f, Cxy)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(0, 50)
    plt.xticks(np.arange(0, 51, 5))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.grid()
    plt.savefig('coherence.png')
    plt.show()
else:
    print('\033[0;31mThe sampling frequency is different!\033[0m')
    print('The sampling frequency of DTCC is:', fs_dtcc)
    print('The sampling frequency of Other is:', fs_other)

#%% calculate the coherence of each pair of traces and plot the average coherence

root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
file_name = glob(folder_path + '/*.SAC')
# read and process the sac files
file_name = glob(folder_path + '/*.SAC')
sac_file_name = []
sac_data = []
for i in trange(len(file_name)):
    sac_file_name.append(os.path.basename(file_name[i]).split('.')[0])
    sac_data.append(read(file_name[i]).traces[0].data)
# combine the sac file name and data, group by 2
sac_file_name_combine = list(combinations(sac_file_name, 2))
sac_data_combine = list(combinations(sac_data, 2))
# calculate the coherence for each pair of sac files
freq = []
CC = []
fs = int(read(file_name[0]).traces[0].stats.sampling_rate) + 1
for i in trange(len(sac_file_name_combine)):
    f, Cxy = signal.coherence(sac_data_combine[i][0], sac_data_combine[i][1], fs, nperseg=1024)
    freq.append(f)
    CC.append(Cxy)
# plot the coherence of each pair of sac files
for i in range(len(CC)):
    plt.plot(freq[i], CC[i])
    plt.xlim(0, 50)
    plt.xticks(np.arange(0, 51, 5))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.title(sac_file_name_combine[i][0] + ' and ' + sac_file_name_combine[i][1])
    plt.grid()
    plt.show()
# plot the coherence of average of all pairs
CC_avg = [np.mean(e) for e in zip(*CC)]
freq_avg = [np.mean(e) for e in zip(*freq)]
plt.plot(freq_avg, CC_avg)
plt.xlim(0, 50)
plt.xticks(np.arange(0, 51, 5))
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.title('Average of all pairs')
plt.grid()
plt.show()

#%% calculate the spectrogram

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
tr = read(file_path, format='SAC')  # todo: if not with "format='SAC'", some data will not be read correctly
# using signal.spectrogram
# ------------------------------
# freq_spec, t_spec, Sxx_spec = signal.spectrogram(tr.traces[0].data, int(1/tr.traces[0].meta.delta) + 1)
# plt.pcolormesh(t_spec, freq_spec, Sxx_spec, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim(0, 50)
# plt.show()
# ------------------------------
# using obspy Plotting Spectrograms
fig = plt.figure()
ax = plt.axes()
tr.spectrogram(log=True, dbscale=False, axes=ax)
plt.ylim(1, 100)
plt.title(os.path.basename(file_path).split('.')[0] + ' ' + str(tr.traces[0].stats.starttime))
plt.show()

#%% calculate the  power spectral density of each trace using Welch's method

for i in range(len(sac_data)):
    f, Pxx_den = signal.welch(sac_data[i], fs, nperseg=1024)
    plt.semilogx(f, Pxx_den)
plt.legend(sac_file_name, ncol=1, loc='upper left')
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
