'''
Collection of function that to be used for data processing of microtremor data by using Horizontal to Vertical Spectrum Ratio method.
There are some functions such as:

nearest     : to search the nearest value from an array
dofilt      : to filter a waveform [band, low and high filters]
coswindow   : to apply the cosine windowing to the trace
              (Tapering of windowed time series, https://gfzpublic.gfz-potsdam.de/rest/items/item_56141/component/file_56140/content)
antrig      : to select the trace of microtremor based on STA/LTA method
            (https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content)
kosmooth    : to smooth the trace by using Konno-Ohmachi smoothing method
              (https://pubs.geoscienceworld.org/ssa/bssa/article/88/1/228/102764)     
'''

import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import math as m
np.seterr(divide='ignore', invalid='ignore')

def nearest(arrayseries, value):
    arrayseries = np.asarray(arrayseries)
    idx = (np.abs(arrayseries - value)).argmin()
    return arrayseries[idx]

class dofilt:
    def band(data,lowf,highf,fs,order):
        nyq = fs/2
        low = lowf/nyq
        high = highf/nyq
        b,a = butter(order, [low,high], btype='band',analog=False)
        y = filtfilt(b,a,data,padtype='odd',padlen=3*(max(len(b),len(a))-1))
        return y

    def low(data,lowf,fs,order):
        nyq = fs/2
        low = lowf/nyq
        b,a = butter(order, low, btype = 'low',analog=False)
        y = filtfilt(b,a,data,padtype='odd',padlen=3*(max(len(b),len(a))-1))
        return y

    def high(data,highf,fs,order):
        nyq = fs/2
        high = highf/nyq
        b,a = butter(order, high, btype = 'high',analog=False)
        y = filtfilt(b,a,data,padtype='odd',padlen=3*(max(len(b),len(a))-1))
        return y

def coswindow(a,n):
    t = np.linspace(0,1,n)
    # Section 1
    i = 0
    res = []
    while t[i] >= 0.0 and t[i] <= a:
        c1 = (1-m.cos(m.pi/a*t[i]))/2
        res.append(c1)
        i = i + 1

    # Section 2
    j = 0
    while t[j] <= a:
        j = j + 1
        while t[j] >= a and t[j] <= (1-a):
            c2 = 1.0
            res.append(c2)
            j = j + 1

    # Section 3
    k = 0
    while t[k] <= (1-a):
        k = k + 1
    kk = k+1
    while kk <= n:
        c3 = (1-m.cos(m.pi/a*(1-t[kk-1])))/2
        res.append(c3)
        kk = kk + 1
    return res

def antrig(traceinput,sta,lta,delta_t,minthres,maxthres,winlength):
    ns = int(sta/delta_t)
    nl = int(lta/delta_t)
    n_trace = len(traceinput)
    abstrace = abs(traceinput)
    # timest = np.linspace(0,n_trace)*delta_t

    # STA
    nsta = []
    i = 0
    for i in range(n_trace):
        n_sta = np.mean(abstrace[i:ns])
        nsta.append(n_sta)
        ns = ns + 1
        if (ns > n_trace):
            ns = n_trace

    # LTA
    nlta = []
    i = 0
    for i in range(n_trace):
        n_lta = np.mean(abstrace[i:nl])
        nlta.append(n_lta)
        nl = nl + 1
        if (nl > n_trace):
            nl = n_trace

    slratio = []
    for i in range(n_trace):
        ra = nsta[i]/nlta[i]
        slratio.append(ra)

    slr = pd.DataFrame(slratio,columns=['SL_Ratio'])

    # Anti-triger process
    mindur = int(winlength/delta_t)
    stepwin = int(winlength/delta_t)
    id = 0
    ntrig = 0
    start_trig = []
    end_trig = []
    while (mindur <= n_trace):
        check1 = all(val1 >= minthres for val1 in slratio[id:mindur])
        check2 = all(val2 <= maxthres for val2 in slratio[id:mindur])
        if (check1 == True):
            if (check2 == True):
                strig = id
                etrig = mindur
                ntrig = ntrig + 1
                start_trig.append(strig)
                end_trig.append(etrig)
        else:
            id = id + stepwin
            mindur = mindur + stepwin

        id = id + stepwin
        mindur = mindur + stepwin

    store_event = [start_trig,end_trig]
    event = pd.DataFrame(store_event, index = ['Start Trigger', 'End Trigger']).T

    return slratio, slr, store_event, event, ntrig

def kosmooth(data,bandwidth):
    L = len(data)
    freqseries = np.arange(0,L)
    frshifted = freqseries/(1+1e-4)
    smooth = np.zeros(L)
    w = np.zeros(L)
    for i in range(L):
        fc = freqseries[i]
        z = np.divide(frshifted,fc)**bandwidth
        w = (np.sin(np.log10(z))/np.log10(z)) ** 4.0
        w[np.isnan(w)] = 0
        smooth[i] = np.dot(w,data)/np.sum(w)
        smooth[np.isnan(smooth)] = 0
    return smooth
