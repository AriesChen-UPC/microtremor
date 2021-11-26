'''
# pySVIM
## Seismic Vulnerability Index of Microtremor - Microtremor Data Processing By Using Horizontal to Vertical Spectral Ratio (HVSR) Method

### Overview
A code to calculate the amplification factor and frequency dominant by using Horizontal to Vertical Spectral Ratio (HVSR) method.
The HVSR methods was introduced by [Nakamura in 1989](https://www.sdr.co.jp/papers/hv_1989.pdf). I have wrote code of SVIM in Matlab 
version with Graphical User Interface. The code calculate seismic vulnerability index directly after the parameters of amplification 
and frequency dominant was obtained from HVSR calculation and plot into map of sesimic vulnerability index directly too. 
This code is apart of [my thesis](https://etd.unsyiah.ac.id/index.php?p=show_detail&id=66042) and [my publication]
(https://iopscience.iop.org/article/10.1088/1755-1315/273/1/012016) in master degree  

However, the SVIM in Matlab version is still under construction to get the proper performance. So, for now, I can share the SVIM code 
in [Python3](https://www.python.org/download/releases/3.0/) version called pySVIM. pySVIM only provide the HVSR calculation or without 
the Graphical User Interface and without plot of seismic vulnerablitty index map.

### The library you need to install:
1. [Obspy](https://pypi.org/project/obspy/)
2. [Numpy](https://pypi.org/project/numpy/)
3. [Pandas](https://pypi.org/project/pandas/)
4. [Scipy](https://pypi.org/project/scipy/)

I also provide the extention library that I created called hvsrlib. This library concists some functions such as:

1. nearest     : to search the nearest value from an array
2. dofilt      : to filter a waveform (band, low and high filters)
3. coswindow   : to apply the cosine windowing to the trace [Tapering of windowed time series](https://gfzpublic.gfz-potsdam.de/rest/items/item_56141/component/file_56140/content)
4. antrig      : to select the trace of microtremor based on [STA/LTA method](https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content)
5. kosmooth    : to smooth the trace by using [Konno-Ohmachi smoothing method](https://pubs.geoscienceworld.org/ssa/bssa/article/88/1/228/102764)

### Contact
Aulia Khalqillah,S.Si.,M.Si
auliakhalqillah.mail@gmail.com
'''

from obspy import read
from numpy import median,add,divide,sqrt,isnan,argmax,zeros,array,arange,linspace,std,where
from scipy.signal import resample
from scipy.fftpack import fft
from hvsrlib import nearest, dofilt, coswindow, antrig, kosmooth
import matplotlib.pyplot as plt
import pandas as pd
import time

# Main Program
s_time = time.time()
dir = ['c0avf180318023636.bhe', 'c0avf180318023636.bhn', 'c0avf180318023636.bhz']
filename = [dir[0][-21:],dir[1][-21:],dir[2][-21:]]
name = filename[0][:-4]
wave = []
start_time = []
end_time = []
samplingrate = []
dt = []
number_sample = []
t = []
for i in range(len(dir)):
    signal = read(dir[i])
    trace = signal[0]
    w = trace.data
    wave.append(w)
    st = trace.stats.starttime
    start_time.append(st)
    et = trace.stats.endtime
    end_time.append(et)
    samprate = trace.stats.sampling_rate
    samplingrate.append(samprate)
    delta = trace.stats.delta
    dt.append(delta)
    numsamp = trace.stats.npts
    number_sample.append(numsamp)
    timeseries = arange(0.0,numsamp)*delta
    t.append(timeseries)

print("-----------------------------------------------------------------------")
print('DATA INFORMATION')
print("-----------------------------------------------------------------------")
# Convert to data frame
data = [filename,start_time,end_time,samplingrate,dt,number_sample]
colname = ['East Component', 'North Component', 'Vertical Component']
indexname = ['Directory','Start Time (UTC)','End Time (UTC)','Sampling Rate (Hz)','dt (sec)','Sample']
info = pd.DataFrame(data, columns = colname, index = indexname).T
print(info,"\n")

# Waveform filter
print("[1/7]>> Waveform Filter...")
lowcut = 0.1
highcut = 10
order = 1
wave_filtered = []
for j in range(len(dir)):
    y = dofilt.band(wave[j],lowcut,highcut,samplingrate[j],order)
    wave_filtered.append(y)

# Anti-triger STA/LTA
print("[2/7]>> Anti-triger by using STA/LTA...")
winlength = 25
maxthres = 2.5
minthres = 0.2
sta = 1
lta = 30
ratio = []
eventtrig = []
# antitriger for each component
for l in range(len(dir)):
    ratiosl, ratio_in_dataframe, eventtriger, event_trig_in_dataframe, ntrigcom = antrig(wave_filtered[l],sta,lta,dt[l],minthres,maxthres,winlength)
    ratio.append(ratiosl)
    eventtrig.append(eventtriger)

# Calculate ratio of STA/LTA by using median filtered signal
medianwavefiltered = [wave_filtered[0],wave_filtered[1],wave_filtered[2]]
median_wavefiltered = median(medianwavefiltered, axis=0)
median_ratiosl, median_ratio_in_dataframe, median_eventtriger, median_event_trig_in_dataframe, ntrigger = antrig(median_wavefiltered,sta,lta,dt[0],minthres,maxthres,winlength)
# median_event_trig_in_dataframe.to_csv(r'/Volumes/MYDRIVE/PYTHON/hvsrpython/event.csv',header=True,index=False)
# median_ratio_in_dataframe.to_csv(r'/Volumes/MYDRIVE/PYTHON/hvsrpython/ratio.csv',header=True,index=False)

window = 0
iwin = 0
newevent_start = []
newevent_end = []
while (iwin < ntrigger):
    ck1 = all(c1 >= minthres for c1 in ratio[0][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]]) and all(d1 <= maxthres for d1 in ratio[0][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]])
    ck2 = all(c2 >= minthres for c2 in ratio[1][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]]) and all(d2 <= maxthres for d2 in ratio[1][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]])
    ck3 = all(c3 >= minthres for c3 in ratio[2][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]]) and all(d3 <= maxthres for d3 in ratio[2][median_eventtriger[0][iwin]:median_eventtriger[1][iwin]])
    if (ck1 == True):
        if (ck2 == True):
            if (ck3 == True):
                nes = median_eventtriger[0][iwin]
                ned = median_eventtriger[1][iwin]
                newevent_start.append(nes)
                newevent_end.append(ned)
                window = window + 1
    iwin = iwin + 1

newevent = [newevent_start,newevent_end]
newevent_in_dataframe = pd.DataFrame(newevent, index = ['Start Trigger', 'End Trigger']).T

# Add Cosine Tapper for each window
print("[3/7]>> Apply cosine window...")
a = 0.5
ct = coswindow(a,int(winlength/dt[0]))
numdat = int(winlength/dt[0])
EAST = zeros((numdat,window))
NORTH = zeros((numdat,window))
VERTICAL = zeros((numdat,window))
for i in range(window):
    EAST[:,i] = wave_filtered[0][newevent_start[i]:newevent_end[i]]
    EAST[:,i] = (EAST[:,i]*ct)
    NORTH[:,i] = wave_filtered[1][newevent_start[i]:newevent_end[i]]
    NORTH[:,i] = (NORTH[:,i]*ct)
    VERTICAL[:,i] = wave_filtered[2][newevent_start[i]:newevent_end[i]]
    VERTICAL[:,i] = (VERTICAL[:,i]*ct)

# Add FFT
print("[4/7]>> FFT...")
nfft = 10240
eastspectrum = zeros((window,int(nfft)))
northspectrum = zeros((window,int(nfft)))
verticalspectrum = zeros((window,int(nfft)))
for i in range(window):
    eastspectrum[i,:] = abs(fft(EAST[:,i],nfft))
    northspectrum[i,:] = abs(fft(NORTH[:,i],nfft))
    verticalspectrum[i,:] = abs(fft(VERTICAL[:,i],nfft))
fre = linspace(0,samplingrate[0],int(nfft))
frn = linspace(0,samplingrate[1],int(nfft))
frv = linspace(0,samplingrate[2],int(nfft))

# Smoothing Kono-Ohmachi
print("[5/7]>> Smoothing Spectrum by using Konno-Ohmachi...")
minfrsmooth = 0
maxfrsmooth = 10 # in Hz
idmaxfrsmooth = int(where(fre == nearest(fre,maxfrsmooth))[0])
frsmoothstep = (maxfrsmooth-minfrsmooth)/idmaxfrsmooth
frsmooth = arange(minfrsmooth,maxfrsmooth,frsmoothstep)
e_smooth = zeros((window,idmaxfrsmooth))
n_smooth = zeros((window,idmaxfrsmooth))
z_smooth = zeros((window,idmaxfrsmooth))
b = float(40)
for i in range(window):
    e_smooth[i,:] = kosmooth(eastspectrum[i,0:idmaxfrsmooth],b)
    n_smooth[i,:] = kosmooth(northspectrum[i,0:idmaxfrsmooth],b)
    z_smooth[i,:] = kosmooth(verticalspectrum[i,0:idmaxfrsmooth],b)

# Calculate HVSR
print("[6/7]>> Calculation of HVSR...")
minfr = 0.1
maxfr = maxfrsmooth # in Hz
nos = 1024
frhvstep = (maxfr-minfr)/nos
frhv = arange(minfr,maxfr,frhvstep)
mergemethod = {1:'Square Average',2:"Geometric Mean"} # merging horizontal component [1: squared average, 2: geometric mean]
optionnumber = 1
mergeopt = mergemethod[optionnumber]
rathv = zeros((window,idmaxfrsmooth))
sumhor = zeros((window,idmaxfrsmooth))
hor = zeros((window,idmaxfrsmooth))
ver = zeros((window,idmaxfrsmooth))
re_rathv = zeros((window,nos))
for i in range(window):
    if (mergeopt == mergemethod[1]):
        hor[i,:] = sqrt(((e_smooth[i,:]**2) + (n_smooth[i,:]**2))/2) # square average
    elif( mergeopt == mergemethod[2]):
        hor[i,:] = sqrt((e_smooth[i,:]*n_smooth[i,:])) # geometric mean
    ver[i,:] = z_smooth[i,:]
    rathv[i,:] = divide(hor[i,:],ver[i,:])
    rathv[i,:][isnan(rathv[i,:])] = 0
    re_rathv[i,:] = resample(rathv[i,:],nos)
hvsr = median(rathv,axis=0)
hvsr = resample(hvsr,nos)
hvsr = kosmooth(hvsr,b)
stdhv = std(re_rathv,axis=0)
minstdhv = hvsr-stdhv
maxstdhv = hvsr+stdhv

# Take the maximum (amplification) of H/V
print("[7/7]>> Searching the amplification (H/V) and dominant frequency...\n")
id_A0 = argmax(hvsr)
A0 = max(hvsr)
F0 = frhv[id_A0]
KG = (A0**2)/F0
minstdA0 = minstdhv[id_A0]
maxstdA0 = maxstdhv[id_A0]
stddivA = maxstdhv/hvsr # std maximum of Amplification/Amplification Average
stdA = stddivA[id_A0]
f0allwin = zeros(window)
for i in range(window):
    idmaxhvallwin = argmax(re_rathv[i])
    f0allwin[i] = frhv[idmaxhvallwin]
stdf0 = std(f0allwin)
f0min = F0 - stdf0
f0max = F0 + stdf0
print("-----------------------------------------------------------------------")
print("OUTPUT INFORMATION OF H/V")
print("-----------------------------------------------------------------------")
print("Window Trigger\t\t:", window)
print("Maximum H/V\t\t:",A0)
print("Frequency H/V\t\t:",F0,"Hz")
print("Vulnerability\t\t:",KG)
print("Std Freq\t\t:",stdf0,"Hz[",f0min,"Hz,",f0max,"Hz]")
print("Std of H/V\t\t:","Min:",minstdA0,"Max:",maxstdA0)
print("Length of H/V Smoothing\t:", len(hvsr))
print("Merging Method\t\t:", mergeopt)
print("Allocate Time\t\t:",(time.time() - s_time),"second")

# Store output information to dictionary
outdict = {
            'filename':name,
            'maxfreq':max(frhv),
            'winlength':winlength,
            'window':window,
            'frhv':frhv,
            'hvsr':hvsr,
            'A0':A0,
            'F0':F0,
            'KG':KG,
            'stdA':stdA,
            'stdf0':stdf0,
            'stdhv':stdhv,
            'f0min':f0min,
            'f0max':f0max,
            'minstdhv':minstdhv,
            'maxstdhv':maxstdhv,
            'minstdA0':minstdA0,
            'maxstdA0':maxstdA0,
            'lengthhv':len(hvsr),
            'mergeopt':mergeopt,
            }

# # # PLOT
# # # Plot Waveform
plt.figure(1,figsize=(10,6))
for i in range(len(dir)):
    plt.subplot(3,1,i+1)
    plt.plot(t[i],wave[i],color='black',linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title(filename[i])
plt.xlabel('Time (sec)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)
# #
# # # Plot ratio of STA/LTA
plt.figure(2,figsize=(10,6))
for i in range(len(dir)):
    plt.subplot(3,1,i+1)
    plt.plot(t[i],ratio[i],color='blue',linewidth=0.5)
    plt.plot([t[i][newevent_in_dataframe.iloc[:,0]],t[i][newevent_in_dataframe.iloc[:,0]]],[min(ratio[i]),max(ratio[i])],color='black')
    plt.plot([t[i][newevent_in_dataframe.iloc[:,1]],t[i][newevent_in_dataframe.iloc[:,1]]],[min(ratio[i]),max(ratio[i])],color='red')
    plt.plot([min(t[i]),max(t[i])],[minthres, minthres],color='green')
    plt.plot([min(t[i]),max(t[i])],[maxthres, maxthres],color='gray')
    plt.ylabel('Amplitude')
    plt.title(filename[i]+' (STA/LTA Ratio)')
plt.xlabel('Time (sec)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)
# #
# # # Plot Waveform Filtered + Antitrigger
plt.figure(3,figsize=(10,6))
for i in range(len(dir)):
    plt.subplot(3,1,i+1)
    plt.plot(t[i],wave_filtered[i],color='red',linewidth=0.5)
    plt.plot([t[i][newevent_in_dataframe.iloc[:,0]],t[i][newevent_in_dataframe.iloc[:,0]]],[min(wave_filtered[i]),max(wave_filtered[i])],color='black')
    plt.plot([t[i][newevent_in_dataframe.iloc[:,1]],t[i][newevent_in_dataframe.iloc[:,1]]],[min(wave_filtered[i]),max(wave_filtered[i])],color='blue')
    plt.ylabel('Amplitude')
    plt.title(filename[i]+' (Filtered)')
plt.xlabel('Time (sec)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)
# #
# # H/V spectrum all window
plt.figure(4,figsize=(10,6))
plt.subplot(3,1,1)
for i in range(window):
    plt.plot(fre,eastspectrum[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('East Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)
plt.plot([fre[1024],fre[1024]],[min(eastspectrum[0]),max(eastspectrum[0])],color='blue',linewidth=2,label="index of 1024")

plt.subplot(3,1,2)
for i in range(window):
    plt.plot(frn,northspectrum[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('North Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)

plt.subplot(3,1,3)
for i in range(window):
    plt.plot(frv,verticalspectrum[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('Vertical Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)
#
#
# Smoothing spectrum all window
plt.figure(5,figsize=(10,6))
plt.subplot(3,1,1)
for i in range(window):
    plt.semilogx(frsmooth,e_smooth[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('East Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)

plt.subplot(3,1,2)
for i in range(window):
    plt.semilogx(frsmooth,n_smooth[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('North Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)

plt.subplot(3,1,3)
for i in range(window):
    plt.semilogx(frsmooth,z_smooth[i],linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.title('Vertical Spectrum')
    plt.xlabel('Freq (Hz)')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.35)

# H/V Curve All Window
plt.figure(6,figsize=(5,5))
for i in range(window):
    plt.semilogx(frhv,re_rathv[i],linewidth=0.5)
plt.semilogx(frhv,hvsr,color='black',linewidth=1.0)
plt.semilogx(frhv,minstdhv,"--",frhv,maxstdhv,"--",color='black',linewidth=1.0)
plt.semilogx([F0,F0],[min(hvsr),max(hvsr)*2],color='red',linewidth=1.0)
plt.semilogx([f0min,f0min],[min(hvsr),max(hvsr)*2],color='blue',linewidth=1.0)
plt.semilogx([f0max,f0max],[min(hvsr),max(hvsr)*2],color='blue',linewidth=1.0)
plt.ylabel('H/V')
plt.title('HVSR')
plt.xlabel('Frequency (Hz)')
plt.grid(True, which="both")

# Show plot
# plt.legend()
plt.show()
