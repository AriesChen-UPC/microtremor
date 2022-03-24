# File name     : geopsy_hvsrcheck.py
# Info          : Program to check reliable and clear peak of H/V curve from Geopsy file (.hv)
# Update        : 24th Dec 2020
# Written by    : LijiongChen
# Source        : GUIDELINES FOR THE IMPLEMENTATION OF THE H/V SPECTRAL RATIO TECHNIQUE ON AMBIENT VIBRATIONS
#                 MEASUREMENTS, PROCESSING AND INTERPRETATION, SESAME European research microtremor
#                 WP12 – Deliverable D23.12
#
# USAGE         : Using hvsrcheck.py as modules file
#                 example: hvsrcheck(input)
#                 input is dictionary type.
#                 "You just need to add the .hv and .log files in filename and logname variable that you have"
# TESTED        : Python >= 3.7
# ------------------------------------------------------------------------------------------------------------------------------------------
import codecs
from numpy import *
import pandas as pd
from hvcheck import hvsrcheck
from pandas import Series, DataFrame

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plotly.offline.init_notebook_mode(connected=True)


def geopsy_hvsrcheck(filename, logname):
    filename = filename  # todo:auto read
    logname = logname  # todo:auto read
    data = open(filename, "r")
    logfile = open(logname, "r")
    # Read .hv file
    index = 0
    frequency = []
    amplification = []
    minamplification = []
    maxamplification = []
    for line in data:
        # Take Dominant Frequency (f0)
        if index == 4:
            fr0 = line.strip().split("\t")
            domfreq = float(fr0[1])
            mindomfreq = float(fr0[2])
            maxdomfreq = float(fr0[3])
        # Take Amplification
        if index == 5:
            hv = line.strip().split("\t")
            hv = float(hv[1])

        # Take the data
        if index >= 9:
            field = line.strip().split("\t")
            freq = float(field[0])
            amp = float(field[1])
            minamp = float(field[2])
            maxamp = float(field[3])
            frequency.append(freq)
            amplification.append(amp)
            minamplification.append(minamp)
            maxamplification.append(maxamp)
        index = index + 1

    frequency = array(frequency)
    amplification = array(amplification)
    minamplification = array(minamplification)
    maxamplification = array(maxamplification)
    KG = ((hv**2)/domfreq)
    idmaxamp = argmax(amplification)
    stddivA = divide(maxamplification, amplification)
    stdA = stddivA[idmaxamp]
    stdf0 = domfreq-mindomfreq
    stdhv = subtract(amplification, minamplification)
    minstdA0 = minamplification[argmax(minamplification)]
    maxstdA0 = maxamplification[argmax(maxamplification)]
    collect = [frequency, amplification, minamplification, maxamplification]
    store_data = pd.DataFrame(collect, index=['Frequency', 'Average', 'Min', 'Max']).T
    print("-----------------------------------------------------------------------")
    print("OUTPUT INFORMATION OF H/V")
    print("-----------------------------------------------------------------------")
    print(store_data)
    print("A0\t\t:", hv)
    print("F0\t\t:", domfreq, "Hz")
    print("KG\t\t:", KG)
    print("MIN F0\t\t:", mindomfreq, "Hz")
    print("MAXF0\t\t:", maxdomfreq, "Hz")
    data.close()

    # Read log file
    recordFile = logname
    FoundFlag = False
    fileObj = codecs.open(recordFile, 'r+', 'utf-8')
    lineTemp = fileObj.readlines()
    count_1 = 1
    count_2 = 1
    server_label_1 = "### Time Windows ###"
    server_label_2 = "# Start time"
    for line in lineTemp:
        # 等于-1表示匹配不到，非-1表示匹配到的索引
        if line.strip().find(server_label_1) == -1:
            FoundFlag = False
            count_1 += 1
            # print("the line is: " + line, end='')
        else:
            break
    for line in lineTemp:
        # 等于-1表示匹配不到，非-1表示匹配到的索引
        if line.strip().find(server_label_2) == -1:
            FoundFlag = False
            count_2 += 1
            # print("the line is: " + line, end='')
        else:
            break
    fileObj.close()

    index = 0
    for line in logfile:
        # Find window Number
        if index == count_1:  # todo:Find key characters
            win = line.strip().split("\t")
            window = int(win[0][-2:])

        # Find window length
        if index == count_2:  # todo:Find key characters
            winl = line.strip().split("\t")
            winlength = float(winl[2])
        index = index + 1

    print("WINDOW\t\t:", window)
    print("WINDOW LENGTH\t:", winlength, "second")
    logfile.close()

    # HVSR CHECK
    # Store output information to dictionary
    input = {
                'filename': filename,
                'maxfreq': max(frequency),
                'winlength': winlength,
                'window': window,
                'frhv': frequency,
                'hvsr': amplification,
                'A0': hv,
                'F0': domfreq,
                'KG': KG,
                'stdA': stdA,
                'stdf0': stdf0,
                'stdhv': stdhv,
                'f0min': mindomfreq,
                'f0max': maxdomfreq,
                'minstdhv': minamplification,
                'maxstdhv': maxamplification,
                'minstdA0': minstdA0,
                'maxstdA0': maxstdA0,
                'lengthhv': len(amplification),
                }

    hvsrcheck(input)

    hv_data = {'freq': Series(frequency), 'amp': Series(amplification), 'min_amp': Series(minamplification),
               'max_amp': Series(maxamplification)}
    df_hv_data = DataFrame(hv_data)

    fig = make_subplots(rows=1, cols=1, x_title='Frequency (Hz)', y_title='Amplitude')
    fig.add_trace(go.Scatter(x=df_hv_data['freq'], y=df_hv_data['amp'], name='amplification',
                             legendgroup='amplification'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_hv_data['freq'], y=df_hv_data['min_amp'], name='minamplification',
                             legendgroup='minamplification'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_hv_data['freq'], y=df_hv_data['max_amp'], name='minamplification',
                             legendgroup='maxamplification'), row=1, col=1)
    # fig.update_xaxes(range=[0, 2], tick0=0.00, dtick=0.1, gridcolor='LightPink')
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, 10], tick0=0.00, dtick=1.0, gridcolor='LightPink')
    fig.update_layout(title=filename)
    fig.update_layout(height=800, width=1000)
    plotly.offline.plot(fig)
