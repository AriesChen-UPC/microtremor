"""
Created on May 31,2021
Update on May 31,2021
@author: LijiongChen
This code is designead to red the spac data form page format and compare the different period spac data.
read all rings data form page and specially plot 3 rings: ring1, ring2, ring3.
"""


from termcolor import colored
import numpy as np
from glob import glob
import gzip
from xml.dom import minidom
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)


# Load SPAC
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print(colored('Folderpath:', 'green', attrs=['bold']))
print(Folderpath)
root.destroy()


def read_page(f, plot=False, write=False):
    p = Path(f)
    name = p.stem
    path = f.strip(name + '.page')

    with open(f, 'rb') as temp:
        x = temp.read()
        y = gzip.decompress(x).decode('utf-8', 'ignore').strip('\x00').replace('\x00', '')
        z = y[y.find('<SciFigs>'):]

    dom = minidom.parseString(z)
    rootdata = dom.documentElement
    ElementList = rootdata.getElementsByTagName("points")

    data = []
    for i in range(ElementList.length):
        temp0 = ElementList[i].firstChild.data
        if temp0.__len__() < 10000:  # todo: need to verify the exact num
            continue
        else:
            RawResult = ElementList[i].firstChild.data
            TESTDATA = StringIO(RawResult.strip('\n'))
            data.append(pd.read_table(TESTDATA, sep='\\s+', header=None, names=['freq', 'spac', 'stderr']))

    if plot:
        plt.figure()
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.errorbar(x=data[i].freq, y=data[i].spac, yerr=data[i].stderr)
        plt.show()

    if write:
        for i in range(3):
            data[i].to_csv(path + name + 'r' + str(i + 1) + '.csv', index=False)

    return data, name


# get page filepath and filename
fs = glob(Folderpath + '/*.page')
freq = np.logspace(np.log10(0.5), np.log10(100.1), 306)
names = []
for i in range(len(fs)):
    names.append(Path(fs[i]).stem)
print(colored('SpacPageName:', 'green', attrs=['bold']))
# read page file and combine the data to tuple
dataPage = []
for i in range(len(fs)):
    print(fs[i].split("\\")[-1].split('.')[0])
    dataPage += read_page(fs[i])
# plot the figure for each page file
for i in range(len(fs)):
    # get the spac ring ifo
    with open(str(fs[i]), 'rb') as temp:
        x = temp.read()
        y = gzip.decompress(x).decode('utf-8', 'ignore').strip('\x00').replace('\x00', '')
        z = y[y.find('<SciFigs>'):]
    dom = minidom.parseString(z)
    rootdata = dom.documentElement
    ElementList = rootdata.getElementsByTagName("text")
    ring1 = ElementList[0].firstChild.data
    ring2 = ElementList[1].firstChild.data
    ring3 = ElementList[2].firstChild.data

# figure
for i in range(int(len(dataPage)/4)):

    fig = make_subplots(rows=1, cols=3, subplot_titles=(ring1, ring2, ring3))
    # ring1
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][0])['freq'], y=pd.DataFrame(dataPage[4*i][0])['spac'] + pd.DataFrame(dataPage[4*i][0])['stderr'], mode='lines',
                            line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][0])['freq'], y=pd.DataFrame(dataPage[4*i][0])['spac'] - pd.DataFrame(dataPage[4*i][0])['stderr'], mode='lines',
                            line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][0])['freq'], y=pd.DataFrame(dataPage[4*i+2][0])['spac'] + pd.DataFrame(dataPage[4*i+2][0])['stderr'],mode='lines',
                             line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][0])['freq'],y=pd.DataFrame(dataPage[4*i+2][0])['spac'] - pd.DataFrame(dataPage[4*i+2][0])['stderr'],mode='lines',
                             line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][0])['freq'], y=pd.DataFrame(dataPage[4*i][0])['spac'], mode='lines',
                             name='Ring1_Period_1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][0])['freq'], y=pd.DataFrame(dataPage[4*i+2][0])['spac'], mode='lines',
                             name='Ring1_Period_2'), row=1, col=1)

    # ring2
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][1])['freq'], y=pd.DataFrame(dataPage[4*i][1])['spac'] + pd.DataFrame(dataPage[4*i][1])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][1])['freq'], y=pd.DataFrame(dataPage[4*i][1])['spac'] - pd.DataFrame(dataPage[4*i][1])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][1])['freq'], y=pd.DataFrame(dataPage[4*i+2][1])['spac'] + pd.DataFrame(dataPage[4*i+2][1])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][1])['freq'], y=pd.DataFrame(dataPage[4*i+2][1])['spac'] - pd.DataFrame(dataPage[4*i+2][1])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][1])['freq'], y=pd.DataFrame(dataPage[4*i][1])['spac'], mode='lines',
                         name='Ring2_Period_1'), row=1, col=2)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][1])['freq'], y=pd.DataFrame(dataPage[4*i+2][1])['spac'], mode='lines',
                         name='Ring2_Period_2'), row=1, col=2)
    # ring3
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][2])['freq'], y=pd.DataFrame(dataPage[4*i][2])['spac'] + pd.DataFrame(dataPage[4*i][2])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][2])['freq'], y=pd.DataFrame(dataPage[4*i][2])['spac'] - pd.DataFrame(dataPage[4*i][2])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][2])['freq'], y=pd.DataFrame(dataPage[4*i+2][2])['spac'] + pd.DataFrame(dataPage[4*i+2][2])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][2])['freq'], y=pd.DataFrame(dataPage[4*i+2][2])['spac'] - pd.DataFrame(dataPage[4*i+2][2])['stderr'], mode='lines',
                         line=dict(color='rgba(255,255,255,0)'), fill='tonexty', showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i][2])['freq'], y=pd.DataFrame(dataPage[4*i][2])['spac'], mode='lines',
                         name='Ring3_Period_1'), row=1, col=3)
    fig.add_trace(go.Scatter(x=pd.DataFrame(dataPage[4*i+2][2])['freq'], y=pd.DataFrame(dataPage[4*i+2][2])['spac'], mode='lines',
                         name='Ring3_Period_2'), row=1, col=3)
    # design the frame properties todo: Optional frame properties
    # define a log frame properties todo: a specify frame properties
    fig.update_xaxes(type="log")
    fig.update_xaxes(range=[0, 2])
    # fig.update_xaxes(range=[0, 100], tick0=0.00, dtick=20)
    fig.update_yaxes(range=[-0.6, 1], tick0=0.00, dtick=0.2)
    fig.update_layout(title=dataPage[4*i+1][0:9])
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Autocorr ratio")
    # plotly.offline.plot(fig, filename=Folderpath + '/' + 'SPAC.html')
    htmlFileName = dataPage[4*i+1][0:9] + '_SPAC' + '.html'
    plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
