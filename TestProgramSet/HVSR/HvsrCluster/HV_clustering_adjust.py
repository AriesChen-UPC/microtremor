from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster.vq import kmeans, vq
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)

# load parameters
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
root.destroy()

p = Path(Folderpath)
pdir = [x for x in p.iterdir() if x.is_dir()]

fs = []
for idir in pdir:
    fs.append(list(idir.glob('*.hv')))

data = list()
freq = list()
for i in range(len(pdir)):
    data0 = list()
    freq0 = list()
    for fi in fs[i]:
        data0.append(pd.read_table(filepath_or_buffer=fi, sep='\t', skiprows=9,
                                   names=['Freq', 'Aver', 'min_data', 'max_data']).Aver)
        freq0.append(pd.read_table(filepath_or_buffer=fi, sep='\t', skiprows=9,
                                   names=['Freq', 'Aver', 'min_data', 'max_data']).Freq)
    data.append(data0)
    freq.append(freq0[0])

freq_std = np.geomspace(0.1, 100.1, 200)  # todo: range of frequency

min_freq = float(input('Min_freq:\n'))
max_freq = float(input('Max_freq:\n'))

sel = (freq_std >= min_freq) & (freq_std <= max_freq)
freq_sel = freq_std[sel]

mean_data = list()
mean_data1 = np.zeros(shape=(len(pdir), len(freq_std)))
mean_data_sel = np.zeros(shape=(len(pdir), len(freq_sel)))
for i in range(len(pdir)):
    mean_data.append(np.mean(data[i], 0))
    mean_data1[i] = np.interp(freq_std, freq[i], mean_data[i])
    mean_data_sel[i] = mean_data1[i][sel]

n_clusters = int(input('Please input the number of clusters:\n'))
centroid, _ = kmeans(mean_data_sel, n_clusters)
result1, _ = vq(mean_data_sel, centroid)
result1 = result1.tolist()
print(result1)


mean_data1 = pd.DataFrame(mean_data1).T
hvname = [f'hv{n}' for n in range(len(fs))]
mean_data1.columns = hvname
freq_std = pd.DataFrame(freq_std)
freq_std.columns = ['freq']


names = []
for i in range(len(pdir)):
    names.append(Path(pdir[i]).stem)

numRow = 2
numCol = int(np.ceil(n_clusters / 2))
fig = make_subplots(rows=numRow, cols=numCol, print_grid=True)


for z in range(n_clusters):
    count = 0
    for k in range(len(fs)):
        if result1[k] == z:
            count = count + 1
            if z < numCol:
               fig.add_trace(go.Scatter(x=freq_std['freq'], y=mean_data1[hvname[k]], name=names[k]),
                             row=1, col=z+1)
            else:
               fig.add_trace(go.Scatter(x=freq_std['freq'], y=mean_data1[hvname[k]], name=names[k]),
                             row=2, col=z+1-numCol)
# update the plot frame
fig.update_xaxes(type="log")
fig.update_xaxes(range=[0, 2])
fig.update_yaxes(range=[0, 10], tick0=0.00, dtick=2)
fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_yaxes(title_text="Amplitude")
fig.update_layout(title='HV cluster', showlegend=False)

for z in range(n_clusters):
    print('Group {:.0f} '.format(z + 1))
    for i in range(len(fs)):
        if result1[i] == z:
            print(names[i])
# save the .html file
htmlFileName = Folderpath.split("/")[-1] + '_HV_Cluster' + '.html'
plotly.offline.plot(fig, filename=Folderpath + '/' + htmlFileName)
