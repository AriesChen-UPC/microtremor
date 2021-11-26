import numpy as np
from get_3ring_data import load_data3
import tkinter as tk
from tkinter import filedialog

# setup freq.
freq = np.logspace(np.log10(0.1), np.log10(100), 400)

# load parameters
root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:',Folderpath)
root.destroy()

# 读取数据
path = Folderpath +'/**'
d3, e3, n = load_data3(path, freq, output=False)

n_data = d3[0].shape[0]

print('\nChecked rings in freq, 2-20Hz\n=============================')
sel = (freq <=20) & (freq>2)

sub = np.zeros([n_data,2])
sub[:, 0] = (d3[0]*sel-d3[1]*sel).sum(1)
sub[:, 1] = (d3[1]*sel-d3[2]*sel).sum(1)


chk = (sub>=0).all(1)
if all(chk):
    print('All data are good!')
else:
    ind = np.where(chk == False)[0].tolist()
    print('Please check the following data:\n{}'.format(n[ind]))

input("Press <enter> to Exit")

