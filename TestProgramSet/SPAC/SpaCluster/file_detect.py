import os
import tkinter as tk
from tkinter import filedialog
import xlwt

root = tk.Tk()
root.withdraw()
Folderpath = filedialog.askdirectory()
print('Folderpath:', Folderpath)
# for i, j, k in os.walk(Folderpath):
#     print(i, j, k)

pointName = []
for name in os.listdir(Folderpath):
    print(name)
    pointName.append(name)

