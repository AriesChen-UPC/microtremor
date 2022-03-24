import glob
import os
from tkinter import filedialog
import tkinter as tk
import plotly
from DataProcess.project.hv.hv_verification.geopsy_hvsrcheck import geopsy_hvsrcheck
plotly.offline.init_notebook_mode(connected=True)

root = tk.Tk()
root.withdraw()
path = filedialog.askdirectory()

hv_file = glob.glob(os.path.join(path, "*.hv"))
print(hv_file)
log_file = glob.glob(os.path.join(path, "*.log"))
print(log_file)

for i in range(len(hv_file)):
    geopsy_hvsrcheck(hv_file[i], log_file[i])
