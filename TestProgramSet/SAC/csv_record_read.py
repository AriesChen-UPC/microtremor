# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 3/25/2022 11:54 AM
@file: csv_record_read.py
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

record = pd.read_csv(file_path)
location = record['Location']
start_time = record['Starttime']
record_select = pd.concat([location, start_time], axis=1)
record_select.duplicated()
record_select_drop = record_select.drop_duplicates()
record_select_drop.to_csv(os.path.dirname(file_path) + '/' + 'log_select.csv', index=False)

