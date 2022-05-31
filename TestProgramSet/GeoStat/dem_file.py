# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/19/2022 1:59 PM
@file: dem_file.py: This script is used to read the DEM file downloaded from NASA in hgt format.
"""

import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt

try:
    from osgeo import gdal
except ImportError:
    import gdal

print('\033[0;31mPlease select the DEM data(.hgt).\033[0m')
root = tkinter.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print('\033[0;32mThe selected file is: %s.\033[0m' % file_path)

raster = gdal.Open(file_path)
raster_coordinates = raster.GetGeoTransform()
raster_array = raster.ReadAsArray()
plt.imshow(raster_array, cmap='gray')
plt.show()
