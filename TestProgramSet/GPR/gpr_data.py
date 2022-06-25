# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 6/9/2022 8:48 AM
@file: gpr_data.py
"""

import os
import tkinter as tk
from tkinter import filedialog
from glob import glob
import h5py
import numpy as np
from osgeo import osr, gdal
import matplotlib.pyplot as plt

#%% read the GPR data stored in the h5 file
# select the directory of the data
print('\033[0;31mPlease select the folder of GPR data: \033[0m')
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory()
print('\033[0;34mThe path of the GPR data is: %s.\033[0m' % folder_path)
print('------------------------------')
# initialize the global variables
gpr_path = glob(os.path.join(folder_path, '*.h5'))
# read the GPR data using h5py
f = h5py.File(gpr_path[0], 'r')
print(list(f.keys()))
gpr_data_04 = f['swath04']['data']
gpr_coord_04 = f['swath04']['coord']
coord = np.delete(gpr_coord_04[:], 0, axis=1)
space = 0.000001141
coord_r_01 = np.array([coord[:, 0] + space, coord[:, 1], coord[:, 2]]).T
coord_r_02 = np.array([coord[:, 0] + 2 * space, coord[:, 1], coord[:, 2]]).T
coord_r_03 = np.array([coord[:, 0] + 3 * space, coord[:, 1], coord[:, 2]]).T
coord_r_04 = np.array([coord[:, 0] + 4 * space, coord[:, 1], coord[:, 2]]).T
coord_r_05 = np.array([coord[:, 0] + 5 * space, coord[:, 1], coord[:, 2]]).T
coord_r_06 = np.array([coord[:, 0] + 6 * space, coord[:, 1], coord[:, 2]]).T
coord_r_07 = np.array([coord[:, 0] + 7 * space, coord[:, 1], coord[:, 2]]).T
coord_l_01 = np.array([coord[:, 0] - space, coord[:, 1], coord[:, 2]]).T
coord_l_02 = np.array([coord[:, 0] - 2 * space, coord[:, 1], coord[:, 2]]).T
coord_l_03 = np.array([coord[:, 0] - 3 * space, coord[:, 1], coord[:, 2]]).T
coord_l_04 = np.array([coord[:, 0] - 4 * space, coord[:, 1], coord[:, 2]]).T
coord_l_05 = np.array([coord[:, 0] - 5 * space, coord[:, 1], coord[:, 2]]).T
coord_l_06 = np.array([coord[:, 0] - 6 * space, coord[:, 1], coord[:, 2]]).T
coord_l_07 = np.array([coord[:, 0] - 7 * space, coord[:, 1], coord[:, 2]]).T


#%% save data to txt file

names = [coord_l_07, coord_l_06, coord_l_05, coord_l_04, coord_l_03, coord_l_02, coord_l_01, coord,
         coord_r_01, coord_r_02, coord_r_03, coord_r_04, coord_r_05, coord_r_06, coord_r_07]

for m in range(len(names)):
    coord_all = []
    for i in range(512):
        coord_all.append([names[m][:, 0], names[m][:, 1], names[m][:, 2] - 0.01 * i])
    coord_all = np.array(coord_all).T
    test_data = gpr_data_04[:, m, :]  # value of the channel
    coord_all_combine = []
    for j in range(512):
        coord_all_combine.append(np.column_stack((coord_all[:, :, j], test_data[:, j])))
    coord_all_combine = np.array(coord_all_combine)
    coord2d = []
    for k in range(512):
        coord2d.append(coord_all_combine[k, :, :])
    coord2txt = []
    for n in range(512):
        coord2txt.extend(coord2d[n])
    filename = 'channel_' + '%02d' % m + '.txt'
    np.savetxt(filename, coord2txt)


# coord_all = []
# for i in range(512):
#     coord_all.append([coord_l_01[:, 0], coord_l_01[:, 1], coord_l_01[:, 2] - 0.01 * i])
# coord_all = np.array(coord_all).T
#
# test_data = gpr_data_02[:, 0, :]  # value of the channel
# coord_all_combine = []
# for i in range(512):
#     coord_all_combine.append(np.column_stack((coord_all[:, :, i], test_data[:, i])))
# coord_all_combine = np.array(coord_all_combine)
#
# coord2d = []
# for i in range(512):
#     coord2d.append(coord_all_combine[i, :, :])
#
# coord2txt = []
# for i in range(512):
#     coord2txt.extend(coord2d[i])
#
# np.savetxt('channel_01.txt', coord2txt)

#%%

import os
import os.path

filedir = 'D:/ProjectMaterials/3dModel/Data/GPR/data/swath04'
filenames = os.listdir(filedir)
f = open('D:/ProjectMaterials/3dModel/Data/GPR/data/swath04/swath04.txt', 'w')

for filename in filenames:
    filepath = filedir + '/' + filename
    for line in open(filepath):
        f.writelines(line)
    f.write('\n')
f.close()

#%% read the GPR data stored in tif files

dataset = gdal.Open("D:/ProjectMaterials/3dModel/Data/GPR/fangcaoxiang/swath01.tif", gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()
width = dataset.RasterXSize
height = dataset.RasterYSize
gt = dataset.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width * gt[4] + height * gt[5]
maxx = gt[0] + width * gt[1] + height * gt[2]
maxy = gt[3]
# create the new coordinate system
wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs .ImportFromWkt(wgs84_wkt)
# create a transform object to convert between coordinate systems
# transform = osr.CoordinateTransformation(old_cs,new_cs)
#
#
# latlong = transform.TransformPoint(minx,miny)

#%% view the point cloud data with laspy

import laspy

las = laspy.read("D:/ProjectMaterials/3dModel/Data/GPR/data/swath01/swath01.las")
print(las.header.point_format)
