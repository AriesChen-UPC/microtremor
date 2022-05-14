# -*- coding:Latin-1 -*-
"""########################################DESCRIPTION########################################"""
"""c_plot_hvsr
This script uses the results from c_plot_hvsr.py script.
It displays the results computed by Geopsy software (www.geopsy.org, Marc WATHELET, ISTerre).

It displays the F0 peak for each measurement point on a map and saves it into .pdf files.
It requires an internet connexion to download data from Arcgis service (see Basemap and Argis websites for more information).

Reference: please cite as: Bottelin, P., (2015) An open-source Python tool for Geopsy HVSR post-processing.
Contact: Pierre BOTTELIN
Post-doctoral teacher and researcher
Applied geophysics, geotechnical engineering and environment
IRAP / OMP Toulouse
pierre.bottelin@irap.omp.eu
"""
"""########################################PARAMETERS########################################"""
"""GENERAL"""
MAP_DATAPATH = "../MAP/" #path to MAP data folders
MAP_FIGURES_PATH = "../MAP/" #path to MAP figures folders
SAVE_IMG = True #save resulting map as pdf in MAP_FIGURES_PATH
#
res = 'l' #resolution of the map (vectors)
DISP_IMAGE = True #display online collected arcgis background image
IMAGE_TYPE = 'World_Imagery' #type of background image 'World_Imagery' or 'Shaded_Relief' or ... see Arcgis service website for more details
DISP_F0_RELIABLE = True #display reliable F0 points
DISP_F0_NOT_RELIABLE = True #display not reliable F0 points
DISP_F0_DUMP = True #display dump F0 points
#
figsize_width = 16 #cm #width of figure
figsize_height = 9 #cm #height of figure
#
ftzse_axes = 8 #pt #fontsize for axes labels
ftzse_nos_fig = 14 #pt #fontsize for figure numbers
linewidth_cadre = 0.5 #pt #linewidth for the frame
linewidth_parmer = 0.5 #pt #fontsize for parralels and meridians
markersize_HV = 40 #pt# #marker size of HV points
linewidth_markers_HV = 0.5 #pt #line width size of HV points

#MAP
min_lat_MAP = [] #minimum latitude of the map. If empty, determined from the points coordinates
max_lat_MAP = [] #maximum latitude of the map. If empty, determined from the points coordinates
min_lon_MAP = [] #minimum longitude of the map. If empty, determined from the points coordinates
max_lon_MAP = [] #maximum longitude of the map. If empty, determined from the points coordinates
MARGIN = 0.02 #° #margin around data coordinates (in degrees) if no coordinates specified

#VS
VS = 590 #m/s

"""Exclude points from list"""
EXCL_POINTS = [] #list of point names to exclude from map plot. Example: EXCL_POINTS = ["POINT24"]
    
"""FMIN FMAX""" #frequency range [Fmin Fmax] for map plot. If F0 lies out of this range, the point is not mapped.
FMIN = 0.5 #Hz
FMAX = 25 #Hz

"""########################################MAIN SCRIPT########################################"""

"""MODULE IMPORTS"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import numpy as np
from pykml import parser
import csv

"""SCRIPT"""

"""Load H/V data"""

"""F0_RELIABLE"""
data = []
with open(MAP_DATAPATH + "F0_RELIABLE.txt", 'r') as fh: #reads text file
    reader = csv.reader(fh, delimiter='\t')
    next(reader, None)  # skip the header line
    for row in reader:
         data.append(row)
NAME_F0_RELIABLE = []
F0_RELIABLE = []
LON_F0_RELIABLE = []
LAT_F0_RELIABLE = []
for point in data:
    NAME_F0_RELIABLE.append(point[0])
    F0_RELIABLE.append(float(point[1]))
    LON_F0_RELIABLE.append(float(point[2]))
    LAT_F0_RELIABLE.append(float(point[3]))

"""F0_NOT_RELIABLE"""
data = []
with open(MAP_DATAPATH + "F0_NOT_RELIABLE.txt", 'r') as fh: #reads text file
    reader = csv.reader(fh, delimiter='\t')
    next(reader, None)  # skip the header line
    for row in reader:
         data.append(row)
NAME_F0_NOT_RELIABLE = []
F0_NOT_RELIABLE = []
LON_F0_NOT_RELIABLE = []
LAT_F0_NOT_RELIABLE = []
for point in data:
    NAME_F0_NOT_RELIABLE.append(point[0])
    F0_NOT_RELIABLE.append(float(point[1]))
    LON_F0_NOT_RELIABLE.append(float(point[2]))
    LAT_F0_NOT_RELIABLE.append(float(point[3]))
    
"""F0_DUMP"""
data = []
with open(MAP_DATAPATH + "F0_DUMP.txt", 'r') as fh: #reads text file
    reader = csv.reader(fh, delimiter='\t')
    next(reader, None)  # skip the header line
    for row in reader:
         data.append(row)
NAME_F0_DUMP = []
F0_DUMP = []
LON_F0_DUMP = []
LAT_F0_DUMP = []
for point in data:
    NAME_F0_DUMP.append(point[0])
    F0_DUMP.append(float(point[1]))
    LON_F0_DUMP.append(float(point[2]))
    LAT_F0_DUMP.append(float(point[3]))

"""EXCLUDE BAD POINTS FROM LIST"""
"""F0_RELIABLE"""
idx_NAME_F0_RELIABLE = [i for i,x in enumerate(NAME_F0_RELIABLE) if x not in EXCL_POINTS]
NAME_F0_RELIABLE = [x for i,x in enumerate(NAME_F0_RELIABLE) if x not in EXCL_POINTS]
F0_RELIABLE = [x for i,x in enumerate(F0_RELIABLE) if i in idx_NAME_F0_RELIABLE]
LON_F0_RELIABLE = [x for i,x in enumerate(LON_F0_RELIABLE) if i in idx_NAME_F0_RELIABLE]
LAT_F0_RELIABLE = [x for i,x in enumerate(LAT_F0_RELIABLE) if i in idx_NAME_F0_RELIABLE]
"""F0_NOT_RELIABLE"""
idx_NAME_F0_NOT_RELIABLE = [i for i,x in enumerate(NAME_F0_NOT_RELIABLE) if x not in EXCL_POINTS]
NAME_F0_NOT_RELIABLE = [x for i,x in enumerate(NAME_F0_NOT_RELIABLE) if x not in EXCL_POINTS]
F0_NOT_RELIABLE = [x for i,x in enumerate(F0_NOT_RELIABLE) if i in idx_NAME_F0_NOT_RELIABLE]
LON_F0_NOT_RELIABLE = [x for i,x in enumerate(LON_F0_NOT_RELIABLE) if i in idx_NAME_F0_NOT_RELIABLE]
LAT_F0_NOT_RELIABLE = [x for i,x in enumerate(LAT_F0_NOT_RELIABLE) if i in idx_NAME_F0_NOT_RELIABLE]
"""F0_DUMP"""
idx_NAME_F0_DUMP = [i for i,x in enumerate(NAME_F0_DUMP) if x not in EXCL_POINTS]
NAME_F0_DUMP = [x for i,x in enumerate(NAME_F0_DUMP) if x not in EXCL_POINTS]
F0_DUMP = [x for i,x in enumerate(F0_DUMP) if i in idx_NAME_F0_DUMP]
LON_F0_DUMP = [x for i,x in enumerate(LON_F0_DUMP) if i in idx_NAME_F0_DUMP]
LAT_F0_DUMP = [x for i,x in enumerate(LAT_F0_DUMP) if i in idx_NAME_F0_DUMP]

"""KEEP POINTS LYING BETWEEN FMIN AND FMAX"""
"""F0_RELIABLE"""
idx_F0_RELIABLE = [i for i,x in enumerate(F0_RELIABLE) if (x >= FMIN) & (x <= FMAX)]
F0_RELIABLE = [x for i,x in enumerate(F0_RELIABLE) if (x >= FMIN) & (x <= FMAX)]
NAME_F0_RELIABLE = [x for i,x in enumerate(NAME_F0_RELIABLE) if i in idx_F0_RELIABLE]
LON_F0_RELIABLE = [x for i,x in enumerate(LON_F0_RELIABLE) if i in idx_F0_RELIABLE]
LAT_F0_RELIABLE = [x for i,x in enumerate(LAT_F0_RELIABLE) if i in idx_F0_RELIABLE]
"""F0_NOT_RELIABLE"""
idx_F0_NOT_RELIABLE = [i for i,x in enumerate(F0_NOT_RELIABLE) if (x >= FMIN) & (x <= FMAX)]
F0_NOT_RELIABLE = [x for i,x in enumerate(F0_NOT_RELIABLE) if (x >= FMIN) & (x <= FMAX)]
NAME_F0_NOT_RELIABLE = [x for i,x in enumerate(NAME_F0_NOT_RELIABLE) if i in idx_F0_NOT_RELIABLE]
LON_F0_NOT_RELIABLE = [x for i,x in enumerate(LON_F0_NOT_RELIABLE) if i in idx_F0_NOT_RELIABLE]
LAT_F0_NOT_RELIABLE = [x for i,x in enumerate(LAT_F0_NOT_RELIABLE) if i in idx_F0_NOT_RELIABLE]
"""F0_DUMP"""
idx_F0_DUMP = [i for i,x in enumerate(F0_DUMP) if (x >= FMIN) & (x <= FMAX)]
F0_DUMP = [x for i,x in enumerate(F0_DUMP) if (x >= FMIN) & (x <= FMAX)]
NAME_F0_DUMP = [x for i,x in enumerate(NAME_F0_DUMP) if i in idx_F0_DUMP]
LON_F0_DUMP = [x for i,x in enumerate(LON_F0_DUMP) if i in idx_F0_DUMP]
LAT_F0_DUMP = [x for i,x in enumerate(LAT_F0_DUMP) if i in idx_F0_DUMP]

"""COMPUTES THE SEDIMENT HEIGH !!! USES THE F0=Vs/(4*H) FORMULA: use in 1D geometry only !!!!"""
H_RELIABLE = [VS/(4*f) for f in F0_RELIABLE] #H sediments (m)
H_NOT_RELIABLE = [VS/(4*f) for f in F0_NOT_RELIABLE] #H sediments (m)
H_DUMP = [VS/(4*f) for f in F0_DUMP] #H sediments (m)

"""MIN AND MAX VALUES"""
vmin_FO = np.nanmin(F0_RELIABLE+F0_NOT_RELIABLE+F0_DUMP)
vmax_FO = np.nanmax(F0_RELIABLE+F0_NOT_RELIABLE+F0_DUMP)
vmin_H = np.nanmin(H_RELIABLE+H_NOT_RELIABLE+H_DUMP)
vmax_H = np.nanmax(H_RELIABLE+H_NOT_RELIABLE+H_DUMP)

"""MAP EXTENT""" #coordinates of map corners
if not min_lon_MAP:
    min_lon_MAP = min(LON_F0_DUMP + LON_F0_NOT_RELIABLE + LON_F0_RELIABLE)-MARGIN
if not max_lon_MAP:
    max_lon_MAP = max(LON_F0_DUMP + LON_F0_NOT_RELIABLE + LON_F0_RELIABLE)+MARGIN
if not min_lat_MAP:
    min_lat_MAP = min(LAT_F0_DUMP + LAT_F0_NOT_RELIABLE + LAT_F0_RELIABLE)-MARGIN
if not max_lat_MAP:
    max_lat_MAP = max(LAT_F0_DUMP + LAT_F0_NOT_RELIABLE + LAT_F0_RELIABLE)+MARGIN
lon_MAP=(min_lon_MAP+max_lon_MAP)/2
lat_MAP=(min_lat_MAP+max_lat_MAP)/2
inc_lon_MAP = (max_lon_MAP-min_lon_MAP)/3.
inc_lat_MAP = (max_lat_MAP-min_lat_MAP)/4.

"""########################################FIGURES########################################"""
plt.close("all")
FIG1 = plt.figure(figsize=(figsize_width/2.54, figsize_height/2.54))

"""############## SUBPLOT1 ##############"""
ax1 = FIG1.add_subplot(131)
plt.text(-0.1, 1, "a)", fontsize=ftzse_nos_fig, fontweight='normal', horizontalalignment='center',
    verticalalignment='center', transform = ax1.transAxes)
    
map = Basemap(llcrnrlon=min_lon_MAP, llcrnrlat=min_lat_MAP,
    urcrnrlon=max_lon_MAP, urcrnrlat=max_lat_MAP, epsg=5520, resolution=res, ax=ax1) #see python basemap module for details and options
if DISP_IMAGE:
    map.arcgisimage(service=IMAGE_TYPE, dpi=300, verbose= True, zorder=0)
map.drawparallels(np.arange(round(min_lat_MAP+inc_lat_MAP/2,2), round(max_lat_MAP,2), inc_lat_MAP), linewidth=linewidth_parmer,
    labels=[True, False, False, False], fmt="%.2f", fontsize=ftzse_axes, rotation= 90, dashes=[2, 2], zorder=1)
map.drawmeridians(np.arange(round(min_lon_MAP,2), round(max_lon_MAP,2), inc_lon_MAP), linewidth=linewidth_parmer,
    labels=[False, False, False, True], fmt="%.2f", fontsize=ftzse_axes, dashes=[2, 2], zorder=1)
map.drawmapboundary(linewidth=linewidth_cadre, fill_color='None')

if DISP_F0_DUMP:   
    MAP_F0_DUMP = map.scatter(LON_F0_DUMP, LAT_F0_DUMP, c=F0_DUMP, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='white') #F0_DUMP points are circled in white
if DISP_F0_NOT_RELIABLE:        
    MAP_F0_NOT_RELIABLE = map.scatter(LON_F0_NOT_RELIABLE, LAT_F0_NOT_RELIABLE,  c=F0_NOT_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='grey') #F0_NOT_RELIABLE points are circled in grey
if DISP_F0_RELIABLE:
    MAP_F0_RELIABLE = map.scatter(LON_F0_RELIABLE, LAT_F0_RELIABLE, c=F0_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='black') #F0_RELIABLE points are circled in black

m=cm.ScalarMappable(cmap=cm.jet, norm=mpl.colors.LogNorm())
m.set_array(F0_RELIABLE + F0_NOT_RELIABLE + F0_DUMP)
cbar = map.colorbar(m, location='right', ticks=[1,2,3,4,5,6,7,8,9,10]) #colorbar ticks
cbar.set_label('$f_{0}$ [Hz]', fontsize=ftzse_axes, labelpad=2)
cbar.ax.tick_params(labelsize=ftzse_axes)
cbar.ax.set_yticklabels([1,2,3,4,5,6,7,8,9,10], rotation=90) #colorbar ticks labels
cbar.ax.get_children()[2].set_linewidth(linewidth_cadre)

"""############## SUBPLOT2 ##############"""
ax2 = FIG1.add_subplot(132)
plt.text(-0.1, 1, "b)", fontsize=ftzse_nos_fig, fontweight='normal', horizontalalignment='center',
    verticalalignment='center', transform = ax2.transAxes)

map = Basemap(llcrnrlon=min_lon_MAP, llcrnrlat=min_lat_MAP,
    urcrnrlon=max_lon_MAP, urcrnrlat=max_lat_MAP, epsg=5520, resolution=res, ax=ax2) #see python basemap module for details and options
if DISP_IMAGE:
    map.arcgisimage(service=IMAGE_TYPE, dpi=300, verbose= True, zorder=0)
map.drawparallels(np.arange(round(min_lat_MAP+inc_lat_MAP/2,2), round(max_lat_MAP,2), inc_lat_MAP), linewidth=linewidth_parmer,
    labels=[True, False, False, False], fmt="%.2f", fontsize=ftzse_axes, rotation= 90, dashes=[2, 2], zorder=1)
map.drawmeridians(np.arange(round(min_lon_MAP,2), round(max_lon_MAP,2), inc_lon_MAP), linewidth=linewidth_parmer,
    labels=[False, False, False, True], fmt="%.2f", fontsize=ftzse_axes, dashes=[2, 2], zorder=1)
map.drawmapboundary(linewidth=linewidth_cadre, fill_color='None')

if DISP_F0_DUMP:   
    MAP_F0_DUMP = map.scatter(LON_F0_DUMP, LAT_F0_DUMP, c=F0_DUMP, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='white') #F0_DUMP points are circled in white
if DISP_F0_NOT_RELIABLE:        
    MAP_F0_NOT_RELIABLE = map.scatter(LON_F0_NOT_RELIABLE, LAT_F0_NOT_RELIABLE,  c=F0_NOT_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='grey') #F0_NOT_RELIABLE points are circled in grey
if DISP_F0_RELIABLE:
    MAP_F0_RELIABLE = map.scatter(LON_F0_RELIABLE, LAT_F0_RELIABLE, c=F0_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='black') #F0_RELIABLE points are circled in black

m=cm.ScalarMappable(cmap=cm.jet)
m.set_array(F0_RELIABLE + F0_NOT_RELIABLE + F0_DUMP)
cbar = map.colorbar(m,location='right', ticks=[1,2,3,4,5,6,7,8,9,10]) #colorbar ticks
cbar.set_label('$f_{0}$ [Hz]', fontsize=ftzse_axes, labelpad=2)
cbar.ax.tick_params(labelsize=ftzse_axes)
cbar.ax.set_yticklabels([1,2,3,4,5,6,7,8,9,10], rotation=90) #colorbar ticks labels
cbar.ax.get_children()[2].set_linewidth(linewidth_cadre)

"""############## SUBPLOT3 ##############"""
ax2 = FIG1.add_subplot(133)
plt.text(-0.1, 1, "c)", fontsize=ftzse_nos_fig, fontweight='normal', horizontalalignment='center',
    verticalalignment='center', transform = ax2.transAxes)

map = Basemap(llcrnrlon=min_lon_MAP, llcrnrlat=min_lat_MAP,
    urcrnrlon=max_lon_MAP, urcrnrlat=max_lat_MAP, epsg=5520, resolution=res, ax=ax2) #see python basemap module for details and options
if DISP_IMAGE:
    map.arcgisimage(service=IMAGE_TYPE, dpi=300, verbose= True, zorder=0)
map.drawparallels(np.arange(round(min_lat_MAP+inc_lat_MAP/2,2), round(max_lat_MAP,2), inc_lat_MAP), linewidth=linewidth_parmer,
    labels=[True, False, False, False], fmt="%.2f", fontsize=ftzse_axes, rotation= 90, dashes=[2, 2], zorder=1)
map.drawmeridians(np.arange(round(min_lon_MAP,2), round(max_lon_MAP,2), inc_lon_MAP), linewidth=linewidth_parmer,
    labels=[False, False, False, True], fmt="%.2f", fontsize=ftzse_axes, dashes=[2, 2], zorder=1)
map.drawmapboundary(linewidth=linewidth_cadre, fill_color='None')

if DISP_F0_DUMP:   
    MAP_H_DUMP = map.scatter(LON_F0_DUMP, LAT_F0_DUMP, c=H_DUMP, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='white', vmin=vmin_H, vmax=vmax_H) #H_DUMP points are circled in white
if DISP_F0_NOT_RELIABLE:        
    MAP_H_NOT_RELIABLE = map.scatter(LON_F0_NOT_RELIABLE, LAT_F0_NOT_RELIABLE,  c=H_NOT_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='grey', vmin=vmin_H, vmax=vmax_H) #H_NOT_RELIABLE points are circled in grey
if DISP_F0_RELIABLE:
    MAP_H_RELIABLE = map.scatter(LON_F0_RELIABLE, LAT_F0_RELIABLE, c=H_RELIABLE, s=markersize_HV, marker='o', latlon='True', zorder=2,
        linewidth=linewidth_markers_HV, edgecolor='black', vmin=vmin_H, vmax=vmax_H) #H_RELIABLE points are circled in black

m=cm.ScalarMappable(cmap=cm.jet_r)
m.set_array(H_RELIABLE + H_NOT_RELIABLE + H_DUMP)
cbar = map.colorbar(m,location='right', ticks=[20,40,60,80,100,120,140,160]) #colorbar ticks
cbar.set_label('$H$ [m]', fontsize=ftzse_axes, labelpad=2)
cbar.ax.tick_params(labelsize=ftzse_axes)
cbar.ax.set_yticklabels([20,40,60,80,100,120,140,160], rotation=90) #colorbar ticks labels
cbar.ax.get_children()[2].set_linewidth(linewidth_cadre)

plt.tight_layout()
"""SAVE"""
if SAVE_IMG:
    plt.savefig(MAP_FIGURES_PATH + "MAP.pdf", format='pdf', bbox_inches='tight')
else:
    plt.show()