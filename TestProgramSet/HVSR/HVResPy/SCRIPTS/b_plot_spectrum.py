# -*- coding:Latin-1 -*-
"""########################################DESCRIPTION########################################"""
"""b_plot_spectrum
This script uses the results from a_process_data.py script.
It displays the results computed by Geopsy software (www.geopsy.org, Marc WATHELET, ISTerre).

It displays the HVSR for each measurement point and saves it into .pdf files.
It requires .hv .log .win files for each measurement point (see output from a_process_data).

The main point consists in automatic checking the reliability of the HVSR curve and peak picking,
in the meaning of SESAME guidelines
(see Bard and the SESAME Team (2004) Guidelines for the implementation of the H/V spectral ratio
technique of ambient vibrations, SESAME European research project WP12 – Deliverable D23.12)

Reference: please cite as: Bottelin, P., (2015) An open-source Python tool for Geopsy HVSR post-processing.
Contact: Pierre BOTTELIN
Post-doctoral teacher and researcher
Applied geophysics, geotechnical engineering and environment
IRAP / OMP Toulouse
pierre.bottelin@irap.omp.eu
"""

"""########################################PARAMETERS########################################"""
POINT_LIST = [] #list of points to process. If empty, process all the points.

SPECTRUM_DATAPATH = "../DATA/SPECTRUM_DATA/" #path to spectrum data folders
SPECTRUM_FIG_PATH = "../FIGURES/SPECTRUM_FIGURES/" #path to spectrum figure folders

DISP_ALL_CURVES = True
#True: displays the Spectrum curve for every window (grey)
#False: displays only mean en +-std spectrum curves

FIG_width = 8 #cm #figure dimension width in cm
FIG_height = 8 #cm #figure dimension height in cm
ftsze = 8 #pt #figure fontsize for tests
lnwdth = 0.5 #figure curves linewidth
col1 = 'red'
col2 = 'lime'
col3 = 'navy'
FIG_FMIN = 0.1 #Hz #frequency axis lower bound
FIG_FMAX = 50 #Hz #frequency axis upper bound

SAVE_SPECTRUM = True #saves the SPECTRUM figure

"""SCRIPT"""

"""Load"""
import glob
import os
import sys
import csv
import numpy as np
from pykml import parser
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from collections import defaultdict

print ("> Running program: b_plot_spectrum.py") #prints message to screen
sys.stdout.flush() #forces the print command

"""Load spectra"""
DEFAULT_LIST = []
TEMP_LIST = glob.glob(SPECTRUM_DATAPATH + "POINT*")
for temp in TEMP_LIST:
    DEFAULT_LIST.append(os.path.basename(temp))

if not POINT_LIST:
    POINT_LIST = DEFAULT_LIST
    print (">> No POINTS were specified: Processing all points")
elif POINT_LIST:
    POINT_LIST = list(set(POINT_LIST).intersection(DEFAULT_LIST))
    
for point in POINT_LIST: #loop on stations
    shortname_point = os.path.basename(point)
    
    print(">> Processing " + shortname_point)
    sys.stdout.flush()
    comp_list = glob.glob(SPECTRUM_DATAPATH + shortname_point + "/" + shortname_point + "*.spec") # list of components for point
    
    """Initialization"""
    NB_SAMPLES = defaultdict(list) #number of frequency samples
    NB_WIN = defaultdict(list) #number of processing windows
    LG_WIN = defaultdict(list) #length of processing windows (s)
    FREQ = defaultdict(list) #frequency data
    SPEC = defaultdict(list) #spectrum data
    WIN_SPEC = defaultdict(list) #temporary spectrum data
    FREQ_stats = defaultdict(list) #frequency data
    SPEC_mean = defaultdict(list) #mean spectrum data
    SPEC_min = defaultdict(list) #mean spectrum data / std
    SPEC_max = defaultdict(list) #mean spectrum data * std
    DATA = defaultdict(list) #temporary data
    
    for comp in comp_list: #loop over 3C components
        COMPNAME = os.path.basename(comp).split('.spec').pop(0)[-1:] #component name
        print(">>> Processing " + COMPNAME) #prints message to screen
        sys.stdout.flush() #forces the print command
        
        print(">>>> Reading .log file ") #prints message to screen
        sys.stdout.flush() #forces the print command
        with open(SPECTRUM_DATAPATH + shortname_point + "/" + shortname_point + "_" + COMPNAME + ".log", 'r') as fh: #opens the .log file containing geopsy computation parameters
            text_logfile = fh.read()
            NB_SAMPLES[COMPNAME] = int(text_logfile.split('SAMPLES NUMBER FREQUENCY = ').pop(1).split('SAMPLING TYPE FREQUENCY').pop(0)) #reads number of frequency samples
            NB_WIN[COMPNAME] = int(text_logfile.split('# Number= ').pop(1).split('# Start time').pop(0)) #reads number of processing windows
            LG_WIN[COMPNAME] = float(text_logfile.split('WINDOW MIN LENGTH (s) = ').pop(1).split('WINDOW MAX LENGTH (s)').pop(0)) #reads length of processing windows (s)
            if not NB_WIN[COMPNAME]:
                NB_WIN[COMPNAME] = 0
            if not LG_WIN[COMPNAME]:
                LG_WIN[COMPNAME] = 0
        
        print(">>>> Reading .win file ") #prints message to screen
        sys.stdout.flush() #forces the print command
        with open(SPECTRUM_DATAPATH + shortname_point + "/" + shortname_point + "_" + COMPNAME + ".win", 'r') as fh: #opens the .win file containing geopsy computation windows
            text_winfile = fh.read()
            WIN_DATA = text_winfile.split('# Window')[1:]
            WIN_SPEC[COMPNAME] = np.empty([NB_SAMPLES[COMPNAME], 2, NB_WIN[COMPNAME]], dtype=float)
            for win_index, specdata in enumerate(WIN_DATA):
                FORMATED_DATA =  np.fromstring(specdata, dtype=float, sep='\t')
                FORMATED_DATA = np.delete(FORMATED_DATA, 0)
                WIN_SPEC[COMPNAME][:,:,win_index] = np.reshape(FORMATED_DATA, (-1, 2)) #window FREQ and SPEC storage
        FREQ[COMPNAME][:] = WIN_SPEC[COMPNAME][:,0,:]
        SPEC[COMPNAME][:] = WIN_SPEC[COMPNAME][:,1,:]
        
        print(">>>> Reading .spec file ") #prints message to screen
        sys.stdout.flush() #forces the print command
        with open(SPECTRUM_DATAPATH + shortname_point + "/" + shortname_point + "_" + COMPNAME + ".spec", 'r') as fh: #opens the .spec file containing geopsy computation mean and std
            reader = csv.reader(fh, delimiter='\t')
            if reader:
                for row in reader:
                    if row:
                        if not row[0].startswith("#"):
                            DATA[COMPNAME].append(row)
        del reader, row
    
    for key, value in DATA.iteritems():
        for line in value:
            FREQ_stats[key].append(float(line[0]))
            SPEC_mean[key].append(float(line[1]))
            SPEC_min[key].append(float(line[2]))
            SPEC_max[key].append(float(line[3]))
    del DATA, WIN_SPEC, FORMATED_DATA, WIN_DATA

    """########################################FIGURES########################################"""
    plt.close("all")
    
    """############ FIG1 LOGLOG ############"""
    print(">>> Plotting")
    sys.stdout.flush()
    fig = plt.figure(figsize=(FIG_width/2.54, FIG_height/2.54))

    for key, value in FREQ.iteritems():
        comp_name = key.split('.spec')[0][-1]
        if comp_name == 'Z':
            if DISP_ALL_CURVES:
                plt.loglog(FREQ[key], SPEC[key], '-', color=col1, linewidth=lnwdth, label='_nolegend_', alpha=0.1, zorder=0)
            plt.loglog(FREQ_stats[key], SPEC_mean[key], '-', color=col1, linewidth=3*lnwdth, label=comp_name, zorder=2)
            plt.loglog(FREQ_stats[key], SPEC_min[key], '--', color=col1, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.loglog(FREQ_stats[key], SPEC_max[key], '--', color=col1, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.text(0.1, 0.05, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))
        if comp_name == 'N':
            if DISP_ALL_CURVES:
                plt.loglog(FREQ[key], SPEC[key], '-', color=col2, linewidth=lnwdth, label='_nolegend_', alpha=0.1, zorder=0)
            plt.loglog(FREQ_stats[key], SPEC_mean[key], '-', color=col2, linewidth=3*lnwdth, label=comp_name, zorder=2)
            plt.loglog(FREQ_stats[key], SPEC_min[key], '--', color=col2, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.loglog(FREQ_stats[key], SPEC_max[key], '--', color=col2, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.text(0.1, 0.15, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))
        if comp_name == 'E':
            if DISP_ALL_CURVES:
                plt.loglog(FREQ[key], SPEC[key], '-', color=col3, linewidth=lnwdth, label='_nolegend_', alpha=0.1, zorder=0)
            plt.loglog(FREQ_stats[key], SPEC_mean[key], '-', color=col3, linewidth=3*lnwdth, label=comp_name, zorder=2)
            plt.loglog(FREQ_stats[key], SPEC_min[key], '--', color=col3, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.loglog(FREQ_stats[key], SPEC_max[key], '--', color=col3, linewidth=lnwdth, label='_nolegend_', zorder=1)
            plt.text(0.1, 0.25, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(FIG_FMIN, FIG_FMAX)
    plt.title(shortname_point)
    plt.xlabel('f [Hz]')
    plt.ylabel('Spectrum')
    plt.legend(labelspacing=0.1, prop={'size':8})
    
    """SAVE""" 
    if SAVE_SPECTRUM:
        if not os.path.exists(SPECTRUM_FIG_PATH +  shortname_point): #if folder for saving spectrum does not exist
            os.makedirs(SPECTRUM_FIG_PATH +  shortname_point) #create folder
        print(">>> Saving")
        plt.savefig(SPECTRUM_FIG_PATH + shortname_point + "/" + shortname_point + "_log.pdf", format='pdf', bbox_inches='tight')
    else:
        plt.show()
    
    """############ FIG2 SEMILOG ############"""
    print(">>> Plotting")
    sys.stdout.flush()
    fig = plt.figure(figsize=(FIG_width/2.54, FIG_height/2.54))

    for key, value in FREQ.iteritems():
        comp_name = key.split('.spec')[0][-1]
        if comp_name == 'Z':
            if DISP_ALL_CURVES:
                plt.semilogx(FREQ[key], SPEC[key], '-', color=col1, linewidth=lnwdth, label='_nolegend_', alpha=0.1)
            plt.semilogx(FREQ_stats[key], SPEC_mean[key], '-', color=col1, linewidth=3*lnwdth, label=comp_name)
            plt.semilogx(FREQ_stats[key], SPEC_min[key], '--', color=col1, linewidth=lnwdth, label='_nolegend_')
            plt.semilogx(FREQ_stats[key], SPEC_max[key], '--', color=col1, linewidth=lnwdth, label='_nolegend_')
            plt.text(0.1, 0.7, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))
        if comp_name == 'N':
            if DISP_ALL_CURVES:
                plt.semilogx(FREQ[key], SPEC[key], '-', color=col2, linewidth=lnwdth, label='_nolegend_', alpha=0.1)
            plt.semilogx(FREQ_stats[key], SPEC_mean[key], '-', color=col2, linewidth=3*lnwdth, label=comp_name)
            plt.semilogx(FREQ_stats[key], SPEC_min[key], '--', color=col2, linewidth=lnwdth, label='_nolegend_')
            plt.semilogx(FREQ_stats[key], SPEC_max[key], '--', color=col2, linewidth=lnwdth, label='_nolegend_')
            plt.text(0.1, 0.8, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))
        if comp_name == 'E':
            if DISP_ALL_CURVES:
                plt.semilogx(FREQ[key], SPEC[key], '-', color=col3, linewidth=lnwdth, label='_nolegend_', alpha=0.1)
            plt.semilogx(FREQ_stats[key], SPEC_mean[key], '-', color=col3, linewidth=3*lnwdth, label=comp_name)
            plt.semilogx(FREQ_stats[key], SPEC_min[key], '--', color=col3, linewidth=lnwdth, label='_nolegend_')
            plt.semilogx(FREQ_stats[key], SPEC_max[key], '--', color=col3, linewidth=lnwdth, label='_nolegend_')
            plt.text(0.1, 0.9, r'$n_{win}$'+ comp_name + ' = ' + str(int(NB_WIN[key])), fontsize = ftsze, transform=plt.gca().transAxes,
                bbox=dict(facecolor='w', edgecolor='None'))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(FIG_FMIN, FIG_FMAX)
    plt.title(shortname_point)
    plt.xlabel('f [Hz]')
    plt.ylabel('Spectrum')
    plt.legend(labelspacing=0.1, prop={'size':8})
    
    """SAVE""" 
    if SAVE_SPECTRUM:
        if not os.path.exists(SPECTRUM_FIG_PATH +  shortname_point): #if folder for saving spectrum does not exist
            os.makedirs(SPECTRUM_FIG_PATH +  shortname_point) #create folder
        print(">>> Saving")
        plt.savefig(SPECTRUM_FIG_PATH + shortname_point + "/" + shortname_point + "_lin.pdf", format='pdf', bbox_inches='tight')
    else:
        plt.show()