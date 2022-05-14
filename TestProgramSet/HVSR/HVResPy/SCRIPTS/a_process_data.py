# -*- coding:Latin-1 -*-
"""########################################DESCRIPTION########################################"""
"""a_process_data
This script requires Geopsy software (www.geopsy.org, Marc WATHELET, ISTerre) installed on the computer.
It must be run on a linux-compatible terminal (for Windows users, you can install MSYS terminal. See instructions 
from Marc WATHELET on www.geopsy.org forum).

It calls geopsy to compute for each measurement point:
-the spectrum of each component (Z,N,E for 3C sensor) (saved in ../DATA/SPECTRUM/POINT)
-the H/V ratio (saved in ../DATA/HV/POINT)
It requires a parameter .log file (see this example or geopsy help online).
The input parameters are listed in the 'PARAMETERS' section hereafter.

Make sure that "output" folder in geopsy graphic window is unchecked.

Reference: please cite as: Bottelin, P., (2015) An open-source Python tool for Geopsy HVSR post-processing.
Contact: Pierre BOTTELIN
Post-doctoral teacher and researcher
Applied geophysics, geotechnical engineering and environment
IRAP / OMP Toulouse
pierre.bottelin@irap.omp.eu
"""

"""########################################PARAMETERS########################################"""
POINT_LIST = [] #list of points to process. If empty, process all the points.
PARAM_DATAPATH = "./" #path to parameter.log file
SEISMIC_DATAPATH = "../DATA/SEISMIC_DATA/" #path to seismic data folders
SPECTRUM_DATAPATH = "../DATA/SPECTRUM_DATA/" #path to spectrum data folders
HV_DATAPATH = "../DATA/HV_DATA/" #path to HVSR data folders
SEISMIC_FORMAT = "SAC" #SEED, SAC, or any seismic data format recognized by Geopsy

"""########################################MAIN SCRIPT########################################"""

"""MODULE IMPORTS"""
import os
import glob
import shutil
import sys
import subprocess

"""SCRIPT"""
print ("> Running program: a_process_data.py") #prints message to screen
sys.stdout.flush() #forces the print command

DEFAULT_LIST = []
TEMP_LIST = glob.glob(SEISMIC_DATAPATH + "POINT*") #detect all the folders containing "POINT" in their name
for temp in TEMP_LIST:
    DEFAULT_LIST.append(os.path.basename(temp))

if not POINT_LIST:  #if empty point list
    POINT_LIST = DEFAULT_LIST
    print (">> No POINTS were specified: Processing all points")
elif POINT_LIST:  #if point list
    POINT_LIST = list(set(POINT_LIST).intersection(DEFAULT_LIST)) #comparison between user POINT_LIST and folder SEISMIC_DATAPATH
    
for point in POINT_LIST: #loop on the point list
    shortname_point = os.path.splitext(os.path.basename(point))[0] #keeps only the "POINT**" folder name instead of the whole path name
    
    print(">> Processing " + shortname_point) #prints message to screen
    sys.stdout.flush() #forces the print command
    
    """Remove existing files"""
    list_existing_files = glob.glob(SPECTRUM_DATAPATH + shortname_point + "/" + shortname_point + "*.*") #list of existing spectrum files in folder "SPECTRUM_DATA/POINT**/"
    if list_existing_files:
        print (">>> Removing existing files") #prints message to screen
        sys.stdout.flush() #forces the print command
        for existing_file in list_existing_files:
            os.remove(existing_file) #removes files
    list_existing_files = glob.glob(SPECTRUM_DATAPATH + shortname_point + "*.*") #list of existing spectrum files in folder "SPECTRUM_DATA/"
    if list_existing_files:
        print (">>> Removing existing files") #prints message to screen
        sys.stdout.flush() #forces the print command
        for existing_file in list_existing_files:
            os.remove(existing_file) #removes files
    list_existing_files = glob.glob(HV_DATAPATH + shortname_point + "/" + shortname_point + "*.*") #list of existing spectrum files in folder "SPECTRUM_DATA/POINT**/"
    if list_existing_files:
        print (">>> Removing existing files") #prints message to screen
        sys.stdout.flush() #forces the print command
        for existing_file in list_existing_files:
            os.remove(existing_file) #removes files
    list_existing_files = glob.glob(HV_DATAPATH + shortname_point + "*.*") #list of existing spectrum files in folder "SPECTRUM_DATA/"
    if list_existing_files:
        print (">>> Removing existing files") #prints message to screen
        sys.stdout.flush() #forces the print command
        for existing_file in list_existing_files:
            os.remove(existing_file) #removes files

    """Processing seismic files"""
    list_seismic_files = glob.glob(SEISMIC_DATAPATH + shortname_point + "/" + shortname_point + "*." + SEISMIC_FORMAT) #list of seismic data files in folder "SEISMIC_DATAPATH/POINT**"
    if not list_seismic_files:
        print (">>> No seismic data file found. Skipping point " + shortname_point) #prints message to screen
        sys.stdout.flush() #forces the print command
    else:
        """SPECTRUM"""
        print (">>> Processing spectrum") #prints message to screen
        sys.stdout.flush() #forces the print command
        
        if not os.path.exists(SPECTRUM_DATAPATH +  shortname_point): #if folder for saving spectrum does not exist
            os.makedirs(SPECTRUM_DATAPATH +  shortname_point) #create folder
        
        for seismic_file in list_seismic_files:
            COMPNAME = os.path.basename(seismic_file).split('.' + SEISMIC_FORMAT).pop(0)[-1:] #component name
            with open(SPECTRUM_DATAPATH +  shortname_point + "/" + shortname_point + "_" + COMPNAME + ".win", "w") as fh: #opens a .win file for storing geopsy computation windows
                cmd = "geopsy " + ' ' + seismic_file + " -- -tool geopsyhv -slot 0 -param " + PARAM_DATAPATH + "param_process.log -autowin -curves -clearpeaks -save " + SPECTRUM_DATAPATH #builds the geopsy command
                subprocess.call(cmd, stdout=fh, shell=True) #calls geopsy and stores output in fh file
            list_output_files = glob.glob(SPECTRUM_DATAPATH + shortname_point + "_" + COMPNAME + "*.*") #list of output files for component COMPNAME in folder "SPECTRUM_DATAPATH"
            for output_file in list_output_files:
                shutil.move(output_file, SPECTRUM_DATAPATH + shortname_point + "/" + os.path.splitext(os.path.basename(output_file))[0] + os.path.splitext(os.path.basename(output_file))[1]) #moves the file in appropriate directory

        """HVSR"""
        print (">>> Processing hvsr") #prints message to screen
        sys.stdout.flush() #forces the print command
        
        if not os.path.exists(HV_DATAPATH +  shortname_point): #if folder for saving hvsr does not exist
            os.makedirs(HV_DATAPATH +  shortname_point) #create folder
            
        with open(HV_DATAPATH +  shortname_point + ".win", "w") as fh:
            cmd = "geopsy " + ' '.join(list_seismic_files) + " -- -tool geopsyhv -slot 1 -param " + PARAM_DATAPATH + "param_process.log -autowin -curves -clearpeaks -addpeak 0.5,50 -save " + HV_DATAPATH #builds the geopsy command
            subprocess.call(cmd, stdout=fh, shell=True) #calls geopsy and stores output in fh file
        list_output_files = glob.glob(HV_DATAPATH + shortname_point + "*.*") #list of output files in folder "HV_DATAPATH"
        for output_file in list_output_files:
            shutil.move(output_file, HV_DATAPATH + shortname_point + "/" + os.path.splitext(os.path.basename(output_file))[0] + os.path.splitext(os.path.basename(output_file))[1]) #moves the file in appropriate directory