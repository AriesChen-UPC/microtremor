# -*- coding:Latin-1 -*-
"""########################################DESCRIPTION########################################"""
"""c_plot_hvsr
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
POINT_LIST = ["POINT24","POINT42","POINT44","POINT47","POINT48","POINT49"] #list of points to process. If empty, process all the points.
FMIN = [0.7,0.7,0.7,0.7,0.7,0.7] #Hz #list of minimal frequency for peak detection range. If isempty, set to 0.1 Hz: !! warning: risk of wrong peak peaking !!
FMAX = [8,8,8,8,8,8] #Hz #list of maximal frequency for peak detection range. If isempty, set to 100 Hz: !! warning: risk of wrong peak peaking !!

HV_DATAPATH = "../DATA/HV_DATA/" #path to HVSR data folders
MAP_DATAPATH = "../MAP/" #path to MAP data folders
HV_FIG_PATH = "../FIGURES/HV_FIGURES/" #path to HVSR figure folders
COORD_FILE_DATAPATH = "../DATA/COORD_DATA/" #path to points COORDinates data folders

DISP_ALL_CURVES = True
#True: displays the HVSR curve for every window (grey) and selected portion for peak detection (blue)
#False: displays only mean en +-std HVSR curves

FIG_width = 11 #cm #figure dimension width in cm
FIG_height = 8 #cm #figure dimension height in cm
ftsze = 8 #pt #figure fontsize for tests
lnwdth = 0.1 #pt linewidth
FIG_FMIN = 0.1 #Hz #frequency axis lower bound
FIG_FMAX = 50 #Hz #frequency axis upper bound
FIG_HVMIN_LOG = 0.1 #HVSR axis lower bound for loglog figure
FIG_HVMAX_LOG = 100 #HVSR axis lower bound for loglog figure
FIG_HVMIN_LIN = 0 #HVSR axis lower bound for semilog figure
FIG_HVMAX_LIN = 30 #HVSR axis lower bound for semilog figure

SAVE_HV = True #saves the HVSR figure
SAVE_TXT = True #saves the text file with F0 peaks and coordinates for plotting

"""########################################MAIN SCRIPT########################################"""

"""MODULE IMPORTS"""
import glob
import os
import sys
import numpy as np
from pykml import parser
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

"""FUNCTIONS"""
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
"""SCRIPT"""
print ("> Running program: c_plot_hvsr.py") #prints message to screen
sys.stdout.flush() #forces the print command

"""Text file initialization"""
if SAVE_TXT: #if text file save is activated
    list_F0_files = glob.glob(MAP_DATAPATH + "F0*.txt") #list of existing F0* peaks files
    for F0_file in list_F0_files:
        os.remove(F0_file) #delete list of existing F0* peaks files
    with open(MAP_DATAPATH + "F0_RELIABLE.txt", 'w') as fh: #open text file for saving reliable F0 peaks
            fh.write("POINT" + "\t" + "F0(Hz)" + "\t" + "Lon(°)" + "\t" + "Lat(°)" + "\n") #prints header line
    with open(MAP_DATAPATH + "F0_NOT_RELIABLE.txt", 'w') as fh: #open text file for saving NOT reliable F0 peaks
        fh.write("POINT" + "\t" + "F0(Hz)" + "\t" + "Lon(°)" + "\t" + "Lat(°)" + "\n") #prints header line
    with open(MAP_DATAPATH + "F0_DUMP.txt", 'w') as fh: #open text file for saving DUMP HVSR curves
        fh.write("POINT" + "\t" + "F0(Hz)" + "\t" + "Lon(°)" + "\t" + "Lat(°)" + "\n") #prints header line
        
"""Load point coordinates"""
list_kml_files = glob.glob(COORD_FILE_DATAPATH + "*.kml") #list of existing F0* peaks files
if not list_kml_files:
    print (">> No .kml file found. No point coordinates will be saved")
elif len(list_kml_files) != 1:
    print (">> Too many .kml files found. No point coordinates will be saved")
elif len(list_kml_files) == 1:
    print (">> One .kml file found. Point coordinates will be saved")
    root = parser.fromstring(open(list_kml_files[0],'r').read())
    dict_coord = {}
    for point in root.Document.Placemark:
        dict_coord[point.name.text] = point.Point.coordinates.text
    coord = []
    for name, position in dict_coord.iteritems():
        temp = [name, np.array(position.split(','))[0].astype(np.float), np.array(position.split(','))[1].astype(np.float)]
        coord.append(temp)
    coord.sort()
    STA_HV = []
    LON_HV = []
    LAT_HV = []
    for point in coord:
        STA_HV.append(point[0]) #station name
        LON_HV.append(point[1]) #station longitude (°)
        LAT_HV.append(point[2]) #station latitude (°)

"""Load HVSR files"""
DEFAULT_LIST = []
TEMP_LIST = glob.glob(HV_DATAPATH + "POINT*") #detect all the folders containing "POINT" in their name
for temp in TEMP_LIST:
    DEFAULT_LIST.append(os.path.basename(temp))

if not POINT_LIST:  #if empty point list
    POINT_LIST = DEFAULT_LIST
    FMIN = 0.1* np.ones(len(POINT_LIST)) #0.1 Hz FMIN default !! risk of wrong peak peaking if left as default!!!
    FMAX = 100* np.ones(len(POINT_LIST)) #100 Hz FMAX default !! risk of wrong peak peaking if left as default!!!
    print (">> No POINTS were specified: Processing all points")
elif POINT_LIST:  #if point list
    POINT_LIST = list(set(POINT_LIST).intersection(DEFAULT_LIST)) #comparison between user POINT_LIST and folder SEISMIC_DATAPATH
    print (">> Processing points in POINT LIST. Please check FMIN and FMAX for peak picking.")
    
for idx_point, point in enumerate(POINT_LIST): #loop on stations
    shortname_point = os.path.basename(point)
    
    print(">> Processing " + shortname_point) #prints message to screen
    sys.stdout.flush()

    print(">>> Reading .log file ") #prints message to screen
    sys.stdout.flush()
    with open(HV_DATAPATH + shortname_point + "/" + shortname_point + ".log", 'r') as fh: #reads .log file
        text_logfile = fh.read()
        NB_SAMPLES = int(text_logfile.split('SAMPLES NUMBER FREQUENCY = ').pop(1).split('SAMPLING TYPE FREQUENCY').pop(0)) #reads number of frequency samples
        NB_WIN = int(text_logfile.split('# Number= ').pop(1).split('# Start time').pop(0)) #reads number of windows
        LG_WIN = float(text_logfile.split('WINDOW MIN LENGTH (s) = ').pop(1).split('WINDOW MAX LENGTH (s)').pop(0)) #reads window length
        if not NB_WIN:
            NB_WIN = 0
        if not LG_WIN:
            LG_WIN = 0
            
    print(">>> Reading .win file ") #prints message to screen
    sys.stdout.flush()
    with open(HV_DATAPATH + shortname_point + "/" + shortname_point + ".win", 'r') as fh: #reads .win file
        text_winfile = fh.read()
        WIN_DATA = text_winfile.split('# Window')[1:]
        WIN_HVSR_STORE = np.empty([NB_SAMPLES, 2, NB_WIN], dtype=float)
        for win_index, hvdata in enumerate(WIN_DATA):
            FORMATED_DATA =  np.fromstring(hvdata, dtype=float, sep='\t')
            FORMATED_DATA = np.delete(FORMATED_DATA, 0)
            WIN_HVSR = np.reshape(FORMATED_DATA, (-1, 2))
            WIN_HVSR_STORE[:,:,win_index] = WIN_HVSR
            
    """SPLITTING""" #splits data into frequency, HVSR arrays
    VECT_F = WIN_HVSR_STORE[:,0,0]
    WIN_HVSR_SPLIT = np.empty([NB_SAMPLES, WIN_HVSR_STORE.shape[2]], dtype=float)
    WIN_HVSR_SPLIT[:,:] = WIN_HVSR_STORE[:,1,:]
    
    """HVSR STATS"""
    WIN_HVSR_SPLIT_CLIPPED = np.clip(WIN_HVSR_SPLIT, 0.1, 100) #clipping at HVSR = 100 maximum and 0.1 minimum
    HVSR_MEAN = np.mean(WIN_HVSR_SPLIT_CLIPPED, axis=1) #computes mean HVSR
    HVSR_STD = np.std(WIN_HVSR_SPLIT_CLIPPED, axis=1) #computes HVSR standard deviation
    HVSR_MAX = np.add(HVSR_MEAN, HVSR_STD) #HVSR+std(HVSR) curve
    HVSR_MAX = np.minimum(HVSR_MAX, 100) #HVSR+std(HVSR) curve clipped to 100
    HVSR_MIN = np.subtract(HVSR_MEAN, HVSR_STD) #HVSR-std(HVSR) curve
    HVSR_MIN = np.maximum(HVSR_MIN, 0.1) #HVSR+std(HVSR) curve clipped to 0.1
    
    """#CUT""" #cuts data into [FMIN FMAX] range
    cut_min_f=find_nearest_idx(VECT_F,FMIN[idx_point])
    cut_max_f=find_nearest_idx(VECT_F,FMAX[idx_point])
    VECT_F_CUT = VECT_F[cut_min_f:cut_max_f]
    HVSR_CUT = WIN_HVSR_SPLIT_CLIPPED[cut_min_f:cut_max_f,:]
    LOCAL_MAX_INDEX = np.empty([1, HVSR_CUT.shape[1]], dtype=float)
    HVSR_MEAN_CUT = HVSR_MEAN[cut_min_f:cut_max_f]
    
    """#F0""" #HVSR peak detection in [FMIN FMAX] range
    idx_HVSR_peak = HVSR_MEAN_CUT.argmax(axis=0) #maximum of mean HVSR curve
    F0_peak = round(VECT_F_CUT[idx_HVSR_peak],2) #frequency (F0) of HVSR peak
    LOCAL_MAX_VAL = HVSR_CUT.max(axis=0)
    LOCAL_MAX_INDEX = HVSR_CUT.argmax(axis=0)
    F0_mean = np.mean(VECT_F_CUT[LOCAL_MAX_INDEX]) #mean frequency (F0*) of individual HVSR curves (see geopsy for more details)
    idx_F0_mean = find_nearest_idx(VECT_F,F0_mean)
    A0 = HVSR_MEAN[idx_F0_mean] #peak amplitude
    STD_F0 = np.std(VECT_F_CUT[LOCAL_MAX_INDEX]) #F0 standard deviation
    F0_max = F0_mean + STD_F0 #upper limit of F0 wander
    F0_min = F0_mean - STD_F0 #lower limit of F0 wander

    """SESAME guidelines verification"""
    """(see Bard and the SESAME Team (2004) Guidelines for the implementation of the H/V spectral ratio
    technique of ambient vibrations, SESAME European research project WP12 – Deliverable D23.12)"""
    
    """CURVE CHECK"""

    """Test1: window length"""
    if F0_peak > (10/LG_WIN):
        TEST1_win_length = True
    else:
            TEST1_win_length = False

    """Test2: number of cycles"""
    if LG_WIN*NB_WIN*F0_peak > 200:
        TEST2_nb_cycles = True
    else:
         TEST2_nb_cycles = False

    """Test3: standard deviation"""
    cut_min_f=find_nearest_idx(VECT_F,0.5*F0_peak)
    cut_max_f=find_nearest_idx(VECT_F,2*F0_peak)
    f_cut = VECT_F[cut_min_f:cut_max_f]
    hvsr_max_cut = HVSR_MAX[cut_min_f:cut_max_f]
    hvsr_min_cut = HVSR_MAX[cut_min_f:cut_max_f]
    hvsr_mean_cut = HVSR_MEAN[cut_min_f:cut_max_f]
    RATIOMAX=hvsr_max_cut/hvsr_mean_cut
    RATIOMIN=hvsr_mean_cut/hvsr_min_cut
    if F0_peak>0.5:
        if all(i<2 for i in RATIOMAX) and all(i<2 for i in RATIOMIN):
            TEST3_std = True
        else:
            TEST3_std = False
    elif F0_peak<=0.5:
        if all(i<3 for i in RATIOMAX) and all(i<3 for i in RATIOMIN):
            TEST3_std = True
        else:
            TEST3_std = False
    else:
        TEST3_std = False
    
    """HVSR PEAK CHECK"""
    THRESHOLD = A0/2

    """Test4: peak left"""
    cut_min_f=find_nearest_idx(VECT_F,0.25*F0_peak)
    cut_max_f=find_nearest_idx(VECT_F,F0_peak)
    f_cut_left = VECT_F[cut_min_f:cut_max_f]
    hvsr_cut_left = HVSR_MEAN[cut_min_f:cut_max_f]
    idx_match_left = [i for i,v in enumerate(hvsr_cut_left) if v < THRESHOLD]
    if idx_match_left:
        TEST4_peak_left = True
    else:
        TEST4_peak_left = False

    """Test5: peak right"""
    cut_min_f=find_nearest_idx(VECT_F,F0_peak)
    cut_max_f=find_nearest_idx(VECT_F,4*F0_peak)
    f_cut_right = VECT_F[cut_min_f:cut_max_f]
    hvsr_cut_right = HVSR_MEAN[cut_min_f:cut_max_f]
    idx_match_right = [i for i,v in enumerate(hvsr_cut_right) if v < THRESHOLD]
    if idx_match_right:
        TEST5_peak_right = True
    else:
        TEST5_peak_right = False

    """Test6: amplification"""
    if A0>2:
        TEST6_amplification = True
    else:
        TEST6_amplification = False

    """Test7: f0 wandering"""
    cut_min_f=find_nearest_idx(VECT_F,FMIN[idx_point])
    cut_max_f=find_nearest_idx(VECT_F,FMAX[idx_point])
    VECT_F_CUT = VECT_F[cut_min_f:cut_max_f]
    HVSR_MAX_CUT = HVSR_MAX[cut_min_f:cut_max_f]
    HVSR_MIN_CUT = HVSR_MIN[cut_min_f:cut_max_f]
    Peak_curve_max_index = HVSR_MAX_CUT.argmax(axis=0)
    Peak_curve_min_index = HVSR_MIN_CUT.argmax(axis=0)
    F_Peak_curve_max = VECT_F_CUT[Peak_curve_max_index]
    F_Peak_curve_min = VECT_F_CUT[Peak_curve_min_index]
    if (0.95*F0_peak<F_Peak_curve_max<1.05*F0_peak) and (0.95*F0_peak<F_Peak_curve_min<1.05*F0_peak):
        TEST7_f0_wandering = True
    else:
        TEST7_f0_wandering = False

    """Test8: peak amplitude wandering"""
    AMAX = HVSR_MAX[idx_F0_mean]
    AMIN = HVSR_MAX[idx_F0_mean]
    if F0_peak<0.2:
        E0 = 0.25*F0_peak
        if STD_F0 < E0:
            TEST8_peak_ampl_threshold = True
        else:
            TEST8_peak_ampl_threshold = False
        T0 = 3.0
        if AMAX/A0<T0:
            TEST9_std_f0_threshold = True
        else:
            TEST9_std_f0_threshold = False
    elif 0.2<=F0_peak<0.5:
        E0 = 0.20*F0_peak
        if STD_F0 < E0:
            TEST8_peak_ampl_threshold = True
        else:
            TEST8_peak_ampl_threshold = False
        T0 = 2.5
        if AMAX/A0<T0:
            TEST9_std_f0_threshold = True
        else:
            TEST9_std_f0_threshold = False
    elif 0.5<=F0_peak<1.0:
        E0 = 0.15*F0_peak
        if STD_F0 < E0:
            TEST8_peak_ampl_threshold = True
        else:
            TEST8_peak_ampl_threshold = False
        T0 = 2.0
        if AMAX/A0<T0:
            TEST9_std_f0_threshold = True
        else:
            TEST9_std_f0_threshold = False
    elif 1.0<=F0_peak<2.0:
        E0 = 0.10*F0_peak
        if STD_F0 < E0:
            TEST8_peak_ampl_threshold = True
        else:
            TEST8_peak_ampl_threshold = False
        T0 = 1.78
        if AMAX/A0<T0:
            TEST9_std_f0_threshold = True
        else:
            TEST9_std_f0_threshold = False
    elif F0_peak>2.0:
        E0 = 0.05*F0_peak
        if STD_F0 < E0:
            TEST8_peak_ampl_threshold = True
        else:
            TEST8_peak_ampl_threshold = False
        T0 = 1.58
        if AMAX/A0<T0:
            TEST9_std_f0_threshold = True
        else:
            TEST9_std_f0_threshold = False
    
    """Conclusions about CURVE and PEAK reliability"""
    if sum([TEST1_win_length, TEST2_nb_cycles, TEST3_std])>=3:
        CURVE_TEST = True
    else:
        CURVE_TEST = False
    if sum([TEST4_peak_left, TEST5_peak_right, TEST6_amplification, TEST7_f0_wandering, TEST8_peak_ampl_threshold, TEST9_std_f0_threshold])>=5:
        PEAK_TEST = True
    else:
        PEAK_TEST = False

    """########################################FIGURES########################################"""
    print(">>> Plotting")
    sys.stdout.flush()
    plt.close("all")
    
    """############ FIG1 LOGLOG ############"""
    FIG1 = plt.figure(figsize=(FIG_width/2.54, FIG_height/2.54))
    ax1 = FIG1.add_axes((0.1,0.1,.6,0.8))
    """STATS F0"""
    plt.loglog((F0_peak, F0_peak), (FIG_HVMIN_LOG, FIG_HVMAX_LOG), '-', color='k', linewidth=2*lnwdth)
    plt.loglog((F0_mean, F0_mean), (FIG_HVMIN_LOG, FIG_HVMAX_LOG), '-', color='0.75', linewidth=2*lnwdth)
    plt.loglog((F0_min, F0_min), (FIG_HVMIN_LOG, FIG_HVMAX_LOG), '--', color='0.75', linewidth=lnwdth)
    plt.loglog((F0_max, F0_max), (FIG_HVMIN_LOG, FIG_HVMAX_LOG), '--', color='0.75', linewidth=lnwdth)
    """ALL CURVES"""
    if DISP_ALL_CURVES:
        for win_index in range(WIN_HVSR_SPLIT.shape[1]):
            plt.loglog(VECT_F, WIN_HVSR_SPLIT[:,win_index], '-', color='0.8', linewidth=lnwdth, alpha=0.6)
            plt.loglog(VECT_F_CUT, HVSR_CUT[:,win_index], '-', color='blue', linewidth=lnwdth, alpha=0.6)
    """STATS HVSR"""
    plt.loglog(VECT_F, HVSR_MEAN, '-', color='k', linewidth=10*lnwdth)
    plt.loglog(VECT_F, HVSR_MAX, '--', color='k', linewidth=4*lnwdth)
    plt.loglog(VECT_F, HVSR_MIN, '--', color='k', linewidth=4*lnwdth)
    plt.loglog(f_cut_left[idx_match_left], hvsr_cut_left[idx_match_left], '-', color='green',
        linewidth=10*lnwdth)
    plt.loglog(f_cut_right[idx_match_right], hvsr_cut_right[idx_match_right], '-', color='green',
        linewidth=10*lnwdth)
    """GAL"""
    plt.grid(True)
    plt.xlim(FIG_FMIN, FIG_FMAX)
    plt.ylim(FIG_HVMIN_LOG, FIG_HVMAX_LOG)
    plt.title(shortname_point)
    plt.xlabel("f [Hz]")
    plt.ylabel("H/V")
    
    if CURVE_TEST: FIG1.text(.72,.85,"CURVE:", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.25,"CURVE:", color='r', fontsize = ftsze)
    if TEST1_win_length: FIG1.text(.72,.8,"Window length", color='g', fontsize = ftsze)
    else: FIG1.text(.75,.8,"Window length", color='r', fontsize = ftsze)
    if TEST2_nb_cycles: FIG1.text(.72,.75,"Nb of cycles", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.75,"Nb of cycles", color='r', fontsize = ftsze)
    if TEST3_std: FIG1.text(.72,.70,"Standard deviation", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.70,"Standard deviation", color='r', fontsize = ftsze)
    if PEAK_TEST: FIG1.text(.72,.60,"PEAK:", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.60,"PEAK:", color='r', fontsize = ftsze)
    if TEST4_peak_left: FIG1.text(.72,.55,"Peak left side", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.55,"Peak left side", color='r', fontsize = ftsze)
    if TEST5_peak_right: FIG1.text(.72,.50,"Peak right side", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.50,"Peak right side", color='r', fontsize = ftsze)
    if TEST6_amplification: FIG1.text(.72,.45,"Amplification", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.45,"Amplification", color='r', fontsize = ftsze)
    if TEST7_f0_wandering: FIG1.text(.72,.40,"f0 wandering", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.40,"f0 wandering", color='r', fontsize = ftsze)
    if TEST8_peak_ampl_threshold: FIG1.text(.72,.35,"Peak ampl. wander", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.35,"Peak ampl. wander", color='r', fontsize = ftsze)
    if TEST9_std_f0_threshold: FIG1.text(.72,.30,"Peak std. wander", color='g', fontsize = ftsze)
    else: FIG1.text(.72,.30,"Peak std. wander", color='r', fontsize = ftsze)
    
    plt.text(5, 55, r'$n_{win} = $' + str(NB_WIN), fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 32, r'$f_0 = $' + str(round(F0_peak,2)) + r' $Hz$', fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 18, r'$f_* = $' + str(round(F0_mean,2)) + r' $Hz$', fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 10, r'$A_0 = $' + str(round(A0,2)), fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))

    """SAVE"""  
    if SAVE_HV:
        if not os.path.exists(HV_FIG_PATH +  shortname_point): #if folder for saving hvsr does not exist
            os.makedirs(HV_FIG_PATH +  shortname_point) #create folder
        print(">>> Saving")
        plt.savefig(HV_FIG_PATH + shortname_point + "/" + shortname_point + "_log.pdf", format='pdf', bbox_inches='tight')
    else:
        plt.show()

        """############ FIG2 LOGLIN ############"""
    FIG2 = plt.figure(figsize=(FIG_width/2.54, FIG_height/2.54))
    ax2 = FIG2.add_axes((0.1,0.1,.6,0.8))
    """STATS F0"""
    plt.semilogx((F0_peak, F0_peak), (FIG_HVMIN_LIN, FIG_HVMAX_LIN), '-', color='k', linewidth=2*lnwdth)
    plt.semilogx((F0_mean, F0_mean), (FIG_HVMIN_LIN, FIG_HVMAX_LIN), '-', color='0.75', linewidth=2*lnwdth)
    plt.semilogx((F0_min, F0_min), (FIG_HVMIN_LIN, FIG_HVMAX_LIN), '--', color='0.75', linewidth=lnwdth)
    plt.semilogx((F0_max, F0_max), (FIG_HVMIN_LIN, FIG_HVMAX_LIN), '--', color='0.75', linewidth=lnwdth)
    """ALL CURVES"""
    if DISP_ALL_CURVES:
        for win_index in range(WIN_HVSR_SPLIT.shape[1]):
            plt.semilogx(VECT_F, WIN_HVSR_SPLIT[:,win_index], '-', color='0.8', linewidth=lnwdth, alpha=0.6)
            plt.semilogx(VECT_F_CUT, HVSR_CUT[:,win_index], '-', color='blue', linewidth=lnwdth, alpha=0.6)
    """STATS HVSR"""
    plt.semilogx(VECT_F, HVSR_MEAN, '-', color='k', linewidth=10*lnwdth)
    plt.semilogx(VECT_F, HVSR_MAX, '--', color='k', linewidth=4*lnwdth)
    plt.semilogx(VECT_F, HVSR_MIN, '--', color='k', linewidth=4*lnwdth)
    plt.semilogx(f_cut_left[idx_match_left], hvsr_cut_left[idx_match_left], '-', color='green',
        linewidth=10*lnwdth)
    plt.semilogx(f_cut_right[idx_match_right], hvsr_cut_right[idx_match_right], '-', color='green',
        linewidth=10*lnwdth)
    """GAL"""
    plt.grid(True)
    plt.xlim(FIG_FMIN, FIG_FMAX)
    plt.ylim(FIG_HVMIN_LIN, FIG_HVMAX_LIN)
    plt.title(shortname_point)
    plt.xlabel("f [Hz]")
    plt.ylabel("H/V")
    
    if CURVE_TEST: FIG2.text(.72,.85,"CURVE:", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.25,"CURVE:", color='r', fontsize = ftsze)
    if TEST1_win_length: FIG2.text(.72,.8,"Window length", color='g', fontsize = ftsze)
    else: FIG2.text(.75,.8,"Window length", color='r', fontsize = ftsze)
    if TEST2_nb_cycles: FIG2.text(.72,.75,"Nb of cycles", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.75,"Nb of cycles", color='r', fontsize = ftsze)
    if TEST3_std: FIG2.text(.72,.70,"Standard deviation", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.70,"Standard deviation", color='r', fontsize = ftsze)
    if PEAK_TEST: FIG2.text(.72,.60,"PEAK:", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.60,"PEAK:", color='r', fontsize = ftsze)
    if TEST4_peak_left: FIG2.text(.72,.55,"Peak left side", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.55,"Peak left side", color='r', fontsize = ftsze)
    if TEST5_peak_right: FIG2.text(.72,.50,"Peak right side", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.50,"Peak right side", color='r', fontsize = ftsze)
    if TEST6_amplification: FIG2.text(.72,.45,"Amplification", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.45,"Amplification", color='r', fontsize = ftsze)
    if TEST7_f0_wandering: FIG2.text(.72,.40,"f0 wandering", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.40,"f0 wandering", color='r', fontsize = ftsze)
    if TEST8_peak_ampl_threshold: FIG2.text(.72,.35,"Peak ampl. wander", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.35,"Peak ampl. wander", color='r', fontsize = ftsze)
    if TEST9_std_f0_threshold: FIG2.text(.72,.30,"Peak std. wander", color='g', fontsize = ftsze)
    else: FIG2.text(.72,.30,"Peak std. wander", color='r', fontsize = ftsze)
    
    plt.text(5, 27.5, r'$n_{win} = $' + str(NB_WIN), fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 25, r'$f_0 = $' + str(round(F0_peak,2)) + r' $Hz$', fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 22.5, r'$f_* = $' + str(round(F0_mean,2)) + r' $Hz$', fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))
    plt.text(5, 20, r'$A_0 = $' + str(round(A0,2)), fontsize = 10,
        bbox=dict(facecolor='w', edgecolor='None'))

    """SAVE"""  
    if SAVE_HV:
        if not os.path.exists(HV_FIG_PATH +  shortname_point): #if folder for saving hvsr does not exist
            os.makedirs(HV_FIG_PATH +  shortname_point) #create folder
        print(">>> Saving")
        plt.savefig(HV_FIG_PATH + shortname_point + "/" + shortname_point + "_lin.pdf", format='pdf', bbox_inches='tight')
    else:
        plt.show()

    """Save results in text file"""
    if SAVE_TXT:
        idx_POINTNAME = STA_HV.index(shortname_point)
        LON_POINT = LON_HV[idx_POINTNAME]
        LAT_POINT = LAT_HV[idx_POINTNAME]
        if PEAK_TEST:
            with open(MAP_DATAPATH + "F0_RELIABLE.txt", 'a') as fh:
                fh.write(shortname_point + "\t" + str(round(F0_peak,2)) + "\t" + str(LON_POINT) + "\t" + str(LAT_POINT) + "\n")
        else:
            if CURVE_TEST:
                with open(MAP_DATAPATH + "F0_NOT_RELIABLE.txt", 'a') as fh:
                    fh.write(shortname_point + "\t" + str(round(F0_peak,2)) + "\t" + str(LON_POINT) + "\t" + str(LAT_POINT) + "\n")
            else:
                with open(MAP_DATAPATH + "F0_DUMP.txt", 'a') as fh:
                    fh.write(shortname_point + "\t" + str(round(F0_peak,2)) + "\t" + str(LON_POINT) + "\t" + str(LAT_POINT) + "\n")