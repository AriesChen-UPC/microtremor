#################### README.txt ####################
This set of Python scripts are designed to process and post-process ambient vibration
measurements with H/V technique. Detailed workflow is given hereunder.

They must be used in combination with Geopsy software (www.geopsy.org; Marc WATHELET, ISTerre).

They allow fast SPECTRUM and HVSR curves computation.
HVSR curves and peak reliability is checked with reference to SESAME guidelines
(see see Bard and the SESAME Team (2004) Guidelines for the implementation of the 
H/V spectral ratio technique of ambient vibrations, SESAME European research
project WP12 � Deliverable D23.12

If a .kml file of points coordinates is available (from GoogleEarth for example, 
HVSR results are quickly displayed on a map).

Reference for this work: if you use those scripts, please acknowledge this work as:
Bottelin, P., (2015) HVResPy: an open-source Python tool for Geopsy HVSR post-processing.

Do not hesitate to contact me for debugging/questions/improvements/collaboration/...

Contact: Pierre BOTTELIN
Post-doctoral teacher and researcher
Applied geophysics, geotechnical engineering and environment
IRAP / OMP Toulouse
pierre.bottelin@irap.omp.eu
or
Pierre BOTTELIN on ResearchGate

#################### WORKFLOW####################

###1.Install pre-requisites softwares
1.1- Install geopsy: visit www.geopsy.org website and follow indications
1.2- Install a linux-like terminal: For Windows users: visit www.geopsy.org website forum
and follow Marc WATHELET's indications for MSYS setup for windows. You can either install
MSYS directly from its website. 
For linux users, use the usual terminal window.
1.3- Install Python distribution. Visit any website distributing a 2.XX Python, and install it
following the instructions.
I use an Anaconda distribution on Windows machine (visit website).

###2.Go to Python scripts
2.1- Those Python scripts and corresponding folders can be put anywhere on the machine.
The pathes are relative to ./SCRIPTS/ folder.
2.2- Open a terminal (linux) or a command window (windows: launch, type "cmd" + enter)
2.3- Go to the ./SCRIPTS/ folder. You can move in the folder with "cd FOLDER_NAME".
Move out a folder with "cd ..". List a folder content with "ls" (linux) or "dir" (windows).

###3.Execute Python scripts
3.1- Once in the ./SCRIPTS/ folder, type "python PYTHON_SCRIPT_NAME.py" in the terminal.
This command makes python executes "PYTHON_SCRIPT_NAME.py" script.

###4.Step by step processing:
4.1- Type "python a_process_data.py". This executes "a_process_data.py" script. You can read this
script with any text editor, like notepad or others. I use Notepad++.
This first script recursively reads the seismic files in the "./DATA/SEISMIC_DATA/" folder.
It uses geopsy to compute the spectra and the HVSR with parameters contained in "./SCRIPTS/param_process.log".
The output files are placed in corresponding "./DATA/SPECTRUM_DATA/" and "./DATA/HV_DATA/".

4.2- Type "python b_plot_spectrum.py". This executes "b_plot_spectrum.py" script.
This script uses the files in "./DATA/SPECTRUM_DATA/" to plot the spectrum of each channel
for each measurement point on a .pdf figure. The figures are place in "./FIGURES/SPECTRUM_FIGURES/" folder.

4.3- Type "python c_plot_hvsr.py". This executes "c_plot_hvsr.py" script.
This script uses the files in "./DATA/HV_DATA/" to plot the spectrum of each channel
for each measurement point on a .pdf figure. The figures are placed in "./FIGURES/HV_FIGURES/" folder.

Peaks in HVSR curves are detected and picked automatically. The frequency range for peak detection
has to be defined !! This step has to be carefully carried out !! Expert advice required.

The particularity of this script is that it tests the reliability of HVSR curve and peak with respect
to SESAME guidelines (see Bard and SESAME Team (2004) Guidelines for the implementation of the 
H/V spectral ratio technique of ambient vibrations, SESAME European research
project WP12 � Deliverable D23.12).

For each test, the test name is diplayed in green (test succeeded) or red (test failed) on the right side of
the figure.
The CURVE reliability and PEAK reliability are derived from individual tests. They follow the same color code.

Results can be saved into 3 separated text files: reliable points, not reliable points, dump points.

Coordinates: This script can automatically use a .kml file (created with GoogleEarth for example) to geo-localize the HVSR
measurements. The point names must be identical in the .kml and in the "./DATA/SEISMIC_DATA/" folders.

4.4- Type "python d_plot_map.py". This executes "d_plot_map.py" script. This script uses the frequency (F0)
of HVSR peaks saved into the 3 separated text files (reliable points, not reliable points, dump point).
A map is drawn, with F0 as color code. The right-side figure shows preliminary interpretation of F0 peaks
as 1-D layer resonance (see literature). !! This interpretation has to be carefully used !! Expert advice required.

Resulting map is saved in "./FIGURES/MAP_FIGURES/" folder.
