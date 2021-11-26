# Version 0: all files generated with releases before 20170901 (default for input)
PARAMETERS_VERSION = 1
# TYPEs:
#   - Signal: from the start or to the end of signal (TEXT are useless)
#   - Delta: a fixed duration counted from the start or the end (e.g. TEXT=1h).
#   - Pick: from or to a time pick (TEXT=time pick name).
#   - Absolute: from or to a fixed time (e.g. TEXT=3d5h6m45s)
FROM_TIME_TYPE = 'Signal'
FROM_TIME_TEXT = '0s'
# TYPEs: Signal, Delta, Absolute
TO_TIME_TYPE = 'Signal'
TO_TIME_TEXT = '0s'
REFERENCE = ''
# TYPEs: Exactly, AtLeast, FrequencyDependent
WINDOW_LENGTH_TYPE = 'FrequencyDependent'
PERIOD_COUNT = 70
WINDOW_MAX_COUNT = 0
WINDOW_POWER_OF_TWO = 'n'
BAD_SAMPLE_TOLERANCE = 0
BAD_SAMPLE_GAP = 0
WINDOW_OVERLAP = 0
# TYPEs: NoSampleThreshold, RelativeSampleThreshold, AbsoluteSampleThreshold
BAD_SAMPLE_THRESHOLD_TYPE = 'NoSampleThreshold'
ANTI_TRIGGERING_ON_RAW_SIGNAL = 'y'
ANTI_TRIGGERING_ON_FILTERED_SIGNAL = 'n'
RAW_STA = 1.0
RAW_LTA = 30.0
RAW_MIN_SLTA = 0.4
RAW_MAX_SLTA = 2.5
# Start a time window for each seismic event available inside the time range.
SEISMIC_EVENT_TRIGGER = 'n'
# SEISMIC_EVENT_TDELAY (s)=-0.1
MINIMUM_FREQUENCY = 0.1
MAXIMUM_FREQUENCY = 100.1
# Either 'log' or 'linear'
SAMPLING_TYPE_FREQUENCY = 'Log'
# Number of samples is either set to a fixed value ('Count') or through a step between samples ('Step')'
STEP_TYPE_FREQUENCY = 'Count'
SAMPLES_NUMBER_FREQUENCY = 400
# STEP=ratio between two successive samples for 'log' scales
# STEP=difference between two successive samples for 'linear' scales
STEP_FREQUENCY = 1.025 
INVERSED_FREQUENCY = 'n'
# Overlap is controled by the WINDOWS parameters, by default non overlapping blocks are selected
BLOCK_OVERLAP = 'n'
# If BLOCK_COUNT is null, BLOCK_COUNT=BLOCK_COUNT_FACTOR*<number of stations>
BLOCK_COUNT = 0
BLOCK_COUNT_FACTOR = 2
# If STATISTIC_COUNT is not null, approx. STATISTIC_COUNT estimates par frequency
STATISTIC_COUNT = 50
# If STATISTIC_MAX_OVERLAP=100%, successive statistics can be computed on overlapping block sets
# If STATISTIC_MAX_OVERLAP=0%, successive statistics are computed on non-overlapping block sets
STATISTIC_MAX_OVERLAP = 0
# Gaussian band width from f*(1-bw) to f*(1+bw), f*bw=stddev
FREQ_BAND_WIDTH = 0.1
# Required when using short and fixed length time windows, avoid classical oblique lines visible in the results
# when the number of frequency samples is higher than the number of points in the spectra.
OVER_SAMPLING_FACTOR = 1
# A station is selected for processing only if it is available over a duration greater or equal to
# SELECT_DURATION_FACTOR*[total required duration]. The factor can vary from 0 to 1
SELECT_DURATION_FACTOR = 0
# A station is selected for processing only if it is located at less than SELECT_ARRAY_RADIUS
# from SELECT_ARRAY_CENTER. SELECT_ARRAY_CENTER is the X, Y coordinates of the center.
SELECT_ARRAY_CENTER = 0
SELECT_ARRAY_RADIUS = 0
# Assuming that north of sensors is aligned to the magnetic north and sensor coordinates to UTM grid,
# relative coordinates between stations are calculated with a correction for the difference between the
# geographical and the local UTM norths and for the magnetic declination. The later can be, for instance,
# calculated at https://www.ngdc.noaa.gov/geomag-web/#declination
# The value must be in degrees.
MAGNETIC_DECLINATION = 0
OUTPUT_BASE_NAME = ''
# List of rings (distances in m): min_radius_1 max_radius_1 min_radius_2 max_radius_2 ...
RINGS = [3.99, 4.01, 4.7, 4.71, 7.6, 7.61]
# Reject autocorr estimates with an imaginary part above this maximum
MAXIMUM_IMAGINARY_PART = 1
