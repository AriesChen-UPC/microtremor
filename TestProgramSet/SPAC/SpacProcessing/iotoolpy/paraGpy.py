import numpy as np

from xcorpy.coreFunc import update_logs


class parameters:
    """Geopsy parameters
    This is the parameter holder for Geopsy command-line calculations.
    It generates a LOG file needed by Geopsy program.

    Attributes
    ----------
    COMMON
        Parameters commonly used in all calculations
    SPAC
        Parameters used in SPAC method
    FK
        Parameters used in FK method
    HV
        Parameters used in HV method
    """
    def __init__(self):
        self.COMMON = self.common()
        self.SPAC = self.spac()
        self.FK = self.fk()
        self.HV = self.hv()

    class common:
        def __init__(self):
            # Version 0: all files generated with releases before 20170901 (default for input)
            self.PARAMETERS_VERSION = 1
            # TYPEs:
            #   - Signal: from the start or to the end of signal (TEXT are useless)
            #   - Delta: a fixed duration counted from the start or the end (e.g. TEXT=1h).
            #   - Pick: from or to a time pick (TEXT=time pick name).
            #   - Absolute: from or to a fixed time (e.g. TEXT=20170314115338.00)
            self.FROM_TIME_TYPE = 'Signal'
            self.FROM_TIME_TEXT = '0s'
            # TYPEs: Signal, Delta, Absolute
            self.TO_TIME_TYPE = 'Signal'
            self.TO_TIME_TEXT = '0s'
            self.REFERENCE = ''
            # TYPEs: Exactly, AtLeast, FrequencyDependent
            self.WINDOW_LENGTH_TYPE = 'FrequencyDependent'
            self.PERIOD_COUNT = 70  # TYPE = FrequencyDependent
            self.WINDOW_MIN_LENGTH = 30  # (s) TYPE = Exactly/AtLeast
            self.WINDOW_MAX_LENGTH = 30  # (s) TYPE = Exactly/AtLeast
            self.WINDOW_MAX_COUNT = 0
            self.WINDOW_POWER_OF_TWO = 'n'
            self.BAD_SAMPLE_TOLERANCE = 0
            self.BAD_SAMPLE_GAP = 0
            self.WINDOW_OVERLAP = 0
            # TYPEs: NoSampleThreshold, RelativeSampleThreshold, AbsoluteSampleThreshold
            self.BAD_SAMPLE_THRESHOLD_TYPE = 'NoSampleThreshold'
            self.ANTI_TRIGGERING_ON_RAW_SIGNAL = 'y'
            # cannot input 2 ANTI-TRIGGERING at the same time
            # ANTI_TRIGGERING_ON_FILTERED_SIGNAL = 'n'
            self.RAW_STA = 1.0
            self.RAW_LTA = 30.0
            self.RAW_MIN_SLTA = 0.2
            self.RAW_MAX_SLTA = 2.5
            # Start a time window for each seismic event available inside the time range.
            self.SEISMIC_EVENT_TRIGGER = 'n'
            # SEISMIC_EVENT_TDELAY (s)=-0.1
            self.MINIMUM_FREQUENCY = 0.1
            self.MAXIMUM_FREQUENCY = 100.1
            # Either 'log' or 'linear'
            self.SAMPLING_TYPE_FREQUENCY = 'Log'
            # Number of samples is either set to a fixed value ('Count') or through a step between samples ('Step')'
            self.STEP_TYPE_FREQUENCY = 'Count'
            self.SAMPLES_NUMBER_FREQUENCY = 400
            # STEP=ratio between two successive samples for 'log' scales
            # STEP=difference between two successive samples for 'linear' scales
            # cannot input STEP when COUNT is selected
            # STEP_FREQUENCY = 1.0
            self.INVERSED_FREQUENCY = 'n'

    class spac:
        def __init__(self):
            # Overlap is controled by the WINDOWS parameters, by default non overlapping blocks are selected
            self.BLOCK_OVERLAP = 'n'
            # If BLOCK_COUNT is null, BLOCK_COUNT=BLOCK_COUNT_FACTOR*<number of stations>
            self.BLOCK_COUNT = 1
            self.BLOCK_COUNT_FACTOR = 0
            # If STATISTIC_COUNT is not null, approx. STATISTIC_COUNT estimates par frequency
            self.STATISTIC_COUNT = 50
            # If STATISTIC_MAX_OVERLAP=100%, successive statistics can be computed on overlapping block sets
            # If STATISTIC_MAX_OVERLAP=0%, successive statistics are computed on non-overlapping block sets
            self.STATISTIC_MAX_OVERLAP = 0
            # Gaussian band width from f*(1-bw) to f*(1+bw), f*bw=stddev
            self.FREQ_BAND_WIDTH = 0.1
            # Required when using short and fixed length time windows, avoid classical oblique lines visible in the results
            # when the number of frequency samples is higher than the number of points in the spectra.
            self.OVER_SAMPLING_FACTOR = 1
            # A station is selected for processing only if it is available over a duration greater or equal to
            # SELECT_DURATION_FACTOR*[total required duration]. The factor can vary from 0 to 1
            self.SELECT_DURATION_FACTOR = 0
            # A station is selected for processing only if it is located at less than SELECT_ARRAY_RADIUS
            # from SELECT_ARRAY_CENTER. SELECT_ARRAY_CENTER is the X, Y coordinates of the center.
            self.SELECT_ARRAY_CENTER = [0, 0]
            self.SELECT_ARRAY_RADIUS = 0
            # Assuming that north of sensors is aligned to the magnetic north and sensor coordinates to UTM grid,
            # relative coordinates between stations are calculated with a correction for the difference between the
            # geographical and the local UTM norths and for the magnetic declination. The later can be, for instance,
            # calculated at https://www.ngdc.noaa.gov/geomag-web/#declination
            # The value must be in degrees.
            self.MAGNETIC_DECLINATION = 0
            self.OUTPUT_BASE_NAME = ''
            # List of rings (distances in m): min_radius_1 max_radius_1 min_radius_2 max_radius_2 ...
            self.RINGS = [3.99, 4.01, 4.7, 4.71, 7.6, 7.61]
            # Reject autocorr estimates with an imaginary part above this maximum
            self.MAXIMUM_IMAGINARY_PART = 1

    class fk:
        def __init__(self):
            # Overlap is controled by the WINDOWS parameters, by default non overlapping blocks are selected
            self.BLOCK_OVERLAP = 'n'
            # If BLOCK_COUNT is null, BLOCK_COUNT=BLOCK_COUNT_FACTOR*<number of stations>
            self.BLOCK_COUNT = 1
            self.BLOCK_COUNT_FACTOR = 0
            # If STATISTIC_COUNT is not null, approx. STATISTIC_COUNT estimates par frequency
            self.STATISTIC_COUNT = 50
            # If STATISTIC_MAX_OVERLAP=100%, successive statistics can be computed on overlapping block sets
            # If STATISTIC_MAX_OVERLAP=0%, successive statistics are computed on non-overlapping block sets
            self.STATISTIC_MAX_OVERLAP = 0
            # Gaussian band width from f*(1-bw) to f*(1+bw), f*bw=stddev
            self.FREQ_BAND_WIDTH = 0.1
            # Required when using short and fixed length time windows, avoid classical oblique lines visible in the results
            # when the number of frequency samples is higher than the number of points in the spectra.
            self.OVER_SAMPLING_FACTOR = 1
            # A station is selected for processing only if it is available over a duration greater or equal to
            # SELECT_DURATION_FACTOR*[total required duration]. The factor can vary from 0 to 1
            self.SELECT_DURATION_FACTOR = 0
            # A station is selected for processing only if it is located at less than SELECT_ARRAY_RADIUS
            # from SELECT_ARRAY_CENTER. SELECT_ARRAY_CENTER is the X, Y coordinates of the center.
            self.SELECT_ARRAY_CENTER = [0, 0]
            self.SELECT_ARRAY_RADIUS = 0
            # Assuming that north of sensors is aligned to the magnetic north and sensor coordinates to UTM grid,
            # relative coordinates between stations are calculated with a correction for the difference between the
            # geographical and the local UTM norths and for the magnetic declination. The later can be, for instance,
            # calculated at https://www.ngdc.noaa.gov/geomag-web/#declination
            # The value must be in degrees.
            self.MAGNETIC_DECLINATION = 0
            self.OUTPUT_BASE_NAME = ''
            # Process types:
            #  [All types can be used with vertical or three component datasets]
            #  Keyword                Beamformer    Comments
            #  DirectSteering         Capon         Cross spectrum made of raw components E, N and Z.
            #                                       Radial and transverse projections included in steering matrix.
            #                                       Combined optimum power.
            #  Omni                   Capon         Same cross spectrum as DirectSteering.
            #                                       Ouput power is the sum of power in all directions
            #  RTBF                   Capon         According to Wathelet et al (2018).
            #                                       Cross spectrum made of radial and transverse projections.
            #  PoggiVertical          Capon         According Poggi et al. (2010)
            #                                       k picked from vertical processing
            #  PoggiRadial            Capon         According Poggi et al. (2010)
            #                                       k picked from radial processing
            #  Conventional           Conventional  Conventional FK processing
            #                                       Cross spectrum made of radial and transverse projections.
            #  ActiveRTBF             Capon         High resolution for active source
            #                                       Cross spectrum made of radial and transverse projections.
            #  ActiveDirectSteering   Capon         Cross spectrum made of raw components E, N and Z.
            #                                       Radial and transverse projections included in steering matrix.
            #  ActiveConventional     Conventional  Conventional FK processing
            #                                       Cross spectrum made of radial and transverse projections.
            #  Experimental modes:
            #  DirectSteeringVertical Capon         Cross spectrum made of raw components E, N and Z.
            #                                       Radial and transverse projections included in steering matrix.
            #                                       Radial ellipticity steering.
            #  DirectSteeringRadial   Capon         Cross spectrum made of raw components E, N and Z.
            #                                       Radial and transverse projections included in steering matrix.
            #                                       Vertical ellipticity steering.
            #  DirectSteeringRefined  Capon         Cross spectrum made of raw components E, N and Z.
            #                                       Radial and transverse projections included in steering matrix.
            #                                       Iterative ellitpticity assessment.
            self.PROCESS_TYPE = 'DirectSteering'
            # For debug purpose, save a bit of time by skipping Love computation
            self.SKIP_LOVE = 'n'  # (y / n)
            # Inversion method used for getting FK peaks: Gradient or RefinedGrid
            INVERSION_METHOD = 'RefinedGrid'
            # Wavenumber fine gridding used as a cache for the FK maps
            self.CACHE_GRID_STEP = 0  # (rad / m)
            # If CACHE_GRID_STEP is null, GRID_STEP is computed from K_MIN*CACHE_GRID_STEP_FACTOR.
            CACHE_GRID_STEP_FACTOR = 0.05
            # Wavenumber coarse gridding used for searching maxima of the FK maps
            self.GRID_STEP = 0  # (rad / m)
            # If GRID_STEP is null, GRID_STEP is computed from K_MIN*GRID_STEP_FACTOR.
            GRID_STEP_FACTOR = 0.1
            self.GRID_SIZE = 0  # (rad / m)
            # Minimum velocity of the searched maxima of the FK map
            self.MIN_V = 50  # (m / s)
            # Maximum velocity of the searched maxima of the FK map
            self.MAX_V = 3500  # (m / s)
            # Minimum azimuth of the searched maxima of the FK map (math)
            self.MIN_AZIMUTH = 0  # (deg.)
            # Maximum azimith of the searched maxima of the FK map (math)
            self.MAX_AZIMUTH = 360  # (deg.)
            # Theoretical Kmin and Kmax computed from array geometry
            # Used only for post-processing (AVIOS project)
            self.K_MIN = 0  # (rad / m)
            self.K_MAX = 0  # (rad / m)
            self.N_MAXIMA = 2147483647
            self.ABSOLUTE_THRESHOLD = 0
            self.RELATIVE_THRESHOLD = 90  # ( %)
            self.EXPORT_ALL_FK_GRIDS = 'n'
            self.DAMPING_FACTOR = 0
            # If provided and PROCESS_TYPE==DirectSteering, the ellipticity is forced to the provided curve.
            # The file must contain two columns: frequency and signed ellipticity.
            # Provided sampling must not necessarily match the processing sampling frequency, linear interpolation is used.
            # Better for precision if the two sampling match.
            # To generate a synthetic curve: gpell M2.1.model -one-mode -R 1 -min 0.5 -max 50 -n 187 > curve.txt
            self.FIXED_ELLIPTICITY_FILE_NAME = ''
            # Minimum distance between source and receiver (for active source only)
            self.MINIMUM_DISTANCE = 0
            # Maximum distance between source and receiver (for active source only)
            self.MAXIMUM_DISTANCE = np.inf
            # Experimental join processing of several arrays
            # Several ARRAY can be defined with a list of station names
            #
            self.SOURCE_GRID_STEP = 1
            self.SOURCE_GRID_SIZE = 0

    class hv:
        def __init__(self):
            self.WINDOW_TYPE = 'Tukey'
            self.WINDOW_REVERSED = 'n'
            self.WINDOW_ALPHA = 0.1
            self.SMOOTHING_METHOD = 'Function'
            self.SMOOTHING_WIDTH_TYPE = 'Log'
            self.SMOOTHING_WIDTH = 0.2
            # Describes the way values are summed: on a linear, log or inversed scale
            self.SMOOTHING_SCALE_TYPE = 'Log'
            self.SMOOTHING_WINDOW_TYPE = 'KonnoOhmachi'
            self.SMOOTHING_WINDOW_REVERSED = 'n'
            self.HIGH_PASS_FREQUENCY = 0
            # Possible values for HORIZONTAL_COMPONENTS: Squared, Energy, Azimuth
            self.HORIZONTAL_COMPONENTS = 'Squared'
            # HORIZONTAL_AZIMUTH is used only when HORIZONTAL_COMPONENTS==Azimuth
            self.HORIZONTAL_AZIMUTH = 0
            # Used only for rotated output
            self.ROTATION_STEP = 10

    def write(self, parafile, ftype='spac', verbose=True):
        keys = [item for item in self.COMMON.__dict__.keys() if not item.startswith("__")]
        # fix "WINDOW_LENGTH_TYPE" attributes
        if self.COMMON.WINDOW_LENGTH_TYPE == 'FrequencyDependent':
            keys.remove('WINDOW_MIN_LENGTH')
            keys.remove('WINDOW_MAX_LENGTH')
        elif self.COMMON.WINDOW_LENGTH_TYPE == 'Exactly' or self.COMMON.WINDOW_LENGTH_TYPE == 'AtLeast':
            keys.remove('PERIOD_COUNT')
        with open(parafile, 'w') as f:
            for key in keys:
                if type(self.COMMON.__dict__[key]) is list:
                    val = ' '.join(format(x, '.5f') for x in self.COMMON.__dict__[key])
                elif type(self.COMMON.__dict__[key]) is float:
                    val = format(self.COMMON.__dict__[key], '.5f')
                else:
                    val = str(self.COMMON.__dict__[key])
                # fix "ANTI-TRIGGERING" attribute name
                if key == 'ANTI_TRIGGERING_ON_RAW_SIGNAL':
                    f.writelines('ANTI-TRIGGERING_ON_RAW_SIGNAL={}\n'.format(val))
                elif key == 'ANTI_TRIGGERING_ON_FILTERED_SIGNAL':
                    f.writelines('ANTI-TRIGGERING_ON_FILTERED_SIGNAL={}\n'.format(val))
                else:
                    f.writelines('{}={}\n'.format(key, val))
            if ftype == 'spac':
                for key in [item for item in self.SPAC.__dict__.keys() if not item.startswith("__")]:
                    if type(self.SPAC.__dict__[key]) is list:
                        val = ' '.join(format(x, '.5f') for x in self.SPAC.__dict__[key])
                    elif type(self.SPAC.__dict__[key]) is float:
                        val = format(self.SPAC.__dict__[key], '.5f')
                    else:
                        val = str(self.SPAC.__dict__[key])
                    f.writelines('{}={}\n'.format(key, val))
            elif ftype == 'fk':
                for key in [item for item in self.FK.__dict__.keys() if not item.startswith("__")]:
                    if type(self.FK.__dict__[key]) is list:
                        val = ' '.join(format(x, '.5f') for x in self.FK.__dict__[key])
                    elif type(self.FK.__dict__[key]) is float:
                        val = format(self.FK.__dict__[key], '.5f')
                    else:
                        val = str(self.FK.__dict__[key])
                    f.writelines('{}={}\n'.format(key, val))
            elif ftype == 'hv':
                for key in [item for item in self.HV.__dict__.keys() if not item.startswith("__")]:
                    if type(self.HV.__dict__[key]) is list:
                        val = ' '.join(format(x, '.5f') for x in self.HV.__dict__[key])
                    elif type(self.HV.__dict__[key]) is float:
                        val = format(self.HV.__dict__[key], '.5f')
                    else:
                        val = str(self.HV.__dict__[key])
                    f.writelines('{}={}\n'.format(key, val))
            else:
                update_logs('ErrorGPY', 'Geopsy type undefined!')
                raise Exception('Geopsy type undefined!')
        if verbose:
            print('Wrote file: ' + parafile)
