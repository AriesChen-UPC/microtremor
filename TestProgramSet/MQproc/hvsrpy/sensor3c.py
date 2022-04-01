# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Class definition for Sensor3c, a 3-component sensor."""

import math
import logging
import json
import warnings

import numpy as np
import obspy
from scipy import signal

from hvsrpy.hvsr import Hvsr
from hvsrpy.hvsrRotated import HvsrRotated
from xcorpy.coreFunc import window

logger = logging.getLogger(__name__)
__version__ = 'ManQiaoTW'
__all__ = ["Sensor3c"]


class baseTS:
    def __init__(self, amplitude, dt, t0=0):
        """Initialize a `TimeSeries` object.

        Parameters
        ----------
        amplitude : ndarray
            Amplitude of the time series at each time step.
        dt : float
            Time step between samples in seconds.
        t0: float or np.ndarray
            Zero time referrence for timestamp

        Returns
        -------
        baseTS
            Instantiated with amplitude information.

        Raises
        ------
        TypeError
            If `amplitude` is not castable to `ndarray` or has
            dimensions not equal to 1. See error message(s) for
            details.

        """
        self.amp, adim = self._check_input("amplitude", amplitude)
        self._dt = dt
        self._t0, tdim = self._check_input("t0", t0)

        if adim != tdim+1:
            raise Exception(f'Data dimension ({adim}-D) and time dimension ({tdim}-D) no match!')

        logger.info(f"Initialize a TimeSeries object.")
        logger.info(f"\tnsamples = {self.shape[-1]}")
        logger.info(f"\tdt = {self._dt}")

    @property
    def dt(self):
        return self._dt

    @property
    def time(self):
        if len(self.shape) == 1:
            return self._t0 + np.arange(self.shape[-1]) * self.dt
        else:
            return self._t0[:, np.newaxis] + np.arange(self.shape[-1]) * self.dt

    @property
    def fs(self):
        return 1/self._dt

    @property
    def fnyq(self):
        return 0.5*self.fs

    @property
    def shape(self):
        return np.shape(self.amp)

    @property
    def windowed(self):
        return len(self.shape) > 1

    @staticmethod
    def _check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Cast `values` to `ndarray`.
            2. Check that `ndarray` is 1D.

        Parameters
        ----------
        name : str
            Name of parameter to be check. Only used to raise
            easily understood exceptions.
        values : any
            Value of parameter to be checked.

        Returns
        -------
        ndarray
            `values` cast to `ndarray`.

        Raises
        ------
        TypeError
            If entries do not comply with checks 1. and 2. listed above.

        """
        try:
            values = np.array(values, dtype=np.double)
        except ValueError:
            msg = f"{name} must be convertable to numeric `ndarray`."
            raise TypeError(msg)

        if values.size == 1:
            dimension = 0
        else:
            dimension = len(values.shape)

        return values, dimension

    def detrend(self):
        """Remove linear trend from time series.

        Returns
        -------
        None
            Removes linear trend from attribute `amp`.

        """
        self.amp = self.amp - np.mean(self.amp, -1)[..., np.newaxis]

    def bandpassfilter(self, flow, fhigh, order=5):
        """Apply bandpass Butterworth filter to time series.

        Parameters
        ----------
        flow : float
            Low-cut frequency (content below `flow` is filtered).
        fhigh : float
            High-cut frequency (content above `fhigh` is filtered).
        order : int, optional
            Filter order, default is 5.

        Returns
        -------
        None
            Filters attribute `amp`.

        """
        fnyq = self.fnyq
        b, a = signal.butter(order, [flow/fnyq, fhigh/fnyq], btype='bandpass')
        # TODO (jpv): Research padlen argument
        self.amp = signal.filtfilt(b, a, self.amp, padlen=3*(max(len(b), len(a))-1))

    def cosine_taper(self, width, axis=-1):
        """Apply cosine taper to time series.

        Parameters
        ----------
        width : {0.-1.}
            Amount of the time series to be tapered.
            `0` is equal to a rectangular and `1` a Hann window.
        axis: int
            Axis to attach the taper

        Returns
        -------
        None
            Applies cosine taper to attribute `amp`.

        """
        self.amp = self.amp * signal.windows.tukey(self.shape[axis], alpha=width)


def baseFT(amplitude, dt, **kwargs):
    """Compute the fast-Fourier transform (FFT) of a time series.

    Parameters
    ----------
    amplitude : ndarray
        Denotes the time series amplitude. If `amplitude` is 1D
        each sample corresponds to a single time step. If
        `amplitude` is 2D each row corresponds to a particular
        section of the time record (i.e., time window) and each
        column corresponds to a single time step.
    dt : float
        Denotes the time step between samples in seconds.
    **kwargs : dict
        Additional keyard arguments to fft.

    Returns
    -------
    Tuple
        Of the form (fft, frq) where:

        fft : ndarray
        Complex amplitudes for the frequencies between zero
        and the Nyquist (if even) or near the Nyquist
        (if odd) with units of the input amplitude.
        If `amplitude` is a 2D array `fft` will also be a 2D
        array where each row is the FFT of each row of
        `amplitude`.

        frq : ndarray
            Positive frequency vector between zero and the
            Nyquist frequency (if even) or near the Nyquist
            (if odd) in Hz.

        """
    if len(amplitude.shape) > 2:
        raise TypeError("`amplitude` cannot have dimension > 2.")

    npts = amplitude.shape[-1] if kwargs.get(
        "n") is None else kwargs.get("n")
    nfrqs = int(npts / 2) + 1 if (npts % 2) == 0 else int((npts + 1) / 2)
    frq = np.abs(np.fft.fftfreq(npts, dt))[0:nfrqs]
    if len(amplitude.shape) == 1:
        return 2 / npts * np.fft.fft(amplitude, **kwargs)[0:nfrqs], frq
    else:
        fft = np.zeros((amplitude.shape[0], nfrqs), dtype=complex)
        for cwindow, amplitude in enumerate(amplitude):
            fft[cwindow] = 2 / npts * np.fft.fft(amplitude, **kwargs)[0:nfrqs]
        return fft, frq


def smooth_konno_ohmachi_mat(frequencies, spectrum, fcs, bandwidth=40):
    """Static method for Konno and Ohmachi smoothing.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies of the spectrum to be smoothed.
    spectrum : ndarray
        Spectrum to be smoothed, can be 1-D or 2-D, whose last axis must be the same size as frequencies.
    fcs : ndarray
        Array of center frequencies where smoothed spectrum is calculated.
    bandwidth : float, optional
        Width of smoothing window, default is 40.

    Returns
    -------
    ndarray
        Spectrum smoothed at the specified center frequencies (`fcs`).

    """
    n = 3
    upper_limit = np.power(10, +n/bandwidth)
    lower_limit = np.power(10, -n/bandwidth)

    smoothed_spectrum = np.zeros(list(spectrum.shape[:-1]) + [len(fcs)])

    for f_index, fc in enumerate(fcs):
        if fc < 1E-6:
            smoothed_spectrum[..., f_index] = 0
            continue

        f_on_fc = frequencies / fc
        fwindow = np.zeros_like(frequencies)
        sel = (frequencies >= 1E-6) * (f_on_fc <= upper_limit) * (f_on_fc >= lower_limit) * (frequencies != fc)
        fwindow[sel] = (np.sin(bandwidth * np.log10(f_on_fc[sel])) / (bandwidth * np.log10(f_on_fc[sel]))) ** 4
        fwindow[abs(frequencies - fc) < 1e-6] = 1

        sumproduct = (spectrum * fwindow).sum(-1)
        sumwindow = fwindow.sum(-1)

        if sumwindow > 0:
            smoothed_spectrum[..., f_index] = sumproduct / sumwindow
        else:
            smoothed_spectrum[..., f_index] = 0

    return smoothed_spectrum


class Sensor3c:
    """Class for creating and manipulating 3-component sensor objects.

    Attributes
    ----------
    ns : baseTS
        North-south component, time domain.
    ew : baseTS
        East-west component, time domain.
    vt : baseTS
        Vertical component, time domain.

    """

    @staticmethod
    def _check_input(values_dict):
        """Perform checks on inputs.

        Specifically:
        1. Ensure all components are `baseTS` objects.
        2. Ensure all components have equal `dt`.
        3. Ensure all components have same `nsamples`. If not trim
        components to the common length.

        Parameters
        ----------
        values_dict : dict
            Key is human readable component name {'ns', 'ew', 'vt'}.
            Value is corresponding `baseTS` object.

        Returns
        -------
        Tuple
            Containing checked components.

        """
        ns = values_dict["ns"]
        if not isinstance(ns, baseTS):
            msg = f"'ns' must be a `TimeSeries`, not {type(ns)}."
            raise TypeError(msg)
        dt = ns.dt
        nsamples = ns.shape[-1]
        flag_cut = False
        for key, value in values_dict.items():
            if key == "ns":
                continue
            if not isinstance(value, baseTS):
                msg = f"`{key}`` must be a `TimeSeries`, not {type(value)}."
                raise TypeError(msg)
            if value.dt != dt:
                msg = "All components must have equal `dt`."
                raise ValueError(msg)
            if value.shape[-1] != nsamples:
                logging.info("Components are not of the same length.")
                flag_cut = True

        if flag_cut:
            min_time = 0
            max_time = np.inf
            for value in values_dict.values():
                min_time = max(min_time, min(value.time))
                max_time = min(max_time, max(value.time))
            logging.info(f"Trimming between {min_time} and {max_time}.")
            for value in values_dict.values():
                value.trim(min_time, max_time)

        return values_dict["ns"], values_dict["ew"], values_dict["vt"]

    def __init__(self, ns, ew, vt, meta=None):
        """Initialize a 3-component sensor (Sensor3c) object.

        Parameters
        ----------
        ns, ew, vt : baseTS
            `TimeSeries` object for each component.
        meta : dict, optional
            Meta information for object, default is None.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.

        """
        self.ns, self.ew, self.vt = self._check_input({"ns": ns,
                                                       "ew": ew,
                                                       "vt": vt})
        self.meta = {} if meta is None else meta

    @property
    def normalization_factor(self):
        """Time history normalization factor across all components."""
        factor = 1E-6
        for attr in ["ns", "ew", "vt"]:
            cmax = np.max(np.abs(getattr(self, attr).amp.flatten()))
            factor = cmax if cmax > factor else factor
        return factor

    @classmethod
    def from_stream(cls, fname, traces):
        if len(traces) != 3:
            msg = f"target {fname} has {len(traces)} traces, but should have 3."
            raise ValueError(msg)
        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel.endswith("E") or trace.meta.channel.endswith("X") and not found_ew:
                ew = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ew = True
            elif trace.meta.channel.endswith("N") or trace.meta.channel.endswith("Y") and not found_ns:
                ns = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ns = True
            elif trace.meta.channel.endswith("Z") and not found_vt:
                vt = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components. See documentation."
                raise ValueError(msg)

        meta = {"File Name": fname}
        return cls(ns, ew, vt, meta)

    @classmethod
    def from_sac(cls, fname=None, fnames_1c=None, time_range=None):
        if fnames_1c is None:
            msg = "`fnames_1c` cannot be `None`."
            raise ValueError(msg)
        if fnames_1c is not None:
            trace_list = []
            for key in range(3):
                stream = obspy.read(fnames_1c[key], format="SAC")
                if len(stream) > 1:
                    msg = f"File {fnames_1c[key]} contained {len(stream)}"
                    msg += "traces, rather than 1 as was expected."
                    raise IndexError(msg)
                trace = stream[0]
                trace.meta.sampling_rate = round(trace.meta.sampling_rate)
                trace.meta.channel = fname.split(', ')[key].split('.')[-2]
                trace_list.append(trace)
            traces = obspy.Stream(trace_list)
        if len(traces) != 3:
            msg = f"miniseed file {fname} has {len(traces)} traces, but should have 3."
            raise ValueError(msg)
        if time_range is not None:
            t0 = trace.meta.starttime
            t1 = t0 + time_range[0]
            t2 = min(trace.meta.endtime, t0 + time_range[1])
            traces.trim(t1, t2)
        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel.endswith("E") or trace.meta.channel.endswith("X") and not found_ew:
                ew = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ew = True
            elif trace.meta.channel.endswith("N") or trace.meta.channel.endswith("Y") and not found_ns:
                ns = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ns = True
            elif trace.meta.channel.endswith("Z") and not found_vt:
                vt = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components. See documentation."
                raise ValueError(msg)

        meta = {"File Name": fname}
        return cls(ns, ew, vt, meta)

    @classmethod
    def from_mseed(cls, fname=None, fnames_1c=None):
        """Create 3-component sensor (Sensor3c) object from .mseed file.

        Parameters
        ----------
        fname : str, optional
            Name of miniseed file, full path may be used if desired.
            The file should contain three traces with the
            appropriate channel names. Refer to the `SEED` Manual
            `here <https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf>`_.
            for specifics, default is `None`.
        fnames_1c : dict, optional
            Some data acquisition systems supply three separate miniSEED
            files rather than a single combined file. To use those types
            of files, simply specify the three files in a `dict` of
            the form `{'e':'east.mseed', 'n':'north.mseed',
            'z':'vertical.mseed'}`, default is `None`.

        Returns
        -------
        Sensor3c
            Initialized 3-component sensor object.

        Raises
        ------
        ValueError
            If both `fname` and `fname_verbose` are `None`.

        """
        if fnames_1c is None and fname is None:
            msg = "`fnames_1c` and `fname` cannot both be `None`."
            raise ValueError(msg)
        if fnames_1c is not None:
            trace_list = []
            for key in ["e", "n", "z"]:
                stream = obspy.read(fnames_1c[key], format="MSEED")
                if len(stream) > 1:
                    msg = f"File {fnames_1c[key]} contained {len(stream)}"
                    msg += "traces, rather than 1 as was expected."
                    raise IndexError(msg)
                trace = stream[0]
                if trace.meta.channel[-1] != key.capitalize():
                    msg = "Component indicated in the header of "
                    msg += f"{fnames_1c[key]} is {trace.meta.channel[-1]} "
                    msg += f"which does not match the key {key} specified. "
                    msg += "Ignore this warning only if you know "
                    msg += "your digitizer's header is incorrect."
                    warnings.warn(msg)
                    trace.meta.channel = trace.meta.channel[:-1] + \
                                         key.capitalize()
                trace_list.append(trace)
            traces = obspy.Stream(trace_list)
        else:
            traces = obspy.read(fname, format="MSEED")

        if len(traces) != 3:
            msg = f"miniseed file {fname} has {len(traces)} traces, but should have 3."
            raise ValueError(msg)

        found_ew, found_ns, found_vt = False, False, False
        for trace in traces:
            if trace.meta.channel.endswith("E") and not found_ew:
                ew = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ew = True
            elif trace.meta.channel.endswith("N") and not found_ns:
                ns = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_ns = True
            elif trace.meta.channel.endswith("Z") and not found_vt:
                vt = baseTS(trace.data, trace.stats.delta, trace.stats.starttime.timestamp)
                found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components. See documentation."
                raise ValueError(msg)

        meta = {"File Name": fname}
        return cls(ns, ew, vt, meta)

    def to_dict(self):
        """Dictionary representation of `Sensor3c` object.

        Returns
        -------
        dict
            With all of the components of the `Sensor3c`.

        """
        dictionary = {}
        for name in ["ns", "ew", "vt"]:
            value = getattr(self, name).to_dict()
            dictionary[name] = value
        dictionary["meta"] = self.meta
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        """Create `Sensor3c` object from dictionary representation.

        Parameters
        ---------
        dictionary : dict
            Must contain keys "ns", "ew", "vt", and may also contain
            the optional key "meta". "ns", "ew", and "vt" must be
            dictionary representations of `TimeSeries` objects, see
            `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
            documentation for details.

        Returns
        -------
        Sensor3c
            Instantiated `Sensor3c` object.

        """
        components = []
        for comp in ["ns", "ew", "vt"]:
            components.append(baseTS(dictionary[comp]["amplitude"], dictionary[comp]["dt"]))
        return cls(*components, meta=dictionary.get("meta"))

    def to_json(self):
        """Json string representation of `Sensor3c` object.

        Returns
        -------
        str
            With all of the components of the `Sensor3c`.

        """
        dictionary = self.to_dict()
        return json.dumps(dictionary)

    @classmethod
    def from_json(cls, json_str):
        """Create `Sensor3c` object from Json-string representation.

        Parameters
        ---------
        json_str : str
            Json-style string, which must contain keys "ns", "ew", and
            "vt", and may also contain the optional key "meta". "ns",
            "ew", and "vt" must be Json-style string representations of
            `TimeSeries` objects, see
            `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
            documentation for details.

        Returns
        -------
        Sensor3c
            Instantiated `Sensor3c` object.

        """
        dictionary = json.loads(json_str)
        return cls.from_dict(dictionary)

    def window(self, windowlength, overlap, atrig, atrig_range):
        """Split component `TimeSeries` into `WindowedTimeSeries`.
        Tianqi Wang: This is THE window fucntion!!!

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        assert len(atrig) == 3
        comp = ["ew", "ns", "vt"]
        for i in range(3):
            assert ~getattr(self, comp[i]).windowed
            wamp = window(getattr(self, comp[i]).amp, getattr(self, comp[i]).dt,
                          windowlength, windowlength * 2, overlap, atrig[i], atrig_range[0], atrig_range[1])
            wt = window(getattr(self, comp[i]).time, getattr(self, comp[i]).dt,
                        windowlength, windowlength * 2, overlap, atrig[i], atrig_range[0], atrig_range[1])
            wtseries = baseTS(wamp, getattr(self, comp[i]).dt, wt[:, 0])
            setattr(self, comp[i], wtseries)

    def detrend(self):
        """Detrend components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.detrend()

    def bandpassfilter(self, flow, fhigh, order):  # pragma: no cover
        """Bandpassfilter components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.bandpassfilter(flow, fhigh, order)

    def cosine_taper(self, width):
        """Cosine taper components.

        Refer to
        `SigProPy <https://sigpropy.readthedocs.io/en/latest/?badge=latest>`_
        documentation for details.

        """
        for comp in [self.ew, self.ns, self.vt]:
            comp.cosine_taper(width)

    def transform(self, **kwargs):
        """Perform Fourier transform on components.

        Returns
        -------
        dict
            With `FourierTransform`-like objects, one for for each
            component, indicated by the key 'ew','ns', 'vt'.

        """
        ffts = {}
        for attr in ["ew", "ns", "vt"]:
            tseries = getattr(self, attr)
            if isinstance(tseries, baseTS):
                fft = baseFT(tseries.amp, tseries.dt, **kwargs)
            else:
                raise NotImplementedError
            ffts[attr] = fft
        return ffts

    def combine_horizontals(self, method, horizontals, azimuth=None):
        """Combine two horizontal components (`ns` and `ew`).

        Parameters
        ----------
        method : {'squared-average', 'geometric-mean', 'single-azimuth', 'multiple-azimuths'}
            Defines how the two horizontal components are combined
            to represent a single horizontal component.
        horizontals : dict
            If combination is done in the frequency-domain (i.e.,
            `method in ['squared-average', 'geometric-mean']`)
            horizontals is a `dict` of `FourierTransform` objects,
            see :meth:`transform <Sensor3c.transform>` for details. If
            combination is done in the time-domain
            (i.e., `method in ['single-azimuth', 'multiple-azimuths']`)
            horizontals is a `dict` of `TimeSeries` objects.
        azimuth : float, optional
            Valid only if `method` is `single-azimuth` in which case an
            azimuth (clockwise positive) from North (i.e., 0 degrees) is
            required.

        Returns
        -------
        TimeSeries or FourierTransform
            Depending upon the specified `method` requires the
            combination to happen in the time or frequency domain.

        """
        if method in ["squared-average", "geometric-mean"]:
            return self._combine_horizontal_fd(method, horizontals)
        elif method in ["azimuth", "single-azimuth"]:
            return self._combine_horizontal_td(method, horizontals, azimuth=azimuth)
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

    @staticmethod
    def _combine_horizontal_fd(method, horizontals, **kwargs):
        ns = horizontals["ns"]
        ew = horizontals["ew"]

        if method == "squared-average":
            horizontal = np.sqrt((abs(ns[0])**2 + abs(ew[0])**2) / 2)
        elif method == "geometric-mean":
            horizontal = np.sqrt(abs(ns[0]) * abs(ew[0]))
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

        return horizontal, ns[1]

    def _combine_horizontal_td(self, method, horizontals, azimuth):
        az_rad = math.radians(azimuth)
        ns = horizontals["ns"]
        ew = horizontals["ew"]

        if method in ["azimuth", "single-azimuth"]:
            horizontal = ns.amp * math.cos(az_rad) + ew.amp * math.sin(az_rad)
        else:
            msg = f"method={method} has not been implemented."
            raise NotImplementedError(msg)

        if isinstance(ns, baseTS):
            return baseTS(horizontal, ns.dt, ns.time[:, 0])
        else:
            raise NotImplementedError

    def hv(self, bp_filter, taper_width, bandwidth,
           resampling, method, azimuth=None):
        """Prepare time series and Fourier transforms then compute H/V.

        Parameters
        ----------
        bp_filter : dict
            Bandpass filter settings, of the form
            `{'flag':bool, 'flow':float, 'fhigh':float, 'order':int}`.
        taper_width : float
            Width of cosine taper, value between `0.` and `1.`.
        bandwidth : float
            Bandwidth (b) of the Konno and Ohmachi (1998) smoothing
            window.
        resampling : dict
            Resampling settings, of the form
            `{'minf':float, 'maxf':float, 'nf':int, 'res_type':str}`.
        method : {'squared-average', 'geometric-mean', 'single-azimuth', 'multiple-azimuths'}
            Refer to :meth:`combine_horizontals <Sensor3c.combine_horizontals>`
            for details.
        azimuth : float, optional
            Refer to
            :meth:`combine_horizontals <Sensor3c.combine_horizontals>`
            for details.

        Returns
        -------
        Hvsr
            Instantiated `Hvsr` object.

        """
        if bp_filter["flag"]:
            self.bandpassfilter(flow=bp_filter["flow"],
                                fhigh=bp_filter["fhigh"],
                                order=bp_filter["order"])
        self.detrend()
        self.cosine_taper(width=taper_width)

        if method in ["squared-average", "geometric-mean", "azimuth", "single-azimuth"]:
            if method == "azimuth":
                msg = "method='azimuth' is deprecated, replace with the more descriptive 'single-azimuth'."
                warnings.warn(msg, DeprecationWarning)
                method = "single-azimuth"
            return self._make_hvsr(method=method, resampling=resampling,
                                   bandwidth=bandwidth, azimuth=azimuth)

        elif method in ["rotate", "multiple-azimuths"]:
            if method == "rotate":
                msg = "method='rotate' is deprecated, replace with the more descriptive 'multiple-azimuths'."
                warnings.warn(msg, DeprecationWarning)
                method = "multiple-azimuths"
            hvsrs = np.empty(len(azimuth), dtype=object)
            for index, az in enumerate(azimuth):
                hvsrs[index] = self._make_hvsr(method="single-azimuth",
                                               resampling=resampling,
                                               bandwidth=bandwidth,
                                               azimuth=az)
            return HvsrRotated.from_iter(hvsrs, azimuth, meta=self.meta)

        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

    def _make_hvsr(self, method, resampling, bandwidth, azimuth=None):
        if method in ["squared-average", "geometric-mean"]:
            ffts = self.transform()
            hor = self.combine_horizontals(method=method, horizontals=ffts)
            ver = ffts["vt"]
            del ffts
        elif method == "single-azimuth":
            hor = self.combine_horizontals(method=method,
                                           horizontals={"ew": self.ew,
                                                        "ns": self.ns},
                                           azimuth=azimuth)
            hor = baseFT(hor.amp, hor.dt)
            ver = baseFT(self.vt.amp, self.vt.dt)
        else:
            msg = f"`method`={method} has not been implemented."
            raise NotImplementedError(msg)

        self.meta["method"] = method
        self.meta["azimuth"] = azimuth

        if resampling["res_type"] == "linear":
            frq = np.linspace(resampling["minf"],
                              resampling["maxf"],
                              resampling["nf"])
        elif resampling["res_type"] == "log":
            frq = np.geomspace(resampling["minf"],
                               resampling["maxf"],
                               resampling["nf"])
        else:
            raise NotImplementedError

        hvsr = smooth_konno_ohmachi_mat(hor[1], abs(hor[0]), frq, bandwidth) /\
               smooth_konno_ohmachi_mat(ver[1], abs(ver[0]), frq, bandwidth)
        hvsr[np.isnan(hvsr)] = 0
        del hor, ver

        self.meta["Window Length"] = self.ns.shape[-1]*self.ns.dt

        return Hvsr(hvsr, frq, find_peaks=False, meta=self.meta)

    def __iter__(self):
        """Iterable representation of a Sensor3c object."""
        return iter((self.ns, self.ew, self.vt))

    def __str__(self):
        """Human-readable representation of `Sensor3c` object."""
        return "Sensor3c"

    def __repr__(self):
        """Unambiguous representation of `Sensor3c` object."""
        return f"Sensor3c(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"
