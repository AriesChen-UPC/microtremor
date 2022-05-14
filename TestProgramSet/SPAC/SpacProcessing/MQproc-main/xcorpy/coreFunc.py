import os
import warnings
import logging
import numpy as np
from scipy import signal, interpolate
from datetime import datetime
from obspy import Stream

from xcorpy.baseConfig import cfgadd, cfgkey

from coop_fetch.Config import Config
from coop_fetch.Fetch import Fetch

# 配置FETCH
config = Config(cfgadd, cfgkey)
fetch = Fetch(config)


def update_prog(msg, prog, status, info=''):
    """Pass progress to coop_fetch server

    Parameters
    ----------
    msg: str
        The message to pass
    prog: list[float, float]
        Current/total progress
    status: int
        Status of the progress, [0] for unfinished, [1] for finished
    info: str
        Extra info to pass, default as ''
    """
    try:
        res = fetch.getObject('server.ctl.main.Center', 'putProgress',
                              [msg, prog, status, info])
        if res:
            print('{}({:.2f}%): {} [updated]'.format(msg, prog[0] / prog[1] * 100, info))
            # fetch.getObject('server.ctl.main.Center', 'addLogs',
            #                 [msg + '[success]', UTCDateTime.now().isoformat(), info])
    except Exception as e:
        print(e)  # 记录错误日志
        fetch.getObject('server.ctl.main.Center', 'addLogs',
                        [msg + '[error]', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(e)])


def update_logs(msg, ltype='log', info=''):
    """Pass a log/error/warning to coop_fetch server

    Parameters
    ----------
    msg: str
        The message of the log
    ltype: str
        ['log', 'error', 'warning']
    info: str
        Extra info to pass, default as ''
    """
    fetch.getObject('server.ctl.main.Center', 'addLogs',
                    ['{}[{}]'.format(msg, ltype), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info])


def update_init():
    """Initiating coop_fetch progressbar"""
    try:
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Initialisation', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Loading SAC', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Back-up process', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Processing', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Subprocess', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['XCOR loop', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['FK loop', [0, 1], 0, ' - '])
        fetch.getObject('server.ctl.main.Center', 'putProgress',
                        ['Single-pair XCOR', [0, 1], 0, ' - '])
    except Exception as e:
        print(e)  # 记录错误日志
        fetch.getObject('server.ctl.main.Center', 'addLogs',
                        ['init_error', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(e)])


def terminal_size():
    """Get the size of a terminal window

    Returns
    -------
    tuple[int,int]
        Pixels of terminal window width/height
    """
    try:
        tw, th = os.get_terminal_size()
    except OSError:
        tw = 80
        th = 40
    return tw, th


def print_line(t='', t0='-'):
    """Print a split-line with given message

    Parameters
    ----------
    t: str
        The message in the split-line, defualt as ''
    t0: str
        The style of the split-line, defualt as '-'
    """
    n = len(t)
    if n != 0:
        t = ' ' + t + ' '
    try:
        n0 = os.get_terminal_size().columns
    except:
        n0 = 80
    n1 = round((n0 - n) * 0.38)
    n2 = n0 - n - n1 - 2
    print('\n' + t0 * n1 + t + t0 * n2)


def next2(x):
    """returns the next power of 2

    Parameters
    ----------
    x: float
        Any positive number

    Returns
    -------
    int
        The next power of 2 of the input
    """
    return int(np.power(2, np.ceil(np.log2(x))))


def conjsq(x):
    """Calculate the conjugate multiplication (for real arrays returns the normal square)

    Parameters
    ----------
    x: np.ndarray

    Returns
    -------
    np.ndarray
    """
    return x * x.conj()


def csgp_f(x):
    """Foward Conjugate-Squared Geostatiionary Projection

    Parameters
    ----------
    x: np.ndarray

    Returns
    -------
    np.ndarray
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return x / np.sqrt(1 - conjsq(x))


def csgp_b(x):
    """Backward Conjugate-Squared Geostatiionary Projection

    Parameters
    ----------
    x: np.ndarray

    Returns
    -------
    np.ndarray
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return x / np.sqrt(1 + conjsq(x))


def projnorm(x, axis=None, proj='none'):
    """Project the variables, Then calculte the mean, std and the lower/upper boundaries.
    Returns the back-projected mean and lower/upper bounds.

    Parameters
    ----------
    x: np.ndarray
        Cellections of vairiables
    axis: int
        Axis to calculate mean, std, etc.
    proj: str
        Type of projection when calculating means and bounds ['none', 'tanh', 'csgp']

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    if proj == 'tanh':
        y = np.arctanh(x)
    elif proj == 'csgp':
        y = csgp_f(x)
    else:
        y = x
    mean = np.mean(y, axis)
    std = np.std(y, axis)
    bound0 = mean + std
    bound1 = mean - std
    if proj == 'tanh':
        return np.tanh(mean), np.tanh(bound0).real, np.tanh(bound1).real
    elif proj == 'csgp':
        return csgp_b(mean), csgp_b(bound0).real, csgp_b(bound1).real
    else:
        return mean, bound0.real, bound1.real


def stalta(d, dt, sta, lta):
    """Calculate STA/LTA for a given signal.
    Note that the output is shorter than the input, which can be fixed by prepAtrig

    Parameters
    ----------
    d: np.ndarray
        Signal data
    dt: float
        Sampling period
    sta: float
        Short-time-average length in second
    lta: float
        Long-time-average length in second

    Returns
    -------
    np.ndarray
    """
    n0 = int(np.floor(len(d) * dt / sta))
    n1 = int(np.floor(sta / dt))
    n2 = int(np.floor(lta / dt))
    slta = np.zeros(n0)
    for k in range(n0):
        ss = abs(d[k * n1 + 1:min((k + 1) * n1 + 1, len(d))]).mean()
        ll = abs(d[int(max(k * n1 - np.floor(n2 / 2 - n1 / 2), 1)):int(
            min((k + 1) * n1 + np.floor(n2 / 2 - n1 / 2), len(d)))]).mean()
        slta[k] = ss / ll
    return slta


def reshape(in_mat, n_sample, n_overlap=0, ext=False):
    """Reshape last axis of matrix/vector to higher dimension

    Parameters
    ----------
    in_mat: np.ndarray
        Input matrix, the last dimension will be extended by reshaping
    n_sample: int
        Number of index in the new dimension
    n_overlap: int
        Number of index to overlap
    ext: bool
        Extended mode. If True, will return one additional window even though
        input length is not enough, overlapping with prior window.

    Returns
    -------
    np.ndarray
    """
    in_shape = list(in_mat.shape)
    n = (in_shape[-1] - n_overlap) // (n_sample - n_overlap)
    if ext:
        out_mat = np.zeros(shape=in_shape[:-1] + [n + 1, n_sample])
    else:
        out_mat = np.zeros(shape=in_shape[:-1] + [n, n_sample])
    for i in range(n):
        in_from = (n_sample - n_overlap) * i
        in_to = in_from + n_sample
        out_mat[..., i, :] = in_mat[..., in_from:in_to]
    if ext:
        out_mat[..., n, :] = in_mat[..., -n_sample:]
    return out_mat


'''
def window(fc, stream, dt=2e-3, win_size=50, overlap=0, atrig=None, atrig0=0.2, atrig1=2.5, verbose=False):
    # prepare window matrices
    n_min = min([trace.stats.npts for trace in stream])
    sig = np.array([trace.data[:n_min] for trace in stream])
    win_width = min(np.floor(1.2 * win_size / fc / dt).astype('int'), sig.shape[1])
    sigmat = reshape(sig, win_width, int((0.2 + overlap) * win_width))
    win_n = sigmat.shape[1]
    if atrig is not None:
        atgmat = reshape(atrig, win_width, int((0.2 + overlap) * win_width))
        qamat = (atgmat >= atrig0) * (atgmat <= atrig1)
        sigmat = sigmat[:, qamat.all(2).all(0), :]
    if verbose:
        print('{:.0f}/{:.0f} windows qualified'.format(sigmat.shape[1], win_n))
    return sigmat


def xcwindow(fc, stream, dt=2e-3, win_size=50, overlap=0, atrig=None, atrig0=0.2, atrig1=2.5, verbose=False):
    if atrig is None:
        return window(fc, stream, dt, win_size, overlap, atrig, atrig0, atrig1, verbose)
    else:
        n_min = min([trace.stats.npts for trace in stream])
        sig = np.array([trace.data[:n_min] for trace in stream])
        win_width = min(np.floor(1.2 * win_size / fc / dt).astype('int'), sig.shape[1])
        assert atrig.shape == sig.shape
        qa = ((atrig >= atrig0) * (atrig <= atrig1)).all(0)
        sig_ = [x for x in np.hsplit(sig, np.where(~qa)[0]) if x.shape[1] >= win_width]
        if len(sig_) > 0:
            return np.concatenate([reshape(isig, win_width, int((0.2 + overlap) * win_width)) for isig in sig_], 1)
        else:
            return np.zeros(shape=(len(stream), 0, win_width))


def hvwindow(sig, dt, winlen0, winlen1, overlap=0, atrig=None, atrig0=0.2, atrig1=2.5):
    """Windowing function with flexible anti-trigger avoiding

    Parameters
    ----------
    sig: np.ndarray
        Pre-windowed signal matrix, can be either 1D or 2D
    dt: float
        Sampling period in second
    winlen0: float
        Minimum window length in second
    winlen1: float
        Target window length in second
    overlap: float
        Overlap rate from 0 to 1
    atrig: np.ndarray
        Anti-triggering matrix, whose shape should be the same as signal matrix
    atrig0: float
        Minimum anti-triggering range
    atrig1: float
        Maximum anti-triggering range
    
    Returns
    -------
    np.ndarray
        Windowed signal matrix, whose second last axis being the windows
    """
    # verify atrig matrix is the same shape as sig
    if atrig is not None:
        assert atrig.shape == sig.shape
        qa = ((atrig >= atrig0) * (atrig <= atrig1)).all(0)
    # else set a default atrig matrix TODO: or use warning/exception?
    else:
        qa = np.ones(sig.shape[-1], dtype=bool)
    # split signal by QA and keep sequneces longer than minimal window length (w/ 10% tukey)
    sig_ = [x for x in np.hsplit(sig, np.where(~qa)[0]) if x.shape[-1] >= winlen0 / dt * 1.1]
    if len(sig_) > 0:
        siglst = list()
        for isig_ in sig_:
            if isig_.shape[-1] * (1 - overlap) % (winlen1 / dt * (1 - overlap)) >= winlen0 / dt:  # TODO: condition incomplete
                # To reshape the signal, we need to calculate the number of zeros to fill
                nfill = int((isig_.shape[-1] * (1 - overlap) // (winlen1 / dt * (1 - overlap)) + 1) * (winlen1 / dt * (1 - overlap)) - isig_.shape[-1] * (1 - overlap))
                siglst.append(
                    reshape(
                        np.hstack([isig_,
                                   np.zeros([isig_.shape[0], nfill])]),
                        int(winlen1 / dt), int((0.2 + overlap) * winlen1 / dt)))
        return np.concatenate(siglst, -2)
    else:
        return np.zeros(shape=(3, 0, int(winlen1 / dt)))
'''


def window(sig, dt, winlen0, winlen1, overlap=0, atrig=None, atrig0=0.2, atrig1=2.5):
    """Windowing function with flexible anti-trigger avoiding

    Parameters
    ----------
    sig: np.ndarray
        Pre-windowed signal matrix, can be either 1D or 2D
    dt: float
        Sampling period in second
    winlen0: float
        Minimum window length in second
    winlen1: float
        Target window length in second
    overlap: float
        Overlap rate from 0 to 1
    atrig: np.ndarray
        Anti-triggering matrix, whose shape should be the same as signal matrix
    atrig0: float
        Minimum anti-triggering range
    atrig1: float
        Maximum anti-triggering range

    Returns
    -------
    np.ndarray
        Windowed signal matrix, whose second last axis being the windows
    """
    # 10% of overlap for tukey window is added by defualt
    overlap += 0.1
    # convert second to index
    n0 = round(winlen0 / dt)
    n1 = round(winlen1 / dt)
    n_ = round(n1 * overlap)
    # verify atrig matrix is the same shape as sig
    if atrig is not None:
        assert atrig.shape == sig.shape
        qa = ((atrig >= atrig0) * (atrig <= atrig1)).all(0)
    else:  # else set a default atrig matrix TODO: or use warning/exception?
        qa = np.ones(sig.shape[1:], dtype=bool)
    # split signal by QA and keep slices longer than minimum window length
    # then reshape window for every qualified slice
    sig_ = [x for x in np.split(sig, np.where(~qa)[0], axis=-1) if x.shape[-1] >= n0]
    if len(sig_) > 0:
        siglst = list()
        for isig in sig_:
            # calculate each signal remainder
            remainder = isig.shape[-1] % (n1 - n_)
            # fill zeros if remainder larger than minimum window length
            if remainder >= n0:
                isig = np.concatenate([isig, np.zeros(list(isig.shape[:-1]) + [n1 - remainder])], axis=-1)
            # reshape slice to target window size
            siglst.append(reshape(isig, n1, n_, ext=False))
        # combine all windowed slices to windowed matrix
        return np.concatenate(siglst, axis=-2)
    else:
        # signal with no qualified slices returns a empty matrix
        return np.zeros(shape=list(sig.shape[:-1]) + [0, n1])


def array_matrix(ifreq, coord, s_min, s_max, s_n, theta_n):
    """Caculate the array transfer function for FK

    Parameters
    ----------
    ifreq: float
        The frequency to transfer
    coord: np.ndarray
        Coordinates of all stations
    s_min: float
        Minimum slowness
    s_max: float
        Maximum slowness
    s_n: int
        Number of slowness
    theta_n: int
        Number of theta

    Returns
    -------
    np.ndarray
        Complex matrix of the array transfer function
    """
    slow = np.linspace(s_min, s_max, s_n)
    theta = np.linspace(0, 2 * np.pi, theta_n)
    dx = coord[:, 0][:, None] - coord[:, 0][None, :]
    dy = coord[:, 1][:, None] - coord[:, 1][None, :]
    dr = np.sin(theta[None, :]) * dx[..., None] + np.cos(theta[None, :]) * dy[..., None]
    delta = slow[None, :] * dr[..., None]
    emat = np.exp(-1j * 2 * np.pi * ifreq * delta)  # TODO: verify sign
    return emat


def dtft(sig, dt, freqs):
    """Discrete time fourier transform by definition (slow but accurate)

    Parameters
    ----------
    sig: np.ndarray
        Signals to transform, can be a multi-dimension matrix (multiple windows/traces).
        Yet only the last dimension will be considered as the time domain
    dt: float
        Sampling period
    freqs: np.ndarray
        Frequencies to calculate

    Returns
    -------
    np.ndarray
        Fourier-transformed matrix, the last dimension is the frequency domain
    """
    n = sig.shape[-1]
    omega = -2 * np.pi * freqs * dt * 1j
    rn = np.sqrt(2 * dt / n)
    base = rn * np.exp(np.linspace(0, n - 1, n)[:, None] * omega[None, :])
    ftrmat = np.dot(sig, base)
    return ftrmat


def dfft(sig, dt, freqs):
    """Discrete-time Fourier transform by FFT (fast and pseudo-accurate)

    Parameters
    ----------
    sig: np.ndarray
        Signals to transform, can be a multi-dimension matrix (multiple windows/traces).
        Yet only the last dimension will be considered as the time domain
    dt: float
        Sampling period
    freqs: np.ndarray
        Frequencies to calculate

    Returns
    -------
    np.ndarray
        Fourier-transformed matrix, the last dimension is the frequency domain
    """
    n = sig.shape[-1]
    fft = np.fft.fft(sig, axis=-1)
    intp = interpolate.interp1d(np.linspace(0, 1 / dt, n), fft, axis=-1)
    return intp(freqs)


def xcorrelation(sigmat, fc, bandwidth=0.05, nf=21, dt=2e-3, do_dtft=False):
    """Cross-correlation at a central frequency

    Parameters
    ----------
    sigmat: np.ndarray
        Signal matrix, the first dimension is to be cross-correlated, the last dimension is the time domain
    fc: float
        Central frequency
    bandwidth: float
        Bandwidth to calculate Fourier transform, ranging in [0, 1), default to 0.05
    nf: int
        Number of frequncies in the band, set to 51 in Geopsy 3.0+, but a nf=21 works just as fine
    dt: float
        Sampling period
    do_dtft: bool
        Calculate DTFT by definition if True, otherwise use the faster FFT method

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized/Pre-normalized cross-correlated matrix (covariance/correlation matrix)
    """
    # discrete time fourier transfer
    sigmat *= signal.windows.tukey(sigmat.shape[-1], 0.2)  # 10% tukey taper on each side
    freqs = fc * np.linspace(1 - bandwidth, 1 + bandwidth, nf)
    if do_dtft:
        ftrmat = dtft(sigmat, dt, freqs)
    else:
        ftrmat = dfft(sigmat, dt, freqs)
    # cross-correlation and normalization
    fweight = np.exp(-((2 / (bandwidth * fc) * (freqs - fc)) ** 2))  # gaussian average over frequency band
    cormat = np.average(ftrmat[:, None] * ftrmat.conj()[None, :], -1, fweight)
    diag = np.rollaxis(cormat.diagonal(), -1)
    assert (diag.real * 1e-10 > diag.imag).all()
    normat = np.sqrt(diag[:, None] * diag.conj()[None, :]).real
    covmat = cormat / normat
    return covmat, cormat


def xcor1freq(i, freqs, sig0, dt=1e-3, bandwidth=0.02, nf=21, do_dtft=False, win_size=50, overlap=0,
              atrig=None, atrig0=0.2, atrig1=2.5, verbose=False, update=False):
    """Cross-correlation at 1 freqeuncy (for parallel computation)

    Parameters
    ----------
    i: int
        Frequency index
    freqs: np.ndarray
        1-D array of central frequencies
    sig0: Stream
        Obspy.Stream of signals
    dt: float
        Sampling period
    bandwidth: float
        Bandwidth for Fourier transform, ranging in [0, 1)
    nf: int
        Number of frequncies in the band, set to 51 in Geopsy 3.0+, but a nf=21 works just as fine
    do_dtft: bool
        Calculate DTFT by definition if True, otherwise use the faster FFT method
    win_size: float
        Frequency dependent window size, meaning the multiples of the wavelength
    overlap: float
        Time window overlapping rate, ranging in [0, 1)
    atrig: np.ndarray
        STA/LTA anti-triggering matrix
    atrig0: float
        Lower STA/LTA limit
    atrig1: float
        Upper STA/LTA limit
    verbose: bool
    update: bool

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized/Pre-normalized cross-correlated matrix (covariance/correlation matrix)
    """
    ifreq = freqs[i]
    if update:
        update_prog('XCOR loop', [i, len(freqs)], 0, 'Freq = {:.3f} Hz'.format(ifreq))
    win_len = win_size / ifreq
    sig = np.array([trace.data for trace in sig0])
    sigmat = window(sig, dt, win_len, win_len, overlap, atrig, atrig0, atrig1)
    if sigmat.size == 0:
        return np.empty(shape=(len(sig0), len(sig0), 0)), np.empty(shape=(len(sig0), len(sig0), 0))
    else:
        covmat, cormat = xcorrelation(sigmat, ifreq, bandwidth, nf, dt, do_dtft)
        return covmat, cormat


def geopsy_component(channel):
    """Convert geopsy component name

    Parameters
    ----------
    channel: str

    Returns
    -------
    str
    """
    if channel == 'Z':
        return 'Vertical'
    if channel == 'N' or channel == 'Y':
        return 'North'
    if channel == 'E' or channel == 'X':
        return 'East'


'''
# test functions for block-averaging method
def get_block_mat(n0, block_num, block_overlap):
    n1 = int(np.floor((n0 - block_num) / (block_num - block_overlap))) + 1
    block_mat = np.zeros(shape=[n0, n1])
    for i in range(n1):
        block_mat[(block_num - block_overlap) * i:(block_num - block_overlap) * i + block_num, i] = np.array(1)
    return block_mat / block_num


# test functions for frequency-bessel method
def frequency_bessel(xc, r):
    # todo: sort r in rising order
    assert xc.shape[0] == len(r)
    freq = np.geomspace(0.1,100,400)
    c = np.linspace(100,1000,901)
    k_mat = 2*np.pi*freq[:,None]/c[None,:]
    kr = k_mat[...,None] * r[None,...]
    dr = np.hstack([r[0], r[1:] - r[:-1]])
    dxc = np.vstack([xc[0],xc[1:] - xc[:-1]])
    b = dxc/dr[:,None]
    B0 = np.zeros(kr.shape)
    for i in range(kr.shape[0]):
        for j in range(kr.shape[1]):
            for k in range(kr.shape[2]):
                B0[i,j,k]=integrate.quad(lambda x: special.j0(x), 0, kr[i,j,k])[0]
    I0 = 1/k_mat[...,None] * xc.T[:,None,:].shape * r[None,None,:] * special.j1(kr) +\
         b.T[:,None,:] / k_mat[...,None]**3 * (kr*special.j0(kr) - B0)
    return I0.sum(-1)
'''
