import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, mlab, ticker, dates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import j0
from scipy.interpolate import Rbf

# experimental: Using plotly for 3D plotting
import plotly.graph_objects as go


def volume(coord, value=None, zaxis=None, zlog=True, zname='Frequency/Hz', pcoord=None, function='multiquadric',
           title=None, name=None, cmap=None, cmin=None, cmax=None, aspectmode='auto', aspectratio=None, saveto=None):
    """Interpolate 1-D measurements from various stations (same scale) to 3-D space, with given 2-D coordinates.

    Parameters
    ----------
    coord: np.ndarray
    value: np.ndarray
    zaxis: np.ndarray or list[np.ndarray]
    zlog: bool
    zname: str
    pcoord: np.ndarray
    function: str
        'multiquadric': sqrt((r/self.epsilon)**2 + 1)
        'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
        'gaussian': exp(-(r/self.epsilon)**2)
        'linear': r
        'cubic': r**3
        'quintic': r**5
        'thin_plate': r**2 * log(r)
    title: str
    name: list[str]
    cmap: list[str]
    cmin: list[float]
    cmax: list[float]
    aspectmode: str
        "cube", this scene's axes are drawn as a cube, regardless of the axes' ranges.
        "data", this scene's axes are drawn in proportion with the axes' ranges.
        "manual", this scene's axes are drawn in proportion with the input of "aspectratio" keyword.
        "auto", this scene's axes are drawn using the results of "data" except when one axis is more than
            four times the size of the two others, where in that case the results of "cube" are used.
    aspectratio: tuple(float)
    saveto: str or Path
    """
    # input check
    assert len(coord.shape) == 2
    if type(value) == list:
        assert all([ivalue.shape == value[0].shape for ivalue in value])
        vshape = value[0].shape
    else:
        vshape = value.shape
        value = [value]
    assert coord.shape[0] == vshape[0]
    assert zaxis.shape[0] == vshape[1]
    if pcoord is None:
        pcoord = coord
    else:
        assert len(pcoord.shape) == 2
    if cmap is None:
        cmap = ['Hot_r'] * len(value)
    else:
        assert len(cmap) == len(value)

    # boundary setup
    edge = 0.05
    xm = (min(np.concatenate([coord[:, 0], pcoord[:, 0]])), max(np.concatenate([coord[:, 0], pcoord[:, 0]])))
    ym = (min(np.concatenate([coord[:, 1], pcoord[:, 1]])), max(np.concatenate([coord[:, 1], pcoord[:, 1]])))
    xb = (xm[0] - (xm[1] - xm[0]) * edge, xm[1] + (xm[1] - xm[0]) * edge)
    yb = (ym[0] - (ym[1] - ym[0]) * edge, ym[1] + (ym[1] - ym[0]) * edge)

    # interpolation  TODO: interpolation is not necessary, try re-organizing data
    ngrid = (20, 20)
    xy = np.meshgrid(np.linspace(xb[0], xb[1], ngrid[0]),
                     np.linspace(yb[0], yb[1], ngrid[1]))
    # eps = np.unique((((coord-coord[:, None])**2).sum(-1))**0.5)[1]
    itp = [Rbf(coord[:, 0], coord[:, 1], ivalue, function=function, mode='N-D')(*xy) for ivalue in value]

    # 3D plot
    xx, yy, zz = np.meshgrid(np.linspace(xb[0], xb[1], ngrid[0]),
                             np.linspace(yb[0], yb[1], ngrid[1]), zaxis)
    if cmin is None:
        cmin = [np.percentile(itp[i], 10) for i in range(len(itp))]
    else:
        assert len(cmin) == len(itp)
    if cmax is None:
        cmax = [np.percentile(itp[i], 90) for i in range(len(itp))]
    else:
        assert len(cmax) == len(itp)
    fig = go.Figure()
    for i in range(len(itp)):
        fig.add_trace(go.Volume(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=itp[i].flatten(),
                                cmin=cmin[i], cmax=cmax[i],
                                # isomin=0.5,  isomax=itp.max(),
                                opacity=0.1,  # needs to be small to see through all surfaces
                                surface_count=20,  # needs to be a large number for good volume rendering
                                colorscale=cmap[i]  # a colormap from plotly
                                ))
    fig.add_trace(go.Scatter3d(x=pcoord[:, 0], y=pcoord[:, 1],
                               z=np.repeat(max(zaxis), pcoord.shape[0]),
                               text=name, opacity=0.8,
                               mode='markers+text', showlegend=False,
                               marker=dict(symbol='diamond', color='black')))
    for i in range(pcoord.shape[0]):
        fig.add_trace(go.Scatter3d(x=np.repeat(pcoord[i, 0], 2), y=np.repeat(pcoord[i, 1], 2),
                                   z=np.array([min(zaxis), max(zaxis)]),
                                   mode='lines', line=dict(color='grey', width=1, dash='dash'),
                                   showlegend=False))
    if zlog:
        fig.update_layout(scene=dict(zaxis=dict(type='log', range=[np.log10(min(zaxis)), np.log10(max(zaxis))])))
    else:
        fig.update_layout(scene=dict(zaxis=dict(type='linear', range=[min(zaxis), max(zaxis)])))
    fig.update_layout(scene=dict(
        xaxis=dict(title='X/m'),
        yaxis=dict(title='Y/m'),
        zaxis=dict(title=zname),
        aspectmode=aspectmode, aspectratio=aspectratio))
    if title:
        fig.update_layout(title=title)
    # fig.show()  # test
    if saveto:
        fig.write_html(str(saveto))
    else:
        fig.show()


def location(sig0, station=None, title=None, saveto=None):
    """Plot station location from obspy.Stream user7/user8

    Parameters
    ----------
    sig0: obspy.Stream or np.ndarray
        Stream class with at least 1 Trace
    station: list[str]
        (If sig0 is array) Station names
    title: str
        Title of the plot
    saveto: str or Path
        Full name of the ouput file
    """
    try:
        if type(sig0) == np.ndarray:
            assert len(sig0.shape) == 2
            x = sig0[:, 0]
            y = sig0[:, 1]
            if station:
                assert len(station) == len(x)
        elif type(sig0).__name__ == 'Stream':
            station = [isig.stats.station for isig in sig0]
            try:
                x = np.array([isig.stats.sac.user7 for isig in sig0])
                y = np.array([isig.stats.sac.user8 for isig in sig0])
            except AttributeError:
                x = np.array([isig.stats.stlo for isig in sig0])
                y = np.array([isig.stats.stla for isig in sig0])
    except:
        raise Exception('Error Loading coordinates')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    for i in range(len(sig0)):
        ax.plot(x[i], y[i], linestyle='', marker='^', markersize=10)
        if station:
            ax.text(x[i], y[i], station[i], fontsize=16, horizontalalignment='right', verticalalignment='bottom')
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    dlim = max(abs(xlim[0] - xlim[1]), abs(ylim[0] - ylim[1])) / 2  # * 1.1
    ax.set_xlim([xlim.mean() - dlim, xlim.mean() + dlim])
    ax.set_ylim([ylim.mean() - dlim, ylim.mean() + dlim])
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid()
    ax.set_aspect(1)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Station locations')
    fig.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


class minuteFormatter(ticker.Formatter):
    """
    Format a tick (in minutes since the epoch) with a
    `~datetime.datetime.strftime` format string.
    """

    def __init__(self, fmt, tz=None):
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string
        tz : `datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone.
        """
        if tz is None:
            tz = dates._get_rc_timezone()
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
        return dates.num2date(x / 24 / 60, self.tz).strftime(self.fmt)

    def set_tzinfo(self, tz):
        self.tz = tz


# plot raw signal with sta/lta
def signals(sig0, dt=None, atrig=None, time=None, trange=None, loc='', channel=False, saveto=None):
    """Plot signal traces with antitrigger backgrounds

    Parameters
    ----------
    sig0: obspy.Stream or np.ndarray or Sensor3c
    dt: float
    atrig: np.ndarray
    time: np.ndarray
    trange: list[float,float]
    loc: str
    channel: bool
    saveto: str or Path
    """
    # create variables needed prior to plotting
    if type(sig0).__name__ == 'Sensor3c':
        n = 3
        data = np.array([getattr(sig0, attr).amp for attr in ['vt', 'ns', 'ew']])
        time = sig0.vt.time
        chn = ['Z', 'N', 'E']
        try:
            fname = sig0.meta['File Name']
        except KeyError:
            fname = '[]'
        sta = [fname[fname.find('[') + 1:fname.find(']')]] * 3
        loc = fname[:fname.find('[')]
    elif type(sig0).__name__ == 'Stream':
        n = len(sig0)
        data = np.array([sig0[i].data for i in range(n)])
        time = sig0[0].times('timestamp') / 60
        chn = [sig0[i].stats.channel for i in range(n)]
        sta = [sig0[i].stats.station for i in range(n)]
        loc = sig0[0].stats.location
    elif type(sig0) == np.ndarray:
        assert len(sig0.shape) == 2
        n = sig0.shape[0]
        if time is None:
            time = np.arange(sig0.shape[1]) * dt
        data = sig0
    else:
        raise TypeError('Signal type inccorect')
    if trange is None:
        trange = [time[0], time[-1]]
    # initiate plotting
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(18, 9))
    yrange = np.percentile(abs(data), 99.99, axis=-1).mean()
    for i in range(n):
        axs[i].plot(time.T, data[i].T, linewidth=0.7, color='k')
        axs[i].set_ylim(-yrange, yrange)
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axs[i].yaxis.set_ticks([])
        if channel:
            iylabel = '.'.join([sta[i], chn[i]])
        else:
            iylabel = sta[i]
        axs[i].set_ylabel(iylabel)
        axs[i].grid('both')
        if atrig is not None:
            pat = axs[i].imshow(np.asmatrix(atrig[i]),
                                cmap='RdBu_r', norm=colors.LogNorm(0.1, 10, 1),
                                aspect='auto', interpolation='none',
                                extent=trange + [-yrange, yrange])
    axs[-1].set_xlim(trange)
    if dt is not None:
        axs[-1].set_xlabel('Time (s)')
    else:
        axs[-1].xaxis.set_major_formatter(minuteFormatter('%m-%d %H:%M'))
    # preparing title
    if loc != '':
        title = '\n' + str(loc)
        topmargin = 0.94
    else:
        title = ''
        topmargin = 0.96
    # remove horizontal space between axes
    if atrig is not None:
        fig.subplots_adjust(top=topmargin, bottom=0.04, left=0.02, right=0.96, hspace=0)
        fig.suptitle('Array signal with STA/LTA' + title)
        cbar = fig.colorbar(pat, cax=fig.add_axes([0.97, 0.04, 0.01, 0.9]), ticks=[0.1, 1, 10])
        cbar.ax.set_yticklabels([0.1, 1, 10])
        cbar.ax.set_xlabel('STA/LTA')
        cbar.ax.xaxis.set_label_position('top')
    else:
        fig.subplots_adjust(top=topmargin, bottom=0.04, left=0.02, right=0.98, hspace=0)
        fig.suptitle('Array signal' + title)
    # save file
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def stft(sig0, nfft=256, novlp=75, saveto=None):
    """Calculate and plot short-time Fourier transform

    Parameters
    ----------
    sig0: obspy.Stream
    nfft: int
        Number of fast Fourier trasnform
    novlp: int
        Number of overlaps
    saveto: str or Path
    """
    fig, axs = plt.subplots(len(sig0), 1, sharex=True, figsize=(18, 9))
    for i in range(len(sig0)):
        pat = axs[i].specgram(sig0[i], nfft, sig0[i].stats.sampling_rate, noverlap=novlp, vmax=-20, vmin=-200)[3]
        axs[i].set_ylim(1, 100)
        axs[i].set_yscale('log')
        axs[i].set_ylabel(sig0[i].stats.station + '\nFrequency (Hz)')
        axs[i].yaxis.set_major_formatter('{x:.16g}')
    axs[-1].set_xlabel('Relative time (s)')
    if sig0[0].stats.location != '':
        fig.suptitle('Short-time-Fourier-Transform @ 1-100 Hz\n{}'.format(sig0[0].stats.location))
    else:
        fig.suptitle('Short-time-Fourier-Transform @ 1-100 Hz')
    # Remove horizontal space between axes
    fig.tight_layout()
    fig.subplots_adjust(right=0.942, hspace=0.08)
    fig.colorbar(pat, cax=fig.add_axes([0.947, 0.055, 0.02, 0.875]), label='Magnitude (dB)')
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def psd(sig0, nfft=256, novlp=75, plog=False, saveto=None, csvto=None):
    """Calculate and plot power spectrum density

    Parameters
    ----------
    sig0: obspy.Stream
    nfft: int
        Number of fast Fourier transform
    novlp: int
        Number of overlaps
    plog: bool
        Whether use log-scale
    saveto: str or Path
    csvto: str
        If a csv of PSD is saved

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        Center frequencies and the according power
    """
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 9))
    psd0 = []
    freq0 = []
    for i in range(len(sig0)):
        temp = ax.psd(sig0[i].data, nfft, Fs=sig0[i].stats.sampling_rate,
                      window=mlab.window_hanning, pad_to=512, noverlap=novlp)
        psd0.append(temp[0])
        freq0.append(temp[1])
    psd1 = np.log10(np.array(psd0))
    freq = np.array(freq0).mean(0)
    ax.set_xlim(0.1, 100)
    if plog:
        ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Power Spectral Density | ' + ' '.join(np.unique([tr.stats.location for tr in sig0])))
    ax.legend(['-'.join([tr.stats.network, tr.stats.station]) for tr in sig0])
    fig.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
        if csvto:
            head = 'Power spectrum density (log)\n' \
                   'Column names:\n' \
                   'Frequency, ' + ', '.join([tr.stats.station for tr in sig0])
            np.savetxt(csvto, np.row_stack([freq, psd1]).T,
                       fmt='%.3f', delimiter=', ', newline='\n', comments='# ', header=head)
    else:
        plt.show()
    return freq, psd1


def spac_curve(freq, spac, uppper=None, lower=None, r=None, ref=None,
               log=True, xrange=None, cp=None, t0=None, saveto=None):
    """Plot SPAC curve (optionally compared to Geopsy)
        # TODO: clarify input types
    Parameters
    ----------
    freq:
        Central frequencies
    spac:
        SPAC curve means
    uppper:
        SPAC upper bounds
    lower:
        SPAC lower bounds
    r:
        Ring radii in metre
    ref:
        If a Geopsy reference is provided
    log:
        If x-axis is in log scale
    xrange:
        X-axis range
    cp:
        # TODO: Compared to reference, print the residual and Pearson-correlation
    t0:
        Super-title of the plot
    saveto:
    """
    assert len(spac.shape) == 2
    _spac = np.zeros(shape=spac.shape, dtype=complex)
    for i in range(len(spac)):
        _spac[i] = np.interp(freq, freq[~np.isnan(spac[i])], spac[i, ~np.isnan(spac[i])], left=0 + 0 * 1j)
    nfig = spac.shape[0] // 3 + 1 if spac.shape[0] % 3 != 0 else spac.shape[0] // 3
    sizefig = [3] * nfig
    sizefig[-1] = spac.shape[0] % 3 if spac.shape[0] % 3 != 0 else 3
    for ifig in range(nfig):
        fig = plt.figure(num='SPatial Auto-Correlation', figsize=(6 * sizefig[ifig], 9))
        axs = fig.subplots(1, sizefig[ifig])
        try:
            axs[0]
        except TypeError:
            axs = [axs]
        for i in range(sizefig[ifig]):
            if uppper is not None and lower is not None:
                axs[i].fill_between(freq, lower[i + ifig * 3], uppper[i + ifig * 3], alpha=0.4)
            axs[i].plot(freq, _spac[i + ifig * 3].real)
            if ref:
                axs[i].plot(ref[i + ifig * 3].freq, ref[i + ifig * 3].spac)
                axs[i].legend(['this program', 'geopsy'])
            else:
                axs[i].plot(freq, _spac[i + ifig * 3].imag, linestyle='--')
                axs[i].legend(['real', 'imag'])
            if log:
                axs[i].set_xscale('log')
                axs[i].xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
                axs[i].xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
                axs[i].xaxis.set_major_formatter('{x:.16g}')
                if xrange is not None:
                    axs[i].set_xlim(xrange)
                else:
                    axs[i].set_xlim([0.5, 100])
            else:
                if xrange is not None:
                    axs[i].set_xlim(xrange)
                else:
                    axs[i].set_xlim([0, 100])
            axs[i].set_ylim([-0.5, 1.05])
            axs[i].grid(which='both')
            if r is None:
                text = 'Ring {}'.format(i + ifig * 3 + 1)
            else:
                text = 'Ring {}: {:.2f} m'.format(i + ifig * 3 + 1, r[i + ifig * 3])
            '''                
            if cp:
                text += '\nResidual = {:.3f}\nCorrelation = {:.1f}%'.format(cp[0][i + ifig * 3],
                                                                            cp[1][i + ifig * 3] * 100)
            '''
            axs[i].set_title(text)
            axs[i].set_xlabel('Frequency (Hz)')
        if t0:
            if nfig == 1:
                fig.suptitle(t0)
            else:
                fig.suptitle(t0 + ' [{}]'.format(ifig + 1))
        fig.tight_layout()
        if saveto:
            if nfig == 1:
                ifile = f'{saveto}.png'
            else:
                ifile = f'{saveto}[{ifig + 1}].png'
            plt.savefig(ifile)
            plt.close()
        else:
            plt.show()


def spac_single_pair(freq, ispac, iup=None, ilow=None, iname='', iring=None, ipath=None, log=True):
    """Plot SPAC curves of each single pair

    Parameters
    ----------
    freq: np.ndarray
        Central frequencies
    ispac: np.ndarray
        1-dimensional SPAC mean value
    iup: np.ndarray
        1-dimensional SPAC upper bound
    ilow: np.ndarray
        1-dimensional SPAC lower bound
    iname: str
        Name of the pair
    iring: float
        Ring radius of the pair
    ipath: Path
        Filepath to save
    log: bool
        If the frequency axis is in log scale
    """
    _spac = np.interp(freq, freq[~np.isnan(ispac)], ispac[~np.isnan(ispac)], left=0 + 0 * 1j)
    fig = plt.figure(1, figsize=(12, 9))
    ax = fig.add_subplot()
    if iup is not None and ilow is not None:
        ax.fill_between(freq, ilow, iup, alpha=0.4)
    ax.plot(freq, _spac.real)
    ax.plot(freq, _spac.imag, linestyle='dashed')
    ax.set_xlim([0.5, 100])
    ax.set_ylim(-1.1, 1.1)
    if log:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
        ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.xaxis.set_major_formatter('{x:.16g}')
    ax.grid(which='both')
    ax.legend(['real', 'imag'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Cross-correlation ratio')
    ax.set_title('{} | r = {:.2f} m'.format(iname, iring))
    fig.tight_layout()
    if iring is None or ipath is None:
        plt.show()
    else:
        plt.savefig(ipath)
        plt.close()


def all_pairs(freq, result, pair, saveto=None):
    """Simple plot of all SPAC pair results (not used)

    Parameters
    ----------
    freq: np.ndarray
        Central frequencies
    result: list
        Multicore corss-correlation results
    pair: list
        Pair IDs
    saveto: str or Path
    """
    plt.figure()
    for i in range(len(result)):
        plt.subplot(3, 7, i + 1)
        temp = np.zeros(len(freq), dtype=complex)
        for j in range(len(freq)):
            temp[j] = np.nanmean(result[i][0][j])
        plt.plot(freq, temp.real)
        plt.plot(freq, temp.imag, dashes=[3, 1])
        plt.title('{}-{}'.format(pair[i][0][0], pair[i][1][0]))
        plt.grid()
        plt.xlim(0, 50)
    # plot every single pair
    pair_name = []
    for ipair in pair:
        pair_name.append('{}-{}'.format(ipair[0][0], ipair[1][0]))
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def all_pairs_group(freq, result, sel, pairname, ffs=None, saveto=None):
    """Simple plot of all SPAC pair results in groups (not used)"""
    plt.figure()
    for i in range(len(sel)):
        ring_ind = list(np.where(sel[i])[0])
        names = [pairname[i] for i in ring_ind]
        for k in ring_ind:
            temp = np.zeros(len(freq), dtype=complex)
            for j in range(len(freq)):
                temp[j] = np.nanmean(result[k][j])
            plt.subplot(2, 3, i + 1)
            plt.plot(freq, temp.real, alpha=0.7)
            plt.legend(names)
            plt.subplot(2, 3, i + 4)
            plt.plot(freq, temp.imag, alpha=0.7)
            plt.legend(names)
    if ffs:
        plt.suptitle('All pairs vs. Geopsy')
        for i in range(3):
            plt.subplot(2, 3, i + 1)
            file = np.loadtxt(ffs[i])
            plt.plot(file[:, 0], file[:, 1], color='k')
            plt.grid()
            plt.xlabel('Frequency/Hz')
            plt.xlim(0, 50)
            plt.title('Ring {:.0f} real'.format(i))
        for i in range(3):
            plt.subplot(2, 3, i + 4)
            plt.plot(freq, np.zeros(len(freq)), color='k')
            plt.grid()
            plt.xlabel('Frequency/Hz')
            plt.xlim(0, 50)
            plt.title('Ring {:.0f} imag'.format(i))
    else:
        plt.suptitle('All pairs grouped')
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def fk_imaging(bp, freq=None, sel=None, theta=np.linspace(-np.pi, np.pi, 181), lim=None, lvls=None, k=None,
               cmap='CMRmap_r', title=None, saveto=None):
    """Plot F-K in polar projection

    Parameters
    ----------
    bp: np.ndarray
    freq: np.ndarray
    sel: list[tuple[float, float]]
    theta: np.ndarray
    lim: tuple[float, float]
    lvls: np.ndarray
    k: np.ndarray
    cmap: str
    title: str
    saveto: str or Path
    """
    # initiate inputs
    if len(bp.shape) == 2:
        bp = bp[None, ...]
    elif len(bp.shape) != 3:
        raise Exception('Wrong input dimension! Beampower matrix must be 2/3-D')
    if freq is None:
        unit = ''
        freq = np.arange(bp.shape[0])
    else:
        unit = 'Hz'
        assert len(freq) == bp.shape[0]
    if sel is None:
        sel = [(min(freq), max(freq))]
    nrow = len(sel) // 3 + 1 if len(sel) % 3 != 0 else len(sel) // 3
    ncol = len(sel) if len(sel) <= 3 else 3
    if k is None:
        k = np.arange(bp.shape[-1])
    # plotting
    fig = plt.figure(figsize=(6 * ncol, 9 * nrow))
    for i in range(len(sel)):
        p = np.nanmean(bp[(freq >= sel[i][0]) * (freq <= sel[i][1])], 0)
        if lvls is None:
            ilvls = np.linspace(0, np.percentile(p, 99.5), 15)
        else:
            ilvls = lvls
        ax = fig.add_subplot(nrow, ncol, i + 1, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pat = ax.contourf(theta, k, p.T, levels=ilvls, cmap=cmap, extend='max', alpha=.9)
        if lim:
            ax.set_rlim(lim[0], lim[1])
        # ax.grid() TODO: make sure if this line is needed
        ax.set_rgrids(ax.get_yticks(), ['{:.0f}m/s'.format(1 / tk) for tk in ax.get_yticks()])
        ax.set_title('Range: {} - {} {}'.format(sel[i][0], sel[i][1], unit))
        cbar = plt.colorbar(pat, orientation='horizontal', spacing='uniform', format=ticker.ScalarFormatter())
        cbar.set_label('Power')
        cbar.ax.xaxis.set_label_position('top')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def gpy_fk(fk, frange, lim=None, title=None, saveto=None):
    """Plot Geopsy F-K .max file in polar projection

    Parameters
    ----------
    fk: np.recarray
    frange: list[tuple[float, float]]
    lim: tuple[float,float]
    title: str
    saveto: str or Path
    """
    # initiate inputs
    if frange is None:
        frange = [(np.floor(min(fk['frequency'])), np.ceil(max(fk['frequency'])))]
    nrow = len(frange) // 3 + 1 if len(frange) % 3 != 0 else len(frange) // 3
    ncol = len(frange) if len(frange) <= 3 else 3
    # plotting
    fig = plt.figure(figsize=(6 * ncol, 9 * nrow))
    for i in range(len(frange)):
        sel = (fk['frequency'] >= frange[i][0]) * (fk['frequency'] <= frange[i][1])
        if sel.sum() <= 1:
            warnings.warn('No peaks found in frequency range')
            continue
        slw_range = fk['slowness'][sel].max() - fk['slowness'][sel].min()
        z = np.histogram2d(fk['azimuth'][sel], fk['slowness'][sel], bins=(60, int(slw_range * 1e4) + 1))
        ax = fig.add_subplot(nrow, ncol, i + 1, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pat = ax.pcolormesh(z[1] / 180 * np.pi, z[2], z[0].T, cmap='gnuplot2_r')
        if lim:
            ax.set_rlim(lim[0], lim[1])
        ax.grid()
        ax.set_rgrids(ax.get_yticks(), ['{:.0f}m/s'.format(1 / tk) for tk in ax.get_yticks()])
        ax.set_title('Range: {} - {} Hz'.format(frange[i][0], frange[i][1]))
        cbar = plt.colorbar(pat, orientation='horizontal', spacing='uniform', format=ticker.ScalarFormatter())
        cbar.set_label('Density')
        cbar.ax.xaxis.set_label_position('top')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def spac2disp(spac, freq, r3, c0=None, stderr=None, log=False, norm=None, shading='auto', t0=None, saveto=None,
              plot=True):
    """Project SPAC curves to phase velocity dispersion domain"""
    if c0 is None:
        c0 = np.linspace(10, 1000, 991)
    invsum = np.zeros([len(c0), len(freq)])
    n = len(spac)
    if len(r3) != n:
        exit('Length of radii/spac not match!')
    for i in range(n):
        ispac = np.interp(freq, freq[~np.isnan(spac[i])], spac[i, ~np.isnan(spac[i])], left=0 + 0 * 1j)
        spac0 = j0(freq * 2 * np.pi * r3[i] / c0[:, None])
        d2 = (spac0 - ispac.real) ** 2
        if stderr is not None:
            d2 = d2 * stderr[i]
        invsum += 1 / d2
    if plot:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        if norm is None:
            norm = colors.LogNorm(10 ** np.floor(np.log10(np.percentile(1 / invsum, 10))), np.median(1 / invsum))
        pat = ax.pcolormesh(freq, c0, 1 / invsum, cmap='hot', norm=norm, shading=shading)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Velocity (m/s)')
        if t0:
            ax.set_title(t0)
        else:
            ax.set_title('SPAC to DISP')
        if log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
            ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
            ax.xaxis.set_major_formatter('{x:.16g}')
        ax.set_xlim(0.5, 100)
        ax.set_ylim(min(c0), max(c0))
        ax.grid(which='both')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "3%", pad="3%")
        plt.colorbar(pat, cax=cax)
        plt.title('residual')
        plt.tight_layout()
        if saveto:
            plt.savefig(saveto)
            plt.close()
        else:
            plt.show()
    return 1 / invsum


def fkbp2disp(bp, freq, slow, log=True, norm=None, normal=False, shading='auto', t0=None, saveto=None, plot=True):
    """Project FK results to phase velocity dispersion domain"""
    sel = ~np.isnan(bp).any((1, 2))
    bp[np.isnan(bp)] = 0
    bp_max = np.nanmax(bp, 1)
    if normal:
        bp_max /= bp_max.mean(1)[:, None]  # TODO: encountered in true_divide
    if plot:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        if norm is None:
            if normal:
                norm = colors.Normalize(1, np.nanmax(bp_max))
            else:
                norm = colors.LogNorm(np.median(bp_max[sel]), 10 ** np.ceil(np.log10(bp_max[sel].max())))
        pat = ax.pcolormesh(freq[sel], 1 / slow, bp_max[sel].T, cmap='gnuplot2_r', norm=norm, shading=shading)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Velocity (m/s)')
        if t0:
            ax.set_title(t0)
        else:
            ax.set_title('FKmax to DISP')
        if log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
            ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
            ax.xaxis.set_major_formatter('{x:.16g}')
        ax.set_xlim(0.5, 100)
        ax.set_ylim(100, 1200)  # TODO: remove fixed velocity range
        ax.grid(which='both')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "3%", pad="3%")
        plt.colorbar(pat, cax=cax)
        if normal:
            plt.title('Norm power', fontsize=10)
        else:
            plt.title('Power', fontsize=10)
        plt.tight_layout()
        if saveto:
            plt.savefig(saveto)
            plt.close()
        else:
            plt.show()
    return bp_max.T


def gpy_fkmd2disp(fk, log=True, t0=None, saveto=None, plot=True):
    """Project Geopsy FK results to phase velocity dispersion domain

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Frequencies, velocities, density matrix

    """
    z = np.histogram2d(np.log10(fk['frequency']), fk['slowness'], bins=(100, 400))
    if plot:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        pat = ax.pcolormesh(10 ** z[1], 1 / z[2], z[0].T, cmap='gnuplot2_r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Velocity (m/s)')
        if t0:
            ax.set_title(t0)
        else:
            ax.set_title('[GPY] FKmax to DISP')
        if log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
            ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
            ax.xaxis.set_major_formatter('{x:.16g}')
        ax.set_xlim(0.5, 100)
        ax.set_ylim(100, 1200)
        ax.grid(which='both')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "3%", pad="3%")
        plt.colorbar(pat, cax=cax)
        plt.title('density', fontsize=10)
        plt.tight_layout()
        if saveto:
            plt.savefig(saveto)
            plt.close()
        else:
            plt.show()
    return 10 ** z[1], 1 / z[2], z[0].T


def hv_class(hv, distribution_mc='log-normal', distribution_f0='log-normal', plot_f0=False, log=True,
             title=None, saveto=None):
    """Plot H/V curves from HVSR class"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    if len(hv.amp.shape) == 2:
        for i in hv.valid_window_indices:
            ax.plot(hv.frq, hv.amp[i], color=cm.get_cmap('rainbow_r')(i / hv.n_windows), linewidth=0.8, alpha=0.8)
    ax.plot(hv.frq, hv.mean_curve(distribution_mc), color='k', linewidth=1.5)
    ax.plot(hv.frq, hv.nstd_curve(-1, distribution_mc), color='k', linewidth=1.5, linestyle='--')
    ax.plot(hv.frq, hv.nstd_curve(+1, distribution_mc), color='k', linewidth=1.5, linestyle='--')
    ymin = 0
    ymax = hv.nstd_curve(1, distribution_mc)[hv.valid_frequencies].max() * 1.1
    if plot_f0:
        ax.axvline(hv.mean_f0_frq(distribution_f0), color='k', linewidth=1, linestyle='-.')
        ax.fill([hv.nstd_f0_frq(-1, distribution_f0)] * 2 + [hv.nstd_f0_frq(+1, distribution_f0)] * 2,
                [ymin, ymax, ymax, ymin], color="#ff8080")

    if log:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
        ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.xaxis.set_major_formatter('{x:.16g}')
    ax.set_ylim(0, ymax)
    ax.set_xlim(min(hv.frq), max(hv.frq))
    ax.grid()
    ax.set_ylabel('HVSR Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def hv_array(hvs, freq, log=True, plottype='mean', legend=None, title=None, saveto=None):
    """Plot H/V curves from np.ndarray

    Parameters
    ----------
    hvs: np.ndarray
        Array of H/V curves
    freq: np.ndarray
        1D array of frequencies
    log: bool
        If the frequency axis is log-scale
    plottype: str
        Type of the plot:
            'mean': Only mean value of one H/V curve;
            'meanstd': mean value and upper/lower bounds calculated from std (log);
            'array': Arrays of H/V curves, without mean or std;
            'all': Arrays of H/V curves, with mean and std
    legend: list[str]
    title: str
    saveto: str or Path
    """
    if plottype.lower() == 'mean':
        assert len(hvs.shape) == 1
        meanhv = hvs
        stdhv = 1
        meancol = 'k'
    else:
        assert len(hvs.shape) == 2
        if plottype.lower() == 'meanstd':
            meanhv = hvs[0]
            stdhv = hvs[1]
            meancol = 'r'
        else:
            meanhv = np.exp(np.log(hvs).mean(0))
            stdhv = np.exp(np.log(hvs).std(0))
            meancol = 'k'
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    if plottype.lower() == 'array' or plottype.lower() == 'all':
        ax.plot(freq, hvs.T, linewidth=.8)
    if plottype.lower() == 'meanstd' or plottype.lower() == 'all':
        ax.plot(freq, meanhv * stdhv, '--k', linewidth=1.2)
        ax.plot(freq, meanhv / stdhv, '--k', linewidth=1.2)
    if plottype.lower() != 'array':
        ax.plot(freq, meanhv, color=meancol, linewidth=1.2)
    if log:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(ticker.LogLocator(10, 'all'))
        ax.xaxis.set_major_locator(ticker.LogLocator(10, [.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.xaxis.set_major_formatter('{x:.16g}')
    ax.set_ylim(0, np.max((meanhv * stdhv)[(1 <= freq) * (freq <= 15)]) * 1.1)
    ax.set_xlim(min(freq), max(freq))
    ax.grid()
    ax.set_ylabel('HVSR Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    if legend:
        ax.legend(legend)
    if title:
        plt.title(title)
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


def hv_azimuth(hv_az, azimuths, freq, vmax=None, title=None, cmap='jet', saveto=None):
    """Plot HVSR at multiple azimuths"""
    if len(hv_az.amp[0].shape) == 1:
        amp = np.array([hv for hv in hv_az.amp])
    else:
        amp = np.array([np.exp(np.log(hv).mean(0)) for hv in hv_az.amp])
    if vmax is None:
        vmax = amp[:, hv_az.hvsrs[0].valid_frequencies].max() * 1.1
    plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='polar')
    pat = ax.contourf(np.concatenate([azimuths, azimuths + np.pi, [np.pi * 2]]), freq,
                      np.row_stack([amp, amp, amp[0]]).T,
                      levels=np.arange(0, vmax, 0.5), cmap=cmap, extend='max', alpha=.9)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rscale('symlog')
    ax.set_rlim(0.2, 100)
    ax.set_rgrids([.5, 1, 2, 5, 10, 20, 50, 100], ['0.5Hz', '1Hz', '2Hz', '5Hz', '10Hz', '20Hz', '50Hz', '100Hz'])
    plt.colorbar(pat, spacing='proportional', format=ticker.ScalarFormatter())
    if title:
        plt.title(title)
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()
