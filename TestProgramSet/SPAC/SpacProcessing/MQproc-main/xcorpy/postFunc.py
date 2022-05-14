import os
import warnings
from functools import partial
from time import time
import multiprocessing as mp

import numpy as np
from scipy.stats import stats
from pathlib import Path
from obspy import Stream
from progressbar import ProgressBar, Bar, Percentage
from itertools import combinations

from tqdm import tqdm

from iotoolpy.ioGpy import SignalDatabase, AutocorrTarget
from xcorpy.baseConfig import binspac
from xcorpy.coreFunc import update_prog, update_logs, array_matrix, projnorm, terminal_size
from xcorpy.plotFunc import spac_single_pair


def post_spac(result, freq, projection, ring_mat, ring_num, ring_grp, ring_nps, ring_azs):
    """Post-processing for SPAC of different rings

    Parameters
    ----------
    result: list of np.ndarray
        Normalized zero-shift cross-spectrum (all frequency/station/window)
    freq: np.ndarray
        Central frequencies
    projection: str
        Type of projection when calculating window means ['none', 'tanh', 'csgp']
    ring_mat: np.ndarray
        Inter-station distance matrix
    ring_num: np.ndarray
        Number of ring groups
    ring_grp: np.ndarray
        Radius of each ring group
    ring_nps: np.ndarray
        Number of pairs per ring group
    ring_azs: np.ndarray
        Azimuth of each pair
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        SPAC, STD, lower bound, upper bound
    """
    # create empty matrix for spac/std output
    spac = np.zeros([ring_num, len(freq)], dtype=complex)
    std = np.zeros([ring_num, len(freq)], dtype=float)
    # maxdev = np.zeros([ring_num, len(freq)], dtype=float)
    upbd = np.zeros([ring_num, len(freq)], dtype=float)
    lwbd = np.zeros([ring_num, len(freq)], dtype=float)
    for i in range(len(freq)):
        if result[i].shape[2] != 0:
            u_res = result[i][np.tril_indices(len(result[i]))]
            u_rad = ring_mat[np.tril_indices_from(ring_mat)]
            u_azs = ring_azs[np.tril_indices_from(ring_azs)]
            u_sel = [u_rad == ring_grp[j] for j in range(ring_num)]
            for j in range(ring_num):
                sel_rad = np.unique(u_rad[u_sel[j]])
                assert len(sel_rad) == 1
                sel_res = u_res[u_sel[j]]
                sel_azs = u_azs[u_sel[j]]
                sort_id = sel_azs.argsort()
                sel_res = sel_res[sort_id]
                sel_azs = sel_azs[sort_id]
                _sel_azs = np.copy(sel_azs)
                _sel_azs[0] += 360
                sel_mid = (sel_azs + np.roll(_sel_azs, -1)) / 2
                weights = sel_mid - np.roll(sel_mid, 1)
                weights[weights <= 0] += 360
                tmpspac = (weights[:, None] * sel_res).sum(0) / weights.sum()
                # calc prjt'd mean, std & bounds
                tmpmean, tmplower, tmpupper = projnorm(tmpspac, proj=projection)
                spac[j, i] = tmpmean
                std[j, i] = (tmpupper - tmplower) / 2
                upbd[j, i] = tmpupper
                lwbd[j, i] = tmplower
                # maxdev[j, i] = max(abs(temp - csgp_b(prj_mean)))
        else:
            spac[:, i] = np.nan + np.nan * 1j
            upbd[:, i] = np.nan
            lwbd[:, i] = np.nan
            continue
    return spac, std, upbd, lwbd


def post_pair(sig0, result, rings, freq, idetailpath, projection='csgp', plot=True, plog=True, csv=True, update=False):
    """Output single-pair cross-correaltion results

    Parameters
    ----------
    sig0: Stream
        Obspy signal information
    result: list of np.ndarray
        Normalized zero-shift cross-spectrum (all frequency/station/window)
    rings: list
        Ring information produced by prepFunc.prep_location
    freq: np.ndarray
        Central frequencies
    idetailpath: Path
        Output path of all pair results
    projection: str
        Type of projection when calculating window means ['none', 'tanh', 'csgp']
    plot: bool
    plog: bool
        If the plot frequency axis is in log scale
    csv: bool
    update: bool
        If the progress is updated to coop_fetch
    """
    # export all single pairs
    cormat = np.empty(shape=(len(sig0), len(sig0), len(result)), dtype=complex)
    stdmat = np.empty(shape=(len(sig0), len(sig0), len(result)), dtype=float)
    uprmat = np.empty(shape=(len(sig0), len(sig0), len(result)), dtype=float)
    lowmat = np.empty(shape=(len(sig0), len(sig0), len(result)), dtype=float)
    for i in range(len(result)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tmpmean, tmplower, tmpupper = projnorm(result[i], axis=2, proj=projection)
            cormat[..., i] = tmpmean
            uprmat[..., i] = tmpupper
            lowmat[..., i] = tmplower
            stdmat[..., i] = (tmpupper - tmplower) / 2
    if update:
        icount = 0
    else:
        print('Saving single-pair results...')
    pbar = ProgressBar(widgets=['Saving: ', Percentage(), Bar('█')], maxval=len(sig0)).start()
    for ipair in pbar(list(combinations(range(len(sig0)), 2))):
        j = ipair[0]
        k = ipair[1]
        ispac = cormat[j, k]
        istd = stdmat[j, k]
        iup = uprmat[j, k]
        ilow = lowmat[j, k]
        iname = '{}[{}-{}]'.format(sig0[0].stats.location, sig0[j].stats.station, sig0[k].stats.station)
        iring = rings[0][j, k]
        if update:
            update_prog('Single-pair XCOR', [icount * 0.99, len(list(combinations(range(len(sig0)), 2)))], 0, iname)
            icount += 1
        # saving single-pair data
        itarget = AutocorrTarget()
        itarget.fromSPAC(freq, ispac[None, :], (iup[None, :] - ilow[None, :]) / 2, [iring])
        itarget.write(str(idetailpath.joinpath('pair-{}{:.0f}cm.target'.format(iname, iring * 100))), verbose=False)
        if plot:
            spac_single_pair(freq, ispac, iup, ilow, iname, iring, log=plog,
                             ipath=str(idetailpath.joinpath('pair-{}{:.0f}cm.png'.format(iname, iring * 100))))
        if csv:
            head = 'Single-pair corss-correlation\n' \
                   'Location: {}\n' \
                   'Station 1 [{}] coordinates(m): ({:.2f}, {:.2f}, {:.2f})\n' \
                   'Station 2 [{}] coordinates(m): ({:.2f}, {:.2f}, {:.2f})\n' \
                   'Ring radius(m): {:.2f}\n' \
                   'Column names:\n' \
                   'Freq, Real, Imag, Std, Up, Low' \
                .format(sig0[0].stats.location,
                        sig0[j].stats.station, sig0[j].stats.sac.user7, sig0[j].stats.sac.user8,
                        sig0[j].stats.sac.user9,
                        sig0[k].stats.station, sig0[k].stats.sac.user7, sig0[k].stats.sac.user8,
                        sig0[k].stats.sac.user9,
                        iring)
            np.savetxt(str(idetailpath.joinpath('pair-{}{:.0f}cm.csv'.format(iname, iring * 100))),
                       np.vstack([freq, ispac.real, ispac.imag, istd, iup, ilow]).T,
                       fmt='%.3f', delimiter=', ', newline='\n', header=head, comments='# ')
    pbar.finish()
    if update:
        update_prog('Single-pair XCOR',
                    [len(list(combinations(range(len(sig0)), 2))), len(list(combinations(range(len(sig0)), 2)))], 1,
                    'Complete')
    else:
        print('Done saving {} single-pair records!'.format(len(list(combinations(range(len(sig0)), 2)))))


def post_gpypair(sigdb, idetailpath, parameter, plot=True, plog=True, csv=True, update=False):
    """Calculate single-pair cross-correaltion with Geopsy

    Parameters
    ----------
    sigdb: SignalDatabase
    idetailpath: Path
    parameter
    plot: bool
    plog: bool
    csv: bool
    update: bool
    """
    npair = len(list(combinations(sigdb.File, 2)))
    para_tmp = parameter
    if update:
        icount = 0
    else:
        print('Saving single-pair results...')
    for ipair in list(combinations(sigdb.File, 2)):
        subsig = SignalDatabase()
        subsig.File = ipair
        acoord = np.array(subsig.File[0].Signal.Receiver)
        bcoord = np.array(subsig.File[1].Signal.Receiver)
        ir = np.sqrt(((acoord - bcoord) ** 2).sum())
        subtxt = '{}[{}-{}]'.format(sigdb.MasterGroup, subsig.File[0].Signal.Name, subsig.File[1].Signal.Name)
        parafile_tmp = str(idetailpath.joinpath('param_spac_tmp.log'))
        para_tmp.SPAC.RINGS = [ir - 0.1, ir + 0.1]
        para_tmp.write(parafile_tmp, 'spac')
        if update:
            icount += 1
            update_prog('Single-pair XCOR', [icount * 0.99, npair], 0, subtxt)
        subgpy = str(idetailpath.joinpath('{}{:.0f}cm.gpy'.format(subtxt, ir * 100)))
        subtar = str(idetailpath.joinpath('gpy_pair-{}{:.0f}cm.target'.format(subtxt, ir * 100)))
        subsig.write(subgpy, verbose=False)
        os.system(' '.join([binspac, '-db', subgpy, '-group-pattern "All signals"', '-param', parafile_tmp,
                            '-nobugreport']))
        os.system('mv "a-All signals.target" ' + subtar)
        try:
            irest = AutocorrTarget(subtar)
        except FileNotFoundError:
            warnings.warn('Failed to calculate single-pair {}! Skipping output...'.format(subtxt))
            if update:
                update_logs('PythonWarning', 'warning',
                            'Failed to calculate single-pair {}! Skipping output...'.format(subtxt))
            continue

        irest.write(subtar, verbose=False)
        ifreq, ispac, istd, iring, iweight = irest.toSPAC()

        # plot single-pair results
        if plot:
            spac_single_pair(ifreq, ispac[0], ispac[0].real + istd[0], ispac[0].real - istd[0], subtxt, ir, log=plog,
                             ipath=str(idetailpath.joinpath('gpy_pair-{}{:.0f}cm.png'.format(subtxt, ir * 100))))

        # single-pair csv data output
        if csv:
            head = 'Single-pair corss-correlation [GPY]\n' \
                   'Location: {}\n' \
                   'Station 1 [{}] coordinates(m): ({:.2f}, {:.2f}, {:.2f})\n' \
                   'Station 2 [{}] coordinates(m): ({:.2f}, {:.2f}, {:.2f})\n' \
                   'Ring radius(m): {:.2f}\n' \
                   'Column names:\n' \
                   'Freq, Real, Imag, Std, Up, Low' \
                .format(subsig.MasterGroup,
                        subsig.File[0].Signal.Name, subsig.File[0].Signal.Receiver[0],
                        subsig.File[0].Signal.Receiver[1], subsig.File[0].Signal.Receiver[2],
                        subsig.File[1].Signal.Name, subsig.File[1].Signal.Receiver[0],
                        subsig.File[1].Signal.Receiver[1], subsig.File[1].Signal.Receiver[2],
                        ir)
            np.savetxt(str(idetailpath.joinpath('gpy_pair-{}{:.0f}cm.csv'.format(subtxt, ir * 100))),
                       np.vstack([ifreq, ispac.real, ispac.imag, istd, ispac.real + istd, ispac.real + istd]).T,
                       fmt='%.3f', delimiter=', ', newline='\n', header=head, comments='# ')
    if update:
        update_prog('Single-pair XCOR', [npair, npair], 1, 'Finished {}'.format(sigdb.MasterGroup))


def post_hrfk(result, freq, coord, s_min=1e-4, s_max=1e-2, s_n=200, theta_n=181, damp=0, method=1, upd=True, mc=False):
    """Post-processing for (High-Resolution) FK

    # TODO: This function calculates beampower maximum from the window-averaged result,
        yet the window-wise beampower calculation is of poor efficiency.
        Need to write a new method to calculate window-wise beampower maximum.

    Parameters
    ----------
    result: list[np.ndarray]
        Pre-normalized zero-shift cross-spectrum (all frequency/station/window)
    freq: np.ndarray
        Central frequencies
    coord: np.ndarray
        Station coordinates
    s_min: float
        Minimum slowness
    s_max: float
        Maximum slowness
    s_n: int
        Number of slowness
    theta_n: int
        Number of theta
    damp: int
        Size of damping matrix
    method: int
        FK method, 0 for conventional, 1 for high resolution
    upd: bool
        If update to interface
    mc: bool
        If use multicore

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        FK beampower, slowness
    """
    assert len(result) == len(freq)
    slow = np.linspace(s_min, s_max, s_n)
    if mc:
        # multicore calc cross-spectrum
        tik = time()
        at_loop = partial(array_transfer, result=result, freq=freq, coord=coord,
                          s_min=s_min, s_max=s_max, s_n=s_n, theta_n=theta_n, damp=damp, method=method, upd=upd)
        mp.freeze_support()  # for Windows support
        NB_PROCESSES = round(mp.cpu_count() * 0.7) - 1
        if upd:
            update_prog('FK loop', [0, len(freq)], 0, 'Calculating FK')  # TODO: change this to frequency index
        else:
            print('The processing will take up {}/{} of the CPU'.format(NB_PROCESSES, mp.cpu_count()))
        pool = mp.Pool(NB_PROCESSES)
        bps = pool.map(at_loop, tqdm(range(len(result)), ncols=terminal_size()[0]))
        pool.close()
        bp = np.array(bps)
        if upd:
            update_prog('FK loop', [len(freq), len(freq)], 1, 'Complete')
        print('Time lapsed: {:.2f}s'.format(time() - tik))
    else:
        tik = time()
        bp = np.zeros(shape=(len(freq), 181, s_n))
        for i in range(len(freq)):
            print(f'Calculating {i + 1}/{len(freq)}, freq={freq[i]}Hz\r')
            bp[i] = array_transfer(i, result=result, freq=freq, coord=coord,
                                   s_min=s_min, s_max=s_max, s_n=s_n, theta_n=theta_n, damp=damp, method=method)
        print('Time lapsed: {:.2f}s'.format(time() - tik))
    return bp, slow


def array_transfer(i, result, freq, coord, s_min=1e-4, s_max=1e-2, s_n=200, theta_n=181, damp=0, method=1, upd=False):
    """Function for batch FK calculation

    Parameters
    ----------
    i: int
    result: list[np.ndarray]
    freq: np.ndarray
        Central frequencies
    coord: np.ndarray
        Station coordinates
    s_min: float
        Minimum slowness
    s_max: float
        Maximum slowness
    s_n: int
        Number of slowness
    theta_n: int
        Number of theta
    damp: int
        Size of damping matrix
    method: int
        FK method, 0 for conventional, 1 for high resolution
    upd: bool
        If update to interface

    Returns
    -------

    """
    if method == 2:
        bp = np.zeros(shape=(len(coord), len(coord), s_n))
    else:
        bp = np.zeros(shape=(181, s_n))
    if not result[i].size == 0:
        cormat = result[i]
        emat = array_matrix(freq[i], coord, s_min, s_max, s_n, theta_n)
        if method == 0:
            bp = (cormat.mean(-1)[:, :, None, None] * emat).sum((0, 1)).real
        elif method == 1:
            invmat = np.linalg.lstsq(np.vstack((cormat.mean(-1), damp * np.identity(cormat.shape[0]))),
                                     np.vstack((np.identity(cormat.shape[0]), np.zeros(cormat.shape[:2]))),
                                     rcond=None)[0]
            bp = 1 / (invmat[:, :, None, None] * emat).sum((0, 1)).real
        elif method == 2:
            bp = (cormat.mean(-1)[:, :, None, None] * emat).max(2).real
            '''
            # optional inversion matrix calculation method
            invmat = np.linalg.pinv(cormat.mean(-1))
            # window-wise calculation
            if len(cormat.shape) == 2:
                cormat = cormat[..., None]
            invmat = np.zeros(shape=cormat.shape, dtype=complex)
            for j in range(cormat.shape[2]):
                invmat[..., j] = np.linalg.pinv(cormat[..., j])
            '''
        else:
            raise Exception('Illegal FK method!')
        if upd:
            update_prog('FK loop', [i, len(freq)], 0, 'Freq = {:.3f} Hz'.format(freq[i]))
    return bp


def compare_curve(spac, freq, page, sel=None):
    """Depreciated: comparing spac result to Geopsy"""
    if sel is None:
        sel = np.ones(len(freq), bool)
    res = np.zeros(len(spac))
    c2 = np.zeros((len(spac)))
    for i in range(len(spac)):
        a = spac[i].real[sel]
        b = np.interp(freq, page[i].freq, page[i].spac)[sel]
        res[i] = np.sqrt(sum((a - b) ** 2) / sel.sum())
        temp = stats.pearsonr(a, b)
        c2[i] = temp[0] if temp[1] < 0.05 else 0
    return res, c2
