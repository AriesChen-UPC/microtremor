import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from time import time

import numpy as np
from progressbar import ProgressBar, Bar, Percentage
from tqdm import tqdm

from iotoolpy.ioGpy import AutocorrTarget
from xcorpy.baseConfig import cfgupdate, cfgdata, cfgpara, cfgprep, cfgresl
from xcorpy.prepFunc import load_param, prep_project, getCP, prep_atrig, prep_location, prep_init
from xcorpy.coreFunc import terminal_size, update_prog, update_logs, print_line, xcor1freq
from xcorpy.postFunc import post_pair, post_spac, post_hrfk
from xcorpy import plotFunc

# import config
update = cfgupdate
data_root = Path(cfgdata)
para_root = Path(cfgpara)
prep_root = Path(cfgprep)
resl_root = Path(cfgresl)


def calc(proj):
    # --------------------------------------------------------------------------------
    # path setup
    data_path, prep_path, resl_path, proj_recd, proj_para = prep_init(proj, data_root, para_root, prep_root, resl_root,
                                                                      update)

    # --------------------------------------------------------------------------------
    # load parameters
    dt, freq, bandwidth, win_size, overlap, nf, sta, lta, atrig_range = load_param(proj_para, verbose=False)
    freq_ind = range(len(freq))
    if update:
        update_prog('Initialisation', [4, 6], 0, 'Loading project files')
    else:
        print('Parameters loaded')

    # --------------------------------------------------------------------------------
    # load sac files
    siglst = prep_project(data_path, proj_recd, proj_para, update=update)

    # --------------------------------------------------------------------------------
    # saving sac back-up (optional)
    if getCP(proj_para, 'basic', 'save_sac', 'bool'):
        if update:
            update_prog('Initialisation', [5, 6], 0, 'Pre-preped data back-up')
        else:
            print_line('Pre-preped data back-up')
            pbar = ProgressBar(widgets=['Saving: ', Percentage(), Bar('█')], maxval=len(siglst)).start()
        for i in range(len(siglst)):
            if update:
                update_prog('Back-up process', [(i + 1) * 0.99, len(siglst)], 0, siglst[i][0].stats.location)
            else:
                pbar.update(i + 1)
            if not prep_path.joinpath(siglst[i][0].stats.location).exists():
                prep_path.joinpath(siglst[i][0].stats.location).mkdir()
            for j in range(len(siglst[i])):
                jid = siglst[i][j].id if siglst[i][j].id[0] != '.' else siglst[i][j].id[1:]
                siglst[i][j].write(str(prep_path.joinpath(siglst[i][0].stats.location, jid + '.sac')))
        if update:
            update_prog('Back-up process', [len(siglst), len(siglst)], 1, 'Complete')
        else:
            pbar.finish()
            print('Done saving back-up')

    # --------------------------------------------------------------------------------
    # start calculation loop
    if update:
        update_prog('Initialisation', [6, 6], 1, 'Complete')
        update_prog('Processing', [0, len(siglst)], 0, 'Start')
    else:
        print_line('Start Calculation')
    ttik = time()
    for i in range(len(siglst)):

        # --------------------------------------------------------------------------------
        # init
        sig0 = siglst[i]
        iname = sig0[0].stats.location
        ioutpath = resl_path.joinpath(iname)
        if not ioutpath.exists():
            ioutpath.mkdir()

        # --------------------------------------------------------------------------------
        # pre-processing
        if update:
            update_prog('Processing', [(i + 1) * 0.99, len(siglst)], 0, sig0[0].stats.location)
            update_prog('Subprocess', [0, 5], 0, 'Pre-processing')
        else:
            print_line('Processing {} ({}/{})'.format(sig0[0].stats.location, i + 1, len(siglst)))
        atrig = prep_atrig(sig0, dt, sta, lta)
        coord, *rings = prep_location(sig0)

        # --------------------------------------------------------------------------------
        # pre-plotting
        if getCP(proj_para, 'basic', 'save_png', 'bool'):
            if update:
                update_prog('Subprocess', [1, 5], 0, 'Plotting figures')
            else:
                print('Plotting figures...')
            plog = getCP(proj_para, 'basic', 'plot_log', 'bool')
            plotFunc.location(sig0, title='Location | {}'.format(iname),
                              saveto=ioutpath.joinpath('location-{}.png'.format(iname)))
            plotFunc.signals(sig0, None, atrig, saveto=ioutpath.joinpath('signal-{}.png'.format(iname)))
            plotFunc.psd(sig0, 1024, 128, saveto=ioutpath.joinpath('psd-{}.png'.format(iname)), plog=plog,
                         csvto=ioutpath.joinpath('psd-{}.csv'.format(iname)))
            # plotFunc.stft(sig0, 512, 64, saveto=ioutpath.joinpath('spectrum-{}'.format(iname)))

        # --------------------------------------------------------------------------------
        # multicore calc cross-spectrum
        tik = time()
        xcor_loop = partial(xcor1freq, freqs=freq, sig0=sig0, dt=dt, bandwidth=bandwidth, win_size=win_size,
                            overlap=overlap, nf=nf, atrig=atrig, atrig0=atrig_range[0], atrig1=atrig_range[1],
                            verbose=False, update=update, do_dtft=False)
        mp.freeze_support()  # for Windows support
        NB_PROCESSES = round(mp.cpu_count() * 0.7) - 1
        if update:
            update_prog('Subprocess', [2, 5], 0, 'Calculating cross-correlations')
        else:
            print('The processing will take up {}/{} of the CPU'.format(NB_PROCESSES, mp.cpu_count()))
        pool = mp.Pool(NB_PROCESSES)
        result = pool.map(xcor_loop, tqdm(freq_ind, ncols=terminal_size()[0]))
        pool.close()
        if update:
            update_prog('XCOR loop', [len(freq), len(freq)], 1, 'Complete')

        # --------------------------------------------------------------------------------
        # post-processing
        if update:
            update_prog('Subprocess', [3, 5], 0, 'Post-processing')
        else:
            print('Post-processing...')
        spac, stderr, upperbd, lowerbd = post_spac([temp[0] for temp in result], freq, 'csgp', *rings)
        fkbp, slow = post_hrfk([temp[1] for temp in result], freq, coord, 8e-4, 1e-2, 300, upd=update, mc=True)
        if update:
            update_logs('Calculation XCOR: {}'.format(sig0[0].stats.location),
                        'Time lapsed: {:.3f} s'.format(time() - tik))
            update_prog('Subprocess', [4, 5], 0, 'saving files')
        else:
            print('Done calculating SPAC / FK!\n  Time lapsed: {:.2f}s'.format(time() - tik))

        # --------------------------------------------------------------------------------
        # post-plotting
        plog = getCP(proj_para, 'basic', 'plot_log', 'bool')
        if getCP(proj_para, 'basic', 'save_png', 'bool'):
            plotFunc.spac_curve(freq, spac, upperbd, lowerbd, rings[2], t0='SPAC | {}'.format(sig0[0].stats.location),
                                log=plog, saveto=ioutpath.joinpath('spacfig-{}'.format(iname)))
            plotFunc.fk_imaging(fkbp, freq, [(1, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 50)], lim=(8e-4, 6e-3),
                                k=slow, title='FK beampower | {}'.format(sig0[0].stats.location),
                                saveto=ioutpath.joinpath('fkbp-{}.png'.format(iname)))
            disp_sum = plotFunc.spac2disp(spac, freq, rings[2], log=plog, shading='nearest', norm=None,
                                          c0=np.linspace(101, 1200, 1100),
                                          saveto=ioutpath.joinpath('spac2disp-{}.png'.format(iname)),
                                          t0='SPAC to DISP | {}'.format(sig0[0].stats.location))
            disp_max = plotFunc.fkbp2disp(fkbp, freq, slow, log=plog, shading='nearest', normal=True,
                                          saveto=ioutpath.joinpath('fkbp2disp-{}.png'.format(iname)),
                                          t0='FKmax to DISP | {}'.format(sig0[0].stats.location))
            if update:
                update_logs('Output XCOR: {}'.format(sig0[0].stats.location), 'SPAC / FK fig saved')
            else:
                print('SPAC / FK fig saved!')
        else:
            disp_sum = plotFunc.spac2disp(spac, freq, rings[2], c0=np.linspace(101, 1200, 1100), plot=False)
            disp_max = plotFunc.fkbp2disp(fkbp, freq, slow, plot=False)

        # --------------------------------------------------------------------------------
        # csv data output
        if getCP(proj_para, 'basic', 'save_csv', 'bool'):
            # save SPAC by rings
            for ir in range(len(rings[2])):
                tmphead = 'SPatial Auto-Correlation\n' \
                          'Location: {}\n' \
                          'Ring ID: {:.0f} of {:.0f}\n' \
                          'Ring radius(m): {:.2f}\n' \
                          'Column names:\n' \
                          'Freq, Real, Imag, Std, Up, Low' \
                    .format(sig0[0].stats.location, ir + 1, len(rings[2]), rings[2][ir])
                np.savetxt(str(ioutpath.joinpath('spac-{}[{}]{:.0f}cm.csv'
                                                 .format(sig0[0].stats.location, ir + 1, rings[2][ir] * 1e2))),
                           np.vstack([freq, spac[ir].real, spac[ir].imag, stderr[ir], upperbd[ir], lowerbd[ir]]).T,
                           fmt='%.5f', delimiter=', ', newline='\n', header=tmphead, comments='# ')
            # save SPAC target
            target = AutocorrTarget()
            target.fromSPAC(freq, spac, stderr, rings[2])
            target.write(str(ioutpath.joinpath(sig0[0].stats.location + '.target')), verbose=False)
            # save SPAC2DISP
            tmphead = 'SPatial Auto-Correlation to Dispersion curve\n' \
                      'Location: {}\n' \
                      'First column is frequency(Hz), others are misfits at each velocity(m/s):\n' \
                          .format(sig0[0].stats.location) \
                      + ', '.join(['freq'] + [str(x) for x in np.linspace(101, 1200, 1100)])
            np.savetxt(str(ioutpath.joinpath('spac2disp-{}.csv'.format(sig0[0].stats.location))),
                       np.vstack([freq, disp_sum]).T,
                       fmt='%.5e', delimiter=', ', newline='\n', comments='# ', header=tmphead)
            # save FK2DISP
            tmphead = 'F-K Beampower to Dispersion curve\n' \
                      'Location: {}\n' \
                      'First column is frequency(Hz), others are beampower at each velocity(m/s):\n' \
                          .format(sig0[0].stats.location) \
                      + ', '.join(['freq'] + [format(x, '.2f') for x in 1 / slow])
            np.savetxt(str(ioutpath.joinpath('fkbp2disp-{}.csv'.format(sig0[0].stats.location))),
                       np.vstack([freq, disp_max]).T,
                       fmt='%.5e', delimiter=', ', newline='\n', comments='# ', header=tmphead)
            if update:
                update_logs('Output XCOR: {}'.format(sig0[0].stats.location), 'SPAC / FK data saved')
            else:
                print('SPAC / FK data saved!')

        # --------------------------------------------------------------------------------
        # single-pair output
        if getCP(proj_para, 'basic', 'single_pair', 'bool'):
            ipairpath = ioutpath.joinpath('pair')
            if not ipairpath.exists():
                ipairpath.mkdir()
            post_pair(sig0, [temp[0] for temp in result], rings, freq, ipairpath, 'csgp',
                      getCP(proj_para, 'basic', 'save_png', 'bool'), plog,
                      getCP(proj_para, 'basic', 'save_csv', 'bool'), update)

        # --------------------------------------------------------------------------------
        # end-loop process
        if update:
            update_prog('Subprocess', [5, 5], 1, 'Complete')
    if update:
        update_prog('Processing', [len(siglst), len(siglst)], 1,
                    'All processes complete! Total time: {:.2f}s'.format(time() - ttik))
    else:
        print_line('Summary')
        print('Total time: {:.2f}s'.format(time() - ttik))
        print('Program XCOR finished!')


if __name__ == '__main__':
    try:
        argv = sys.argv[1]
        print('Input proj: {}'.format(argv))
        calc(argv)
    except Exception as e:
        raise e
