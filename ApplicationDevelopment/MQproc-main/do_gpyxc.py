import os
import sys
from itertools import combinations
from pathlib import Path
from time import time

import numpy as np

from iotoolpy.ioGpy import AutocorrTarget, readMAX
from iotoolpy.paraGpy import parameters
from xcorpy.baseConfig import cfgupdate, cfgdata, cfgpara, cfgprep, cfgresl, bingpy, binspac, binfk
from xcorpy.prepFunc import getCP, prep_project, load_param, prep_location, prep_init
from xcorpy.coreFunc import update_prog, update_logs, print_line
from xcorpy.postFunc import post_gpypair
from xcorpy import plotFunc

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
    # load/prep parameters
    _, _, bandwidth, win_size, overlap, nf, sta, lta, atrig_range = load_param(proj_para, verbose=False)
    para_gpy = parameters()
    para_gpy.COMMON.MINIMUM_FREQUENCY = getCP(proj_para, 'freq', 'freq_from', 'float')
    para_gpy.COMMON.MAXIMUM_FREQUENCY = getCP(proj_para, 'freq', 'freq_to', 'float')
    if getCP(proj_para, 'freq', 'log', 'bool'):
        para_gpy.COMMON.SAMPLING_TYPE_FREQUENCY = 'Log'
    else:
        para_gpy.COMMON.SAMPLING_TYPE_FREQUENCY = 'Linear'
    para_gpy.COMMON.SAMPLES_NUMBER_FREQUENCY = getCP(proj_para, 'freq', 'freq_n', 'int')
    para_gpy.SPAC.FREQ_BAND_WIDTH = bandwidth
    para_gpy.FK.FREQ_BAND_WIDTH = bandwidth
    para_gpy.COMMON.PERIOD_COUNT = win_size
    para_gpy.COMMON.WINDOW_OVERLAP = overlap
    para_gpy.SPAC.STATISTIC_COUNT = nf
    para_gpy.FK.STATISTIC_COUNT = nf
    para_gpy.COMMON.RAW_STA = sta
    para_gpy.COMMON.RAW_LTA = lta
    para_gpy.COMMON.RAW_MIN_SLTA = atrig_range[0]
    para_gpy.COMMON.RAW_MAX_SLTA = atrig_range[1]
    if update:
        update_prog('Initialisation', [4, 6], 0, 'Loading project files')
    else:
        print('Parameters loaded')

    # --------------------------------------------------------------------------------
    # load sac files, prep gpydb
    siglst = prep_project(data_path, proj_recd, proj_para, update=update, geopsy=True)

    # --------------------------------------------------------------------------------
    # saving sac back-up (optional)
    if getCP(proj_para, 'basic', 'save_sac', 'bool'):
        if update:
            update_prog('Initialisation', [5, 6], 0, 'Pre-preped data back-up')
        else:
            print_line('Pre-preped data back-up')
        for i in range(len(siglst)):
            if update:
                update_prog('Back-up process', [(i + 1) * 0.99, len(siglst)], 0, siglst[i].MasterGroup)
            isig_path = prep_path.joinpath(siglst[i].MasterGroup)
            if not isig_path.exists():
                isig_path.mkdir()
            os.chdir(isig_path)
            for j in range(len(siglst[i].File)):
                tmpt0, tmpt1 = siglst[i].File[j].Signal.TimeRange.split(' ')
                with open(isig_path.joinpath('waveform.qs'), 'w') as f:
                    f.writelines(['setHeader(0, "Name","{}")\n'.format(siglst[i].File[j].Signal.Name),
                                  'setHeader(0, "Component","{}")\n'.format(siglst[i].File[j].Signal.Component),
                                  'setHeader(0, "ReceiverX",{})\n'.format(siglst[i].File[j].Signal.Receiver[0]),
                                  'setHeader(0, "ReceiverY",{})\n'.format(siglst[i].File[j].Signal.Receiver[1]),
                                  'setHeader(0, "ReceiverZ",{})\n'.format(siglst[i].File[j].Signal.Receiver[2]),
                                  'cut("[abs {}; abs {}]");\n'.format(tmpt0, tmpt1),
                                  'subtractValue(0);\n',
                                  'removeTrend();\n',
                                  'exportFile("GPY.{}.{}.{}.sac");'.format(siglst[i].MasterGroup,
                                                                           siglst[i].File[j].Signal.Name,
                                                                           siglst[i].File[j].Signal.Component)])
                os.system(' '.join([bingpy, siglst[i].File[j].name, '-waveform',
                                    str(isig_path.joinpath('waveform.qs')), '-nobugreport']))
            os.remove(str(isig_path.joinpath('waveform.qs')))
        if update:
            update_prog('Back-up process', [len(siglst), len(siglst)], 1, 'Complete')
        else:
            print('Done saving back-up!')

    # --------------------------------------------------------------------------------
    # start calculation loop
    if update:
        update_prog('Initialisation', [6, 6], 1, 'Complete')
        update_prog('Processing', [0, len(siglst)], 0, 'Start')
    else:
        print_line('Calculating with Geopsy')
    ttik = time()
    for i in range(len(siglst)):
        if update:
            update_prog('Processing', [(i + 1) * 0.99, len(siglst)], 0, siglst[i].MasterGroup)
            update_prog('Subprocess', [0, 4], 0, 'Pre-processing')

        # --------------------------------------------------------------------------------
        # saving gpy
        sigdb = siglst[i]
        tab = sigdb.toTab()
        coord = np.array([tab[np.unique(tab['ID'], True)[1]]['ReceiverX'],
                          tab[np.unique(tab['ID'], True)[1]]['ReceiverY']]).T
        if sigdb.MasterGroup == '':
            sigdb.MasterGroup = 'temp_{}'.format(i + 1)
        ioutpath = resl_path.joinpath(sigdb.MasterGroup)
        if not ioutpath.exists():
            ioutpath.mkdir()
        os.chdir(ioutpath)
        gpypath = str(ioutpath.joinpath(sigdb.MasterGroup + '.gpy'))
        sigdb.write(gpypath, verbose=False)

        # --------------------------------------------------------------------------------
        # get ring data from coordinates
        rings = prep_location(coord=np.column_stack([tab['ReceiverX'], tab['ReceiverY']]))[3]
        para_gpy.SPAC.RINGS = []
        for r in rings:
            para_gpy.SPAC.RINGS.append(r - 0.1)
            para_gpy.SPAC.RINGS.append(r + 0.1)

        # --------------------------------------------------------------------------------
        # transfer timerange to param
        if len(np.unique(tab['TimeRange0'])) == 1 and len(np.unique(tab['TimeRange1'])) == 1:
            para_gpy.COMMON.FROM_TIME_TYPE = 'Absolute'
            para_gpy.COMMON.FROM_TIME_TEXT = sigdb.File[0].Signal.TimeRange.split(' ')[0]
            para_gpy.COMMON.TO_TIME_TYPE = 'Absolute'
            para_gpy.COMMON.TO_TIME_TEXT = sigdb.File[0].Signal.TimeRange.split(' ')[1]

        # --------------------------------------------------------------------------------
        # write .log parameter file
        parafile_spac = str(ioutpath.joinpath('param_spac.log'))
        parafile_fk = str(ioutpath.joinpath('param_fk.log'))
        tarpath = str(ioutpath.joinpath('gpy_' + sigdb.MasterGroup + '.target'))
        maxpath = str(ioutpath.joinpath('gpy_' + sigdb.MasterGroup + '.max'))
        para_gpy.write(parafile_spac, 'spac', verbose=False)
        para_gpy.write(parafile_fk, 'fk', verbose=False)

        # --------------------------------------------------------------------------------
        # run geopsy command (SPAC + FK)
        if update:
            tik = time()
            update_prog('Subprocess', [1, 4], 0, 'Processing')
        # calculate SPAC
        os.system(' '.join([binspac, '-db', gpypath, '-group-pattern "All signals"', '-param', parafile_spac, '-o temp',
                            '-nobugreport']))
        os.system('mv "temp-All signals.target" ' + tarpath)
        # SPAC target file is read and re-written in ascii format
        if Path(tarpath).exists():
            results = AutocorrTarget(tarpath)
            results.write(tarpath, verbose=False)
            freq, spac, stddev, rings, weight = results.toSPAC()
        # calculate HRFK
        os.system(' '.join([binfk, '-db', gpypath, '-group-pattern "All signals"', '-param', parafile_fk, '-o temp',
                            '-nobugreport']))
        os.system('mv "All signals.max" ' + maxpath)
        if Path(maxpath).exists():
            tmpfk = readMAX(maxpath)
        if len(tmpfk) == 0:
            print('Empty F-K read! skipping F-K output...')
            if update:
                update_logs('PythonWarning', 'warning', 'Empty F-K read! skipping F-K output...')

        # --------------------------------------------------------------------------------
        # plot
        if update:
            update_logs('Calculation GPY_XC: {}'.format(sigdb.MasterGroup),
                        'Time lapsed: {:.3f}s'.format(time() - tik))
            update_prog('Subprocess', [2, 4], 0, 'Saving files')
        if getCP(proj_para, 'basic', 'save_png', 'bool'):
            plotFunc.location(coord, tab[np.unique(tab['ID'], True)[1]]['Name'].tolist(),
                              title='Location | {}'.format(sigdb.MasterGroup),
                              saveto=ioutpath.joinpath('location-{}.png'.format(sigdb.MasterGroup)))
            plog = getCP(proj_para, 'basic', 'plot_log', 'bool')
            if Path(tarpath).exists():
                plotFunc.spac_curve(freq, spac, spac.real + stddev, spac.real - stddev, rings,
                                    t0='[GPY] SPAC | {}'.format(sigdb.MasterGroup), log=plog,
                                    saveto=ioutpath.joinpath('gpy_spacfig-{}'.format(sigdb.MasterGroup)))
                disp_sum = plotFunc.spac2disp(spac, freq, rings, c0=np.linspace(101, 1200, 1100),
                                              shading='nearest', log=plog,
                                              saveto=ioutpath.
                                              joinpath('gpy_spac2disp-{}.png'.format(sigdb.MasterGroup)),
                                              t0='[GPY] SPAC to DISP | {}'.format(sigdb.MasterGroup))
            if Path(maxpath).exists() and len(tmpfk) != 0:
                plotFunc.gpy_fk(tmpfk, [(1, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 50)], lim=(8e-4, 6e-3),
                                title='[GPY] FKmax density | {}'.format(sigdb.MasterGroup),
                                saveto=ioutpath.joinpath('gpy_fkmd-{}.png'.format(sigdb.MasterGroup)))
                fkf, fkv, fkm = plotFunc.gpy_fkmd2disp(tmpfk, plog, 'GPY_FKmax to DISP | {}'.format(sigdb.MasterGroup),
                                                       saveto=ioutpath.
                                                       joinpath('gpy_fkmd2disp-{}.png'.format(sigdb.MasterGroup)))
            if update:
                update_logs('Output XCOR: {}'.format(sigdb.MasterGroup), 'SPAC/FK fig saved')
            else:
                print('SPAC/FK fig saved!')
        else:
            if len(tmpfk) != 0:
                fkf, fkv, fkm = plotFunc.gpy_fkmd2disp(tmpfk)
            disp_sum = plotFunc.spac2disp(spac, freq, rings, c0=np.linspace(101, 1200, 1100), plot=False)

        # --------------------------------------------------------------------------------
        # csv data output
        if getCP(proj_para, 'basic', 'save_csv', 'bool'):
            # save SPAC by rings
            for ir in range(len(rings)):
                tmphead = 'SPatial Auto-Correlation [GPY]\n' \
                          'Location: {}\n' \
                          'Ring ID: {:.0f} of {:.0f}\n' \
                          'Ring radius(m): {:.2f}\n' \
                          'Column names:\n' \
                          'Freq, Real, Imag, Std, Up, Low' \
                    .format(sigdb.MasterGroup, ir + 1, len(rings), rings[ir])
                np.savetxt(str(ioutpath.joinpath('gpy_spac-{}[{}]{:.0f}cm.csv'
                                                 .format(sigdb.MasterGroup, ir + 1, rings[ir] * 1e2))),
                           np.vstack([freq, spac[ir].real, spac[ir].imag, stddev[ir],
                                      spac[ir].real + stddev[ir], spac[ir].real - stddev[ir]]).T,
                           fmt='%.5f', delimiter=', ', header=tmphead, comments='# ')
            # save SPAC2DISP
            tmphead = 'SPatial Auto-Correlation to Dispersion curve\n' \
                      'Location: {}\n' \
                      'First column is frequency(Hz), others are misfits at each velocity(m/s):\n' \
                          .format(sigdb.MasterGroup) \
                      + ', '.join(['freq'] + [str(x) for x in np.linspace(101, 1200, 1100)])
            np.savetxt(str(ioutpath.joinpath('gpy_spac2disp-{}.csv'.format(sigdb.MasterGroup))),
                       np.vstack([freq, disp_sum]).T,
                       fmt='%.5e', delimiter=',', comments='', header=tmphead)
            # save FK2DISP
            if len(tmpfk) != 0:
                fkfreq = (fkf[1:] + fkf[:-1]) / 2
                fkvel = (fkv[1:] + fkv[:-1]) / 2
                tmphead = 'F-K Beampower to Dispersion curve\n' \
                          'Location: {}\n' \
                          'First column is frequency(Hz), others are beampower at each velocity(m/s):\n' \
                              .format(sigdb.MasterGroup) \
                          + ', '.join(['freq'] + [format(x, '.2f') for x in fkvel])
                np.savetxt(str(ioutpath.joinpath('gpy_fkmd2disp-{}.csv'.format(sigdb.MasterGroup))),
                           np.vstack([fkfreq, fkm]).T,
                           fmt='%.5e', delimiter=', ', newline='\n', comments='# ', header=tmphead)
            if update:
                update_logs('Output XCOR: {}'.format(sigdb.MasterGroup), 'SPAC/FK data saved')
            else:
                print('data csv saved!')

        # --------------------------------------------------------------------------------
        # single pair xcorrelation (optional)
        if getCP(proj_para, 'basic', 'single_pair', 'bool'):  # proj_para.getboolean('basic', 'single_pair'):
            if update:
                update_prog('Subprocess', [3, 4], 1, 'Calculating single-pair')
                update_prog('Single-pair XCOR',
                            [0, len(list(combinations(range(max([int(x) for x in tab['ID']])), 2)))], 0,
                            'Startting {}'.format(sigdb.MasterGroup))
            ipairpath = ioutpath.joinpath('pair')
            if not ipairpath.exists():
                ipairpath.mkdir()
            post_gpypair(sigdb, ipairpath, para_gpy, getCP(proj_para, 'basic', 'save_png', 'bool'),
                         getCP(proj_para, 'basic', 'plot_log', 'bool'), getCP(proj_para, 'basic', 'save_csv', 'bool'),
                         update)
        if update:
            update_prog('Subprocess', [4, 4], 1, 'Complete')
    if update:
        update_prog('Processing', [len(siglst), len(siglst)], 1,
                    'All processes complete! Total time: {:.2f}s'.format(time() - ttik))
    else:
        print_line('Summary')
        print('Total time: {:.2f}s'.format(time() - ttik))
        print('Program GPY_XCOR finished!')


if __name__ == '__main__':
    try:
        argv = sys.argv[1]
        print('Input proj: {}'.format(argv))
        calc(argv)
    except Exception as e:
        raise e
