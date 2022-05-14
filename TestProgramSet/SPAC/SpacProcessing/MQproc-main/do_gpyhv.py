import os
import sys
from pathlib import Path
from time import time

import numpy as np

from iotoolpy.ioGpy import HVfile
from iotoolpy.paraGpy import parameters
from xcorpy.baseConfig import cfgupdate, cfgdata, cfgpara, cfgprep, cfgresl, bingpy, binhvsr
from xcorpy.prepFunc import getCP, prep_project, load_param, prep_init
from xcorpy.coreFunc import update_prog, update_logs, print_line
from xcorpy import plotFunc

update = cfgupdate
data_root = Path(cfgdata)
para_root = Path(cfgpara)
prep_root = Path(cfgprep)
resl_root = Path(cfgresl)


def calc(proj):
    # --------------------------------------------------------------------------------
    # path setup
    data_path, prep_path, resl_path, proj_recd, proj_para = prep_init(proj, data_root, para_root, prep_root, resl_root, update)

    # --------------------------------------------------------------------------------
    # load parameters
    _, _, bandwidth, win_size, overlap, nf, sta, lta, atrig_range = load_param(proj_para, verbose=False)
    para_gpy = parameters()
    para_gpy.COMMON.MINIMUM_FREQUENCY = getCP(proj_para, 'freq', 'freq_from', 'float')
    para_gpy.COMMON.MAXIMUM_FREQUENCY = getCP(proj_para, 'freq', 'freq_to', 'float')
    if getCP(proj_para, 'freq', 'log', 'bool'):
        para_gpy.COMMON.SAMPLING_TYPE_FREQUENCY = 'Log'
    else:
        para_gpy.COMMON.SAMPLING_TYPE_FREQUENCY = 'Linear'
    para_gpy.COMMON.SAMPLES_NUMBER_FREQUENCY = getCP(proj_para, 'freq', 'freq_n', 'int')
    # para_gpy.COMMON.FREQ_BAND_WIDTH = bandwidth
    para_gpy.COMMON.WINDOW_LENGTH_TYPE = 'AtLeast'
    para_gpy.COMMON.WINDOW_MIN_LENGTH = getCP(proj_para, 'hvsr', 'win_len', 'float')
    para_gpy.COMMON.WINDOW_MAX_LENGTH = getCP(proj_para, 'hvsr', 'win_len', 'float') * 2
    para_gpy.COMMON.WINDOW_OVERLAP = overlap
    # para_gpy.COMMON.STATISTIC_COUNT = nf
    para_gpy.COMMON.RAW_STA = sta
    para_gpy.COMMON.RAW_LTA = lta
    para_gpy.COMMON.RAW_MIN_SLTA = atrig_range[0]
    para_gpy.COMMON.RAW_MAX_SLTA = atrig_range[1]
    # Konno-Ohmachi smoothing constant
    para_gpy.HV.SMOOTHING_WIDTH = 10 ** (np.pi / getCP(proj_para, 'hvsr', 'ko_smoothing', 'float')) - 1
    # azimuthal parameters
    if getCP(proj_para, 'hvsr', 'method') == 'squared-average':
        para_gpy.HORIZONTAL_COMPONENTS = 'Squared'
    elif getCP(proj_para, 'hvsr', 'method') == 'azimuth':
        para_gpy.HORIZONTAL_COMPONENTS = 'Azimuth'
        para_gpy.HORIZONTAL_AZIMUTH = getCP(proj_para, 'hvsr', 'azimuth', 'float')
    if update:
        update_prog('Initialisation', [4, 6], 0, 'Loading project files')
    else:
        print('Parameters loaded')

    # --------------------------------------------------------------------------------
    # load sac files, prep gpydb
    siglst = prep_project(data_path, proj_recd, proj_para, channel=['E', 'N', 'Z'], update=update, geopsy=True)

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
            print('Done saving back-up')

    # --------------------------------------------------------------------------------
    # start main loop
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
        gpypath = str(ioutpath.joinpath(sigdb.MasterGroup + '_3c.gpy'))
        sigdb.write(gpypath)

        # --------------------------------------------------------------------------------
        # transfer timerange to param
        if len(np.unique(tab['TimeRange0'])) == 1 and len(np.unique(tab['TimeRange1'])) == 1:
            para_gpy.COMMON.FROM_TIME_TYPE = 'Absolute'
            para_gpy.COMMON.FROM_TIME_TEXT = sigdb.File[0].Signal.TimeRange.split(' ')[0]
            para_gpy.COMMON.TO_TIME_TYPE = 'Absolute'
            para_gpy.COMMON.TO_TIME_TEXT = sigdb.File[0].Signal.TimeRange.split(' ')[1]

        # --------------------------------------------------------------------------------
        # write .log parameter file
        parafile = str(ioutpath.joinpath('param_hv.log'))
        hvpath = str(ioutpath.joinpath('gpy_' + sigdb.MasterGroup + '.hv'))
        stapath = ioutpath.joinpath('station')
        if not stapath.exists():
            os.mkdir(str(stapath))
        tmppath = ioutpath.joinpath('tmp-')
        if not tmppath.exists():
            os.mkdir(str(tmppath))
        para_gpy.write(parafile, 'hv')

        # --------------------------------------------------------------------------------
        # run geopsy command (HVSR)
        if update:
            tik = time()
            update_prog('Subprocess', [1, 3], 0, 'Processing')
        os.system(' '.join([binhvsr, '-db', gpypath, '-group-pattern "All signals" -param', parafile,
                            '-nobugreport -o tmp']))  # output file to 'tmp-' directory
        hvs = list(tmppath.glob('*.hv'))
        hvs.sort()
        stas = list()
        for ihvs in hvs:
            stas.append(HVfile(fromfile=str(ihvs)))
            os.rename(str(ihvs), str(stapath.joinpath('gpy_' + sigdb.MasterGroup + '.' + ihvs.name)))
        os.system('rm -r '+str(tmppath))
        hv_group = HVfile(stas)
        hv_group.write(hvpath)

        # --------------------------------------------------------------------------------
        # plot
        if update:
            update_logs('Calculation GPY_HV: {}'.format(siglst[i].MasterGroup), 'log'
                        'Time lapsed: {:.3f}s'.format(time() - tik))
            update_prog('Subprocess', [2, 3], 0, 'Saving figures')
        else:
            print('Done HVSR calculation!')
        if getCP(proj_para, 'basic', 'save_png', 'bool'):
            plog = getCP(proj_para, 'basic', 'plot_log', 'bool')
            for ista in stas:
                plotFunc.hv_array(np.array([ista.meanHV, np.sqrt(ista.maxHV / ista.minHV)]), ista.frequency, log=plog,
                                  plottype='meanstd', title='[GPY] HVSR | {} - {}'.format(sigdb.MasterGroup, ista.name),
                                  saveto=stapath.joinpath('gpy_hvsta-{}[{}].png'.format(sigdb.MasterGroup, ista.name)))
            plotFunc.hv_array(np.array([ista.meanHV for ista in stas]), hv_group.frequency, log=plog,
                              plottype='all', legend=[ista.name for ista in stas],
                              title='[GPY] HVSR by station | {}'.format(sigdb.MasterGroup),
                              saveto=ioutpath.joinpath('gpy_hv-[{}].png'.format(sigdb.MasterGroup)))
            plotFunc.hv_array(np.array([hv_group.meanHV, np.sqrt(hv_group.maxHV / hv_group.minHV)]), hv_group.frequency,
                              log=plog, plottype='meanstd', title='[GPY] HVSR mean | {}'.format(sigdb.MasterGroup),
                              saveto=ioutpath.joinpath('gpy_hvmean-[{}].png'.format(sigdb.MasterGroup)))
            plotFunc.location(coord, [ista.name for ista in stas], title='Location | {}'.format(sigdb.MasterGroup),
                              saveto=ioutpath.joinpath('location-{}.png'.format(sigdb.MasterGroup)))
            if update:
                update_logs('Output HV: {}'.format(siglst[i].MasterGroup), 'log', 'HV fig saved')
            else:
                print('HV fig saved!')
        if update:
            update_prog('Subprocess', [3, 3], 1, 'Complete')
    if update:
        update_prog('Processing', [len(siglst), len(siglst)], 1,
                    'All processes complete! Total time: {:.2f}s'.format(time() - ttik))


if __name__ == '__main__':
    try:
        argv = sys.argv[1]
        print('Input proj: {}'.format(argv))
        calc(argv)
    except Exception as e:
        raise e
