import time
import sys
import numpy as np
from pathlib import Path
from progressbar import ProgressBar, Bar, Percentage

from hvsrpy.sensor3c import Sensor3c
from xcorpy import plotFunc
from xcorpy.baseConfig import cfgupdate, cfgdata, cfgpara, cfgprep, cfgresl
from xcorpy.prepFunc import getCP, load_param, prep_project, prep_location, prep_atrig, prep_init
from xcorpy.coreFunc import update_prog, update_logs, print_line, window
from iotoolpy.ioGpy import HVfile

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
    # shared parameters
    dt, freq, _, _, overlap, _, sta, lta, atrig_range = load_param(proj_para, verbose=False)
    # hard-written parameter (for testing)
    rejection_bool = True
    distribution_mc = 'log-normal'
    distribution_f0 = 'log-normal'
    # HVSR specified parameters
    windowlength = getCP(proj_para, 'hvsr', 'win_len', 'float')
    KOsmoothing = getCP(proj_para, 'hvsr', 'ko_smoothing', 'float')
    method = getCP(proj_para, 'hvsr', 'method', 'str')
    azimuth_degree = getCP(proj_para, 'hvsr', 'azimuth', 'float')
    # optional parameters
    res_type = 'log' if getCP(proj_para, 'freq', 'log', 'bool') else 'linear'
    bp_filter = {"flag": False, "flow": min(freq),
                 "fhigh": max(freq), "order": 1}  # order min at 1, larger for time domain, smaller for frequency domain
    resampling = {"minf": min(freq), "maxf": max(freq),
                  "nf": len(freq), "res_type": res_type}
    # azimuthal parameters
    azimuth0 = np.arange(0, 180, 10)
    if method == "rotate":
        azimuth = azimuth0
    elif method == "azimuth":
        azimuth = azimuth_degree
    else:
        azimuth = None
    if update:
        update_prog('Initialisation', [4, 6], 0, 'Loading project files')
    else:
        print('Parameters loaded')

    # --------------------------------------------------------------------------------
    # load sac files
    siglst = prep_project(data_path, proj_recd, proj_para, ['E', 'N', 'Z'], update=update, geopsy=False)

    # --------------------------------------------------------------------------------
    # saving sac back-up (optional)
    if getCP(proj_para, 'basic', 'save_sac', 'bool'):  # proj_para.getboolean('basic', 'save_sac'):
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
            update_prog('Back-up process', [len(siglst), len(siglst)], 1, 'Back-up complete')
        else:
            pbar.finish()
            print('Done saving back-up!')

    # --------------------------------------------------------------------------------
    # start main loop
    if update:
        update_prog('Initialisation', [6, 6], 1, 'Complete')
        update_prog('Processing', [0, len(siglst)], 0, 'Start')
    else:
        print_line('Start Calculation')
    ttik = time.time()
    # loop each location
    for i in range(len(siglst)):
        ioutpath = resl_path.joinpath(siglst[i][0].stats.location)
        if not ioutpath.exists():
            ioutpath.mkdir()
        hv_group = list()
        if update:
            update_prog('Processing', [(i + 1) * 0.99, len(siglst)], 0, siglst[i][0].stats.location)
            update_prog('Subprocess', [0, len(np.unique([isig.stats.station for isig in siglst[i]]))], 0, 'Start')
            icount = 0
        # loop each station TODO: make parallel
        for istation in np.unique([isig.stats.station for isig in siglst[i]]):
            try:
                if update:
                    icount += 1
                    update_prog('Subprocess', [icount * 0.99, len(np.unique([isig.stats.station for isig in siglst[i]]))],
                                0, istation)
                tik = time.time()
                # preparing signal
                isig = siglst[i].select(station=istation)
                isig.detrend('constant')
                icoord = [isig[0].meta.sac['user7'], isig[0].meta.sac['user8'], isig[0].meta.sac['user9']]
                itimes = isig[0].times('timestamp')
                if rejection_bool:
                    atrig = prep_atrig(isig, dt, sta, lta)
                else:
                    atrig = None
                staname = f'{isig[0].stats.location}[{isig[0].stats.station}]'

                # HVSR calculation
                # TODO: results has minor difference to GPY
                sensor = Sensor3c.from_stream(staname, isig)
                sensor.window(windowlength, overlap, atrig, atrig_range)

                hv = sensor.hv(bp_filter, 0.1, KOsmoothing, resampling, method, azimuth=azimuth)
                hv_file = HVfile(hv, distribution_mc, distribution_f0, pos=icoord)
                hv_group.append(hv_file)
                hv_az = sensor.hv(bp_filter, 0.1, KOsmoothing, resampling, 'multiple-azimuths', azimuth=azimuth0)

                # output station
                idetailpath = ioutpath.joinpath('station')
                if not idetailpath.exists():
                    idetailpath.mkdir()
                hv_file.write(str(idetailpath.joinpath('{}.hv'.format(staname))))
                if getCP(proj_para, 'basic', 'save_png', 'bool'):
                    plotFunc.signals(sensor, atrig=atrig, channel=True, trange=[itimes[0], itimes[-1]],
                                     saveto=idetailpath.joinpath('signal3c-{}.png'.format(staname)))
                if getCP(proj_para, 'basic', 'save_csv', 'bool'):
                    np.savetxt(str(idetailpath.joinpath('hv_{}.csv'.format(hv_file.name))),
                               np.vstack([freq, hv_file.meanHV, hv_file.minHV, hv_file.maxHV]).T,
                               fmt='%.5f', delimiter=',', header='Freq, Mean, Low, Up', comments='')
                    if update:
                        update_logs('Output HVSR: {}'.format(siglst[i][0].stats.location),
                                    '{} H/V data saved'.format(istation))
                if getCP(proj_para, 'basic', 'save_png', 'bool'):
                    plog = getCP(proj_para, 'basic', 'plot_log', 'bool')
                    plotFunc.hv_class(hv, distribution_mc, distribution_f0, log=plog,
                                      title='HVSR by window | {}'.format(staname),
                                      saveto=idetailpath.joinpath(f'hvsta-{staname}.png'))
                    plotFunc.hv_azimuth(hv_az, azimuth0 / 180 * np.pi, freq, cmap='hot_r',
                                        title='HVSR multi-azimuths | {}'.format(staname),
                                        saveto=idetailpath.joinpath(f'hvaz-{staname}.png'))
                    if update:
                        update_logs('Output HVSR: {}'.format(siglst[i][0].stats.location),
                                    '{} H/V fig saved'.format(istation))
            except Exception as e:
                print(e)
                if update:
                    update_logs('Output HVSR: {}'.format(siglst[i][0].stats.location),
                                '{} H/V failed'.format(istation))
                continue
        if update:
            update_prog('Subprocess', [len(np.unique([isig.stats.station for isig in siglst[i]])),
                                       len(np.unique([isig.stats.station for isig in siglst[i]]))], 1, 'Complete')

        # output array
        hv_file0 = HVfile(hv_group, distribution_mc, distribution_f0)
        hv_file0.write((str(ioutpath.joinpath('{}.hv'.format(siglst[i][0].stats.location)))))
        if getCP(proj_para, 'basic', 'save_csv', 'bool'):  # proj_para.getboolean('basic', 'save_csv'):
            np.savetxt(str(ioutpath.joinpath('hv_{}.csv'.format(hv_file0.name))),
                       np.vstack([freq, hv_file0.meanHV, hv_file0.minHV, hv_file0.maxHV]).T,
                       fmt='%.5f', delimiter=',', header='Freq, Mean, Low, Up', comments='')
            if update:
                update_logs('Output HVSR: {}'.format(siglst[i][0].stats.location), 'Array H/V data saved')
        if getCP(proj_para, 'basic', 'save_png', 'bool'):
            plog = getCP(proj_para, 'basic', 'plot_log', 'bool')  # proj_para.getboolean('basic', 'plot_log')
            plotFunc.hv_array(np.array([ihv.meanHV for ihv in hv_group]), freq, log=plog, plottype='all',
                              saveto=ioutpath.joinpath(f'hv-{siglst[i][0].stats.location}.png'),
                              title='HVSR of all station @ {}'.format(siglst[i][0].stats.location),
                              legend=[ihv.name[ihv.name.index('[')+1:ihv.name.index(']')] for ihv in hv_group])
            # TODO: HV station location error
            plotFunc.volume(np.array([ihv.meta['position'][:2] for ihv in hv_group]),
                            np.array([ihv.meanHV for ihv in hv_group])[:, freq > 0.5],
                            zaxis=freq[freq > 0.5], zlog=True, zname='Frequency/Hz',
                            title='HVSR volume plot @ {}'.format(siglst[i][0].stats.location),
                            name=[ihv.name[ihv.name.index('[')+1:ihv.name.index(']')] for ihv in hv_group],
                            saveto=ioutpath.joinpath(f'hv-{siglst[i][0].stats.location}.html'))
            if update:
                update_logs('Output HVSR: {}'.format(siglst[i][0].stats.location), 'Array H/V fig saved')
        if update:
            update_logs('Calculation H/V: {}'.format(siglst[i][0].stats.location),
                        'Time lapsed: {:.2f}s'.format(time.time() - tik))
        else:
            print('Time lapsed: {:.2f} sec'.format(time.time() - tik))
    if update:
        update_prog('Processing', [len(siglst), len(siglst)], 1,
                    'All processes complete! Total time: {:.2f}s'.format(time.time() - ttik))
    else:
        print_line('Summary')
        print('Total time: {:.2f}s'.format(time.time() - ttik))
        print('Program GPY_HVSR finished!')


if __name__ == "__main__":
    try:
        argv = sys.argv[1]
        print('Input proj: {}'.format(argv))
        calc(argv)
    except Exception as e:
        raise e
