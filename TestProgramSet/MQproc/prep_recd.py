import pandas as pd
import numpy as np
import shutil
import zipfile
import re
from sys import argv
from pathlib import Path


def dk2num(dk):
    """
    Convert Distance Kilometre to metres (format: '{prefix}DK{dk}+{dm}')

    Parameters
    ----------
    dk: str
        Distance Kilometre in string format

    Returns
    -------
    float
        Distance in metres
    """
    assert re.match('^[A-Z]DK\d+\+\d{1,3}(\.\d+)?', dk, re.IGNORECASE) is not None, 'Invalid DK format!'
    km = int(dk[3:re.match('^[A-Z]DK\d+\+', dk, re.IGNORECASE).end()-1])
    m = float(dk[re.match('^[A-Z]DK\d+\+', dk, re.IGNORECASE).end():re.match('^[A-Z]DK\d+\+\d{1,3}(\.\d+)?', dk, re.IGNORECASE).end()])
    return km * 1000 + m


def polyXY(n_poly, radius, x0=0, y0=0):
    x = np.zeros(n_poly + 1) + x0
    y = np.zeros(n_poly + 1) + y0
    for j in range(n_poly):
        x[j + 1] += np.sin(2 * np.pi * j / n_poly) * radius
        y[j + 1] += np.cos(2 * np.pi * j / n_poly) * radius
    return x, y


def gridXY(nx, ny, dx, dy, x0=0, y0=0):
    x = np.arange(x0, x0 + nx * dx, dx)
    y = np.arange(y0, y0 + ny * dy, dy)
    x0, y0 = np.meshgrid(x, y)
    return x0.flatten(), y0.flatten()


def get_xy(codename, **kwargs):
    kind, *n = codename.split('_')
    if kind == 'poly':
        n = int(n[0])
        return polyXY(n, **kwargs)
    elif kind == 'grid':
        if 'radius' in kwargs.keys() and 'dx' not in kwargs.keys() and 'dy' not in kwargs.keys():
            kwargs['dx'] = kwargs['radius']
            kwargs['dy'] = kwargs['radius']
            kwargs.pop('radius')
        n = [int(i) for i in n]
        kwargs['x0'] = -(n[0]-1) * kwargs['dx'] / 2
        return gridXY(*n, **kwargs)


def unzip(file, outpath=None, write_table=False):
    # prepare path
    root = Path(file)
    assert root.is_file() and root.suffix == '.zip', 'file must be a zip file'
    if outpath is None:
        outpath = root.parent
    else:
        outpath = Path(outpath)
    if not outpath.exists():
        outpath.mkdir()
    imgdir = outpath.joinpath('img')
    imgdir.mkdir(exist_ok=True)
    # fs = [fi for fi in root.iterdir() if fi.suffix == '.zip' and '表单反馈导出' in fi.stem]
    # unzip file
    ids = []
    with zipfile.ZipFile(file, 'r') as zf:
        for f in zf.namelist():
            # concatenate tables
            if f.endswith('.xlsx'):
                tab = pd.read_excel(zf.open(f), skiprows=1, skipfooter=3)  # , nrows=1)
                # df.index = [i]
                # tab = pd.concat([tab, df], ignore_index=True)
            # unzip images to img folder
            elif f.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                tmpimg = Path(zf.extract(f, imgdir))
                ids.append(tmpimg.stem.split("_")[0])
                imgdir.joinpath(f'{int(tmpimg.stem.split("_")[0]):03d}').mkdir(exist_ok=True)
                tmpimg.rename(imgdir.joinpath(f'{int(tmpimg.stem.split("_")[0]):03d}', tmpimg.stem + '.jpg'))
                [shutil.rmtree(fd) for fd in imgdir.iterdir() if fd in tmpimg.parents]
    # output table
    # tab = tab.loc[tab['序号'].isin(ids)]
    tab.sort_values(by='序号', inplace=True)
    if write_table:
        tab.to_csv(outpath.joinpath('table.csv'), index=False, encoding='utf-8-sig')
    return tab


def tab2recd(tab):
    recd = pd.DataFrame(columns=['Location', 'Station', 'Network', 'Starttime', 'Endtime', 'X/poly_n', 'Y/radius', 'Z',
                                 'Note', 'Env-wind', 'Env-rain', 'Env-traffic', 'Env-machine',
                                 'Proj-ID', 'Proj-uploaded', 'Proj-date', 'Proj-section'])

    groups = ['A', 'B', 'C', 'D', 'E']  # TODO: inflexible group names

    dates = tab['施工日期'].unique()
    for _, itab in tab.iterrows():
        for igroup in groups:
            if not pd.isna(itab[igroup + '台阵半径（米）']):
                loccol = [ikey for ikey in itab.keys() if igroup+'台阵黄点里程' in ikey][0]
                tmpx, tmpy = get_xy(itab['台阵形状'], radius=itab[igroup + '台阵半径（米）'], y0=dk2num(itab[loccol]))
                for ist in range(len(tmpx)):
                    tmp = pd.DataFrame({'Location': itab[loccol], 'Station': f'{igroup}.{ist + 1}', 'Network': 'DTCC',
                                        'Starttime': itab['采集开始时间'], 'Endtime': itab['采集结束时间'],
                                        'X/poly_n': tmpx[ist], 'Y/radius': tmpy[ist], 'Z': 0.0,
                                        'Env-wind': itab['-[风力]'], 'Env-rain': itab['-[雨]'],
                                        'Env-traffic': itab['-[来往车辆]'], 'Env-machine': itab['-[大型机械]'],
                                        'Proj-ID': itab['序号'], 'Proj-uploaded': itab['提交时间'],
                                        'Proj-date': itab['施工日期'], 'Proj-section': itab['项目名称-[成都19号线]'],
                                        'Note': itab['======备注======']}, index=[0])
                    recd = pd.concat([recd, tmp], ignore_index=True)
    return [recd[recd['Proj-date'] == idate] for idate in dates], \
           [idate.translate({ord('-'): None, ord('/'): None}) for idate in dates]


if __name__ == '__main__':
    try:
        script, filename = argv
        table = unzip(filename, write_table=False)
        records, rdates = tab2recd(table)
        # records, rdates = tab2recd(pd.read_csv('/Users/tianqi/Desktop/MQ2022/Data/表格整理/table.csv'))

        outroot = Path(filename).parent.joinpath('PARAout')
        outroot.mkdir(exist_ok=True)
        for irec, irecord in enumerate(records):
            tmproot = outroot.joinpath(rdates[irec])
            tmproot.mkdir(exist_ok=True)
            irecord.to_csv(tmproot.joinpath('record.csv'), index=False, encoding='utf-8-sig')
        print(f'{len(records)} record(s) saved to {outroot}')
    except ValueError:
        print("Usage: prep_recd.py <zipfile>")
        exit()
