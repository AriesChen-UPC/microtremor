#%%

import warnings
from pathlib import Path

# import cv2
import timeit  # time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import tri as mtri, path as mpath
from osgeo import gdal
from pyproj import Proj, Transformer
from scipy.interpolate import interp1d


def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(f'Array{arg.shape}' if type(arg) is np.ndarray else repr(arg) for arg in args)
        kwargs_str = ', '.join(f'{key}={value}' for key, value in kwargs.items())
        print('[%0.5fsec] %s(%s, %s)' % (elapsed, name, arg_str, kwargs_str))
        return result
    return clocked


@clock
def triInterp(ix, iy, ic, x0, y0, method='linear'):
    """
    Interpolate the data using matplotlib triangulation.

    Parameters
    ----------
    ix: array_like
        Input x matrix.
    iy: array_like
        Input y matrix.
    ic: array_like
        Input data matrix. Note that the input matrix shape should be the same as ix and iy.
    x0: array_like
        X gridmesh for interpolation.
    y0: array_like
        Y gridmesh for interpolation.
    method: str
        Interpolation method, default is 'linear'.

    Returns
    -------
    array_like
        Interpolated data.
    """
    if not ix.shape == iy.shape == ic.shape:
        raise Exception('xmat, ymat, cmat must have the same shape')
    triang = mtri.Triangulation(ix.flatten(), iy.flatten())
    poly = mpath.Path(vertices=np.array([outerMat(ix), outerMat(iy)]).T, closed=True)
    triang.set_mask(~poly.contains_points(np.array([ix.flatten()[triang.triangles].mean(1),
                                                    iy.flatten()[triang.triangles].mean(1)]).T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        triang.set_mask(mtri.TriAnalyzer(triang).get_flat_tri_mask(1e-5))
    if method == 'linear':
        interp = mtri.LinearTriInterpolator(triang, ic.flatten())
    elif method == 'geom':
        interp = mtri.CubicTriInterpolator(triang, ic.flatten(), kind='geom')
    elif method == 'min_E':
        interp = mtri.CubicTriInterpolator(triang, ic.flatten(), kind='min_E')
    else:
        raise Exception('method must be linear, geom or min_E')
    cmat = interp(x0, y0)
    ''' 
    # test plot
    plt.contourf(x0, y0, cmat, cmap='jet', levels=10)
    plt.triplot(triang, '-', color='k')
    plt.colorbar()
    '''
    return cmat


def readblock(file):
    """
    Reads a block of data from a file.
    """
    with open(file, 'rb') as f:
        head = f.read(512).strip(b'x\00').decode('utf-8').split('\n')[:-1]
        dict_head = {}
        for line in head:
            key, value = line.split('\t')
            dict_head[key] = value
        if '<dtype>' in dict_head.keys():
            dtype = dict_head['<dtype>']
        else:
            dtype = 'float32'
        data = np.frombuffer(f.read(), dtype=dtype)
        if '<nx,ny,nz>' in dict_head.keys():
            nx, ny, nz = dict_head['<nx,ny,nz>'].strip('()').split(',')
            data.resize(int(nx), int(ny), int(nz))
    return data, dict_head


def readswath(folder):
    """
    Reads a swath of data from a file.
    """
    folder = Path(folder)
    files = sorted([fi for fi in folder.iterdir() if fi.is_file() and fi.suffix == '.bin'])
    data = []
    for fi in files:
        data.append(readblock(fi)[0])
    return np.concatenate(data)


def arr2raster(arr, raster_file, prj=None, trans=None, verbose=True):
    """
    将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :param verbose: bool
    :return:
    """

    if type(raster_file) is not str:
        raster_file = str(raster_file)
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Int16, options=['COMPRESS=LZW'])

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    # 将数组的各通道写入图片
    Z = np.ones(arr.shape) * -32768
    if np.ma.isMaskedArray(arr):
        Z[~arr.mask] = arr[~arr.mask]
    else:
        Z[~np.isnan(arr)] = arr[~np.isnan(arr)]
    dst_ds.GetRasterBand(1).WriteArray(Z)
    dst_ds.GetRasterBand(1).SetNoDataValue(-32768)

    dst_ds.FlushCache()
    del dst_ds
    print("successfully convert array to raster") if verbose else None


# calculate x, y for all blocks/traces/swaths
def normLine(xcoord, ycoord):
    ixy = np.array([xcoord, ycoord])
    dxy = np.diff(ixy, n=1, axis=-1)
    vector = dxy / np.sqrt(np.sum(dxy ** 2, 0))
    norm0 = np.diff(vector, n=1, axis=-1)
    norm0 *= -1 * (np.cross(vector[:, :-1].T, norm0.T) < 0)  # flip norms to one side
    ind0 = np.where(np.sum(norm0 ** 2, 0) == 0)[0]
    norm0[:, ind0] = vector[[1, 0]][:, ind0] * np.array([-1, 1])[:, None]
    norm = np.concatenate([(vector[[1, 0]][:, 0] * [-1, 1])[:, None], norm0,
                           (vector[[1, 0]][:, -1] * [-1, 1])[:, None]], axis=-1)
    norm = norm / np.sqrt(np.sum(norm ** 2, 0))  # normalization
    return norm


def traceCoord(xcoord, ycoord, dtrace=0.1, ntrace=15):
    ixy = np.array([xcoord, ycoord])
    norms = normLine(xcoord, ycoord)
    mtrace = np.arange(-(ntrace - 1) / 2, ntrace / 2) * dtrace * np.ones([norms.shape[-1], ntrace])
    return ixy[..., None] + mtrace[None] * norms[..., None]


def outerMat(inmat):
    """
    Extracts the outer bound of a 2-D matrix

    Parameters
    ----------
    inmat: np.ndarray
        2-D matrix

    Returns
    -------
    np.ndarray
        1-D array
    """
    if inmat.ndim !=2:
        raise Exception('The outerMat function only accepts 2-D matrix!')
    polygon = [inmat[0, :-1], inmat[:-1, -1], inmat[-1, 1:][::-1], inmat[1:, 0][::-1]]
    return np.concatenate(polygon)


def genTraces(icoord, icscan, dtrace=0.1, x0=None, y0=None, mode='lonlat', transformer=None, method='linear'):
    """
    Generate trace map from coordinates and data
    Parameters
    ----------
    icoord: dataframe or array_like
    icscan: array_like
    dtrace: float
    x0: array_like
    y0: array_like
    mode: str
    transformer: Transformer
    method: str
    verbose: bool

    Returns
    -------

    """
    assert icoord.ndim == icscan.ndim == 2, 'The input data should be 2-D matrix!'
    assert icoord.shape[0] == icscan.shape[0], 'The number of coordinates and data should be the same!'
    ntrace = icscan.shape[1]
    if mode == 'lonlat':
        # print('LON/LAT mode')
        if transformer is None:
            transformer = Transformer.from_proj(Proj('epsg:4326'), Proj('epsg:3857'))
        if type(icoord) == pd.DataFrame:
            assert 'lon' in icoord.columns and 'lat' in icoord.columns, 'coordinates should be named as lon and lat!'
            ix, iy = traceCoord(*transformer.transform(icoord['lon'], icoord['lat']), dtrace=dtrace, ntrace=ntrace)
        else:
            assert icoord.shape[1] == 2, 'The coordinates should be in lon/lat format!'
            ix, iy = traceCoord(*transformer.transform(icoord[:, 0], icoord[:, 1]), dtrace=dtrace, ntrace=ntrace)
    else:
        # print('X/Y mode')
        assert icoord.shape[1] == 2, 'The coordinates should be in x/y format!'
        ix, iy = traceCoord(icoord[:, 0], icoord[:, 1], dtrace=dtrace, ntrace=icscan.shape[1])
    if x0 is None or y0 is None:
        x0, y0 = np.meshgrid(np.arange(ix.min()-1, ix.max()+1, 0.1), np.arange(iy.min()-1, iy.max()+1, 0.1))
    cmat = triInterp(ix, iy, icscan, x0, y0, method)
    return cmat


#%%
if __name__ == '__main__':
    # setup projection transformer
    proj_wgs = Proj(proj='lonlat', datum='WGS84')
    proj_utm = Proj(proj='utm', zone=48, datum='WGS84')
    transformer = Transformer.from_proj(proj_wgs, proj_utm)

    # read data and coordinates
    path = Path('D:/ProjectMaterials/3dModel/Data/GPR/dataTransform/Raw')
    coords = sorted([fi for fi in path.iterdir() if fi.is_file() and fi.suffix == '.csv'])
    swaths = sorted([fi for fi in path.iterdir() if fi.is_dir()])
    names = [fi.stem for fi in coords]
    xtmp = []
    ytmp = []
    ctmp = []
    for i, iname in enumerate(names):
        # print(i, iname)
        icoord = pd.read_csv(coords[i])
        iswath = readswath(swaths[i])

        itp = interp1d(np.arange(icoord.shape[0]), icoord.to_numpy(), axis=0, fill_value='extrapolate') \
            (np.linspace(0, icoord.shape[0], iswath.shape[0]))

        cscan = iswath[:, :, 20:25].mean(-1)
        xtmp.append(itp[..., 1])
        ytmp.append(itp[..., 2])
        ctmp.append(cscan)
    print(f'Data loaded!')

    # grid setup
    resolution = 0.05
    margin = 1
    xval, yval = transformer.transform(np.concatenate(xtmp, axis=0), np.concatenate(ytmp, axis=0))
    xrange = np.arange(xval.min() - margin, xval.max() + margin, resolution)
    yrange = np.arange(yval.min() - margin, yval.max() + margin, resolution)
    xgrid, ygrid = np.meshgrid(xrange, yrange)

    # interpolation
    cscans = []
    for i, ic in enumerate(ctmp):
        print(f'Processing swath {i+1}/{len(ctmp)}')
        cscans.append(genTraces(np.array([xtmp[i], ytmp[i]]).T, ctmp[i], dtrace=0.1, x0=xgrid, y0=ygrid,
                                mode='lonlat', transformer=transformer, method='linear'))
    cmerge = np.ma.mean(cscans, 0)
    arr2raster(np.flip(cmerge, axis=0), f'D:/ProjectMaterials/3dModel/Data/GPR/dataTransform/Raw/{path.name}-merged.tif',
               prj=proj_utm.to_proj4(), trans=[xrange[0], resolution, 0, yrange[-1], 0, -resolution])

