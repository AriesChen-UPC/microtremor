import gzip
import os
import re
import zlib
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from xml.dom import minidom

import numpy as np

# from hvsrpy.hvsr import Hvsr
# from iotoolpy.ioSac import read_sac


class AutocorrTarget:
    """Class for loading target files"""
    def __init__(self, file=None):
        self.selected: bool = True
        self.misfitWeight: float = 1
        self.minimumMisfit: float = 0
        self.misfitType: str = 'L2_NormalizedBySigmaOnly'
        self.AutocorrCurves = self.autocorrClass()
        if file:
            self.load(file)

    class autocorrClass:
        def __init__(self):
            self.AutocorrRing: list = list()
            self.ModalCurve: list = list()

        class modelClass:
            def __init__(self):
                self.Mode = self.modeClass()
                self.RealStatisticalPoint: np.recarray = np.recarray(shape=0, dtype=[('x', float), ('imag', float),
                                                                                     ('mean', float), ('stddev', float),
                                                                                     ('weight', int), ('valid', bool)])

            class modeClass:
                def __init__(self):
                    self.slowness: str = 'Phase'
                    self.polarisation: str = 'Vertical'
                    self.ringIndex: int = 1
                    self.index: int = 0

                def __setitem__(self, key, value):
                    orig = self.__getitem__(key)
                    otype = type(orig)
                    if otype is not list and otype is not np.ndarray:
                        val = np.nan if value == '-12345' else otype(value)
                    else:
                        itype = type(orig[0]) if len(orig) > 0 else str
                        val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
                    self.__setattr__(key, val)

                def __getitem__(self, key, txt=False):
                    val = self.__getattribute__(key)
                    if txt:
                        return cvt2str(val)
                    else:
                        return val

            def __setitem__(self, key, value):
                orig = self.__getitem__(key)
                otype = type(orig)
                if otype is not list and otype is not np.ndarray:
                    val = np.nan if value == '-12345' else otype(value)
                else:
                    itype = type(orig[0]) if len(orig) > 0 else str
                    val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
                self.__setattr__(key, val)

            def __getitem__(self, key, txt=False):
                val = self.__getattribute__(key)
                if txt:
                    return cvt2str(val)
                else:
                    return val

        def __setitem__(self, key, value):
            orig = self.__getitem__(key)
            otype = type(orig)
            if otype is not list and otype is not np.ndarray:
                val = np.nan if value == '-12345' else otype(value)
            else:
                itype = type(orig[0]) if len(orig) > 0 else str
                val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
            self.__setattr__(key, val)

        def __getitem__(self, key, txt=False):
            val = self.__getattribute__(key)
            if txt:
                return cvt2str(val)
            else:
                return val

    def __setitem__(self, key, value):
        orig = self.__getitem__(key)
        otype = type(orig)
        if otype is not list and otype is not np.ndarray:
            val = np.nan if value == '-12345' else otype(value)
        else:
            itype = type(orig[0]) if len(orig) > 0 else str
            val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
        self.__setattr__(key, val)

    def __getitem__(self, key, txt=False):
        val = self.__getattribute__(key)
        if txt:
            return cvt2str(val)
        else:
            return val

    def load(self, file, cleanup=True):
        if cleanup:
            self.__init__()
        rootdata = loadXML(file, 'utf-16')
        root0 = rootdata.getElementsByTagName('AutocorrTarget')[0]
        node0 = [x for x in self.__dict__.keys()]
        for inode in node0:
            if inode != 'AutocorrCurves':
                val = root0.getElementsByTagName(inode)[0].firstChild.data.strip('\n ')
                self.__setitem__(inode, val)
            elif inode == 'AutocorrCurves':
                root1 = root0.getElementsByTagName(inode)[0]
                ring_n = len(root1.getElementsByTagName('AutocorrRing'))
                rings = [round((float(root1.getElementsByTagName('AutocorrRing')[i].getElementsByTagName('minRadius')
                                      [0].firstChild.data) +
                                float(root1.getElementsByTagName('AutocorrRing')[i].getElementsByTagName('maxRadius')
                                      [0].firstChild.data)) / 2, 2)
                         for i in range(ring_n)]
                self.AutocorrCurves.__setattr__('AutocorrRing', rings)
                tmp1 = root1.getElementsByTagName('ModalCurve')
                assert len(tmp1) == ring_n
                models = list()
                for i in range(ring_n):
                    root2 = tmp1[i]
                    tmp2 = self.autocorrClass.modelClass()
                    root31 = root2.getElementsByTagName('Mode')[0]
                    node2 = [x for x in self.autocorrClass.modelClass.modeClass().__dict__.keys()]
                    for jnode in node2:
                        if jnode == 'polarisation':
                            try:
                                val = root31.getElementsByTagName(jnode)[0].firstChild.data
                            except IndexError:
                                val = root31.getElementsByTagName('polarization')[0].firstChild.data
                        else:
                            val = root31.getElementsByTagName(jnode)[0].firstChild.data
                        tmp2.Mode.__setitem__(jnode, val)
                    root32 = root2.getElementsByTagName('RealStatisticalPoint')
                    npts = len(root32)
                    tmp2.RealStatisticalPoint = np.recarray(shape=npts, dtype=[('x', float), ('imag', float),
                                                                               ('mean', float), ('stddev', float),
                                                                               ('weight', int), ('valid', bool)])
                    node3 = tmp2.RealStatisticalPoint.dtype.names
                    for j in range(npts):
                        for jnode in node3:
                            try:
                                val = root32[j].getElementsByTagName(jnode)[0].firstChild.data
                            except IndexError:
                                val = 0
                            if jnode == 'valid':
                                if val.lower() == 'true':
                                    val = True
                                elif val.lower() == 'false':
                                    val = False
                                else:
                                    val = None
                            tmp2.RealStatisticalPoint[jnode][j] = val
                    models.append(tmp2)
                self.AutocorrCurves.__setattr__('ModalCurve', models)
        # print('Loaded file: ' + file)

    def write(self, filename='/Users/tenqei/Desktop/test_xml.target'):
        # create new root element
        doc = minidom.Document()
        rootdata = doc.createElement('AutocorrTarget')
        node0 = [x for x in self.__dict__.keys()]
        for inode in node0:
            if inode != 'AutocorrCurves':
                tmp0 = doc.createElement(inode)
                tmp0.appendChild(doc.createTextNode(self.__getitem__(inode, True)))
                rootdata.appendChild(tmp0)
            elif inode == 'AutocorrCurves':
                tmp0 = doc.createElement(inode)
                n1 = len(self.AutocorrCurves.AutocorrRing)
                n2 = len(self.AutocorrCurves.ModalCurve)
                assert n1 == n2
                ring_n = n1
                for i in range(ring_n):
                    tmp1 = doc.createElement('AutocorrRing')
                    rMin = doc.createElement('minRadius')
                    rMax = doc.createElement('maxRadius')
                    r = self.AutocorrCurves.AutocorrRing[i]
                    rMin.appendChild(doc.createTextNode(str(r - 1e-2)))
                    rMax.appendChild(doc.createTextNode(str(r + 1e-2)))
                    tmp1.appendChild(rMin)
                    tmp1.appendChild(rMax)
                    tmp0.appendChild(tmp1)
                    tmp2 = doc.createElement('ModalCurve')
                    tmp2.setAttribute('type', "autocorr")
                    mode = doc.createElement('Mode')
                    for jnode in self.AutocorrCurves.ModalCurve[i].Mode.__dict__.keys():
                        tmp3 = doc.createElement(jnode)
                        tmp3.appendChild(doc.createTextNode(
                            self.AutocorrCurves.ModalCurve[i].Mode.__getitem__(jnode, True)))
                        mode.appendChild(tmp3)
                    tmp2.appendChild(mode)
                    for j in range(self.AutocorrCurves.ModalCurve[i].RealStatisticalPoint.size):
                        stat = doc.createElement('RealStatisticalPoint')
                        for jnode in self.AutocorrCurves.ModalCurve[i].RealStatisticalPoint.dtype.names:
                            tmp3 = doc.createElement(jnode)
                            tmp3.appendChild(doc.createTextNode(
                                str(self.AutocorrCurves.ModalCurve[i].RealStatisticalPoint[jnode][j]).lower()))
                            stat.appendChild(tmp3)
                        tmp2.appendChild(stat)
                    tmp0.appendChild(tmp2)
                rootdata.appendChild(tmp0)
        tmp = doc.createElement('TargetList')
        tmp.appendChild(rootdata)
        doc.appendChild(tmp)
        # write to xml file
        with open(filename, 'w') as f:
            # f.write(doc.toprettyxml(indent = '\t', newl = '\n', encoding = 'utf-8'))
            doc.writexml(f, indent='  ', newl='\n', addindent='  ', encoding='utf-8')
            f.close()
        print('Wrote file: ' + filename)

    def fromSPAC(self, freq: np.ndarray, spac: np.ndarray, std: np.ndarray, rings=None, weights=None):
        self.__init__()
        if rings is None:
            rings = list(range(spac.shape[0]))
        if weights is None:
            weights = np.ones(shape=len(freq))
        if len(freq) != spac.shape[1] or len(freq) != std.shape[1]:

            raise Exception('Frequency points no match!')
        if len(rings) != spac.shape[0] or len(rings) != std.shape[0]:
            raise Exception('Ring number no match!')
        npts = len(freq)
        self.AutocorrCurves.AutocorrRing = list(rings)
        for iring in range(len(rings)):
            tmp0 = self.autocorrClass.modelClass()
            tmp0.Mode.ringIndex = iring
            tmp0.RealStatisticalPoint = np.recarray(shape=npts, dtype=[('x', float), ('imag', float),
                                                                       ('mean', float), ('stddev', float),
                                                                       ('weight', int), ('valid', bool)])
            tmp0.RealStatisticalPoint['x'] = freq
            tmp0.RealStatisticalPoint['valid'] = ~np.isnan(spac[iring])
            tmp0.RealStatisticalPoint['mean'] = spac[iring].real
            tmp0.RealStatisticalPoint['imag'] = spac[iring].imag
            tmp0.RealStatisticalPoint['stddev'] = std[iring]
            tmp0.RealStatisticalPoint['weight'] = weights
            self.AutocorrCurves.ModalCurve.append(tmp0)

    def toSPAC(self):
        rings = np.array(self.AutocorrCurves.AutocorrRing)
        nring = len(rings)
        npts = len(self.AutocorrCurves.ModalCurve[0].RealStatisticalPoint)
        spac = np.empty(shape=(nring, npts), dtype=complex)
        std = np.empty(shape=(nring, npts), dtype=float)
        weights = np.array(self.AutocorrCurves.ModalCurve[0].RealStatisticalPoint['weight'])
        freq = np.array(self.AutocorrCurves.ModalCurve[0].RealStatisticalPoint['x'])
        for iring in range(nring):
            spac[iring] = self.AutocorrCurves.ModalCurve[iring].RealStatisticalPoint['mean'] + \
                          self.AutocorrCurves.ModalCurve[iring].RealStatisticalPoint['imag'] * 1j
            std[iring] = self.AutocorrCurves.ModalCurve[iring].RealStatisticalPoint['stddev']
        return freq, spac, std, rings, weights


class SignalDatabase:
    """Class for loading gpy database files"""
    def __init__(self, file=None):
        self.version: str = '8'
        self.SharedMetaData: str = ''
        self.File: list = list()
        self.MasterGroup: str = ''
        self.SeismicEventTable: str = ''
        if file:
            self.load(file)

    class fileClass:
        def __init__(self):
            self.name: str = ''
            self.format: str = 'SacLittleEndian'
            self.original: str = 'true'
            self.size: int = 0
            self.crc32: int = 0
            self.Signal: classmethod = self.sigClass()

        def __setitem__(self, key, value):
            orig = self.__getitem__(key)
            otype = type(orig)
            if otype is not list and otype is not np.ndarray:
                val = np.nan if value == '-12345' else otype(value)
            else:
                itype = type(orig[0]) if len(orig) > 0 else str
                val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
            self.__setattr__(key, val)

        def __getitem__(self, key, txt=False):
            val = self.__getattribute__(key)
            if txt:
                return cvt2str(val)
            else:
                return val

        class sigClass:
            def __init__(self):
                self.ID: str = ''
                self.Name: str = ''
                self.Component: str = 'Vertical'
                self.StartTime: float = 0.0
                self.SamplingPeriod: float = 0.0
                self.Type: str = 'Waveform'
                self.NSamples: int = 0
                self.CountPerVolt: int = 1
                self.VoltPerUnit: int = 1
                self.AmplitudeUnit: str = 'Velocity'
                self.NumberInFile: int = 0
                self.Receiver: list = [0.0, 0.0, 0.0]
                self.TimeRange: str = ''
                self.UtmZone: str = '###'
                self.OffsetInFile: int = 632  # unknown
                self.ByteIncrement: int = 0  # unknown

            def __setitem__(self, key, value):
                orig = self.__getattribute__(key)
                otype = type(orig)
                if otype is not list and otype is not np.ndarray:
                    val = np.nan if value == '-12345' else otype(value)
                else:
                    itype = type(orig[0]) if len(orig) > 0 else str
                    val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
                self.__setattr__(key, val)

            def __getitem__(self, key, txt=False):
                val = self.__getattribute__(key)
                if txt:
                    return cvt2str(val)
                else:
                    return val

    def __setitem__(self, key, value):
        orig = self.__getitem__(key)
        otype = type(orig)
        if otype is not list and otype is not np.ndarray:
            val = np.nan if value == '-12345' else otype(value)
        else:
            itype = type(orig[0]) if len(orig) > 0 else str
            val = [np.nan if x == '-12345' else itype(x) for x in re.split('[, \t]', value) if len(x) > 0]
        self.__setattr__(key, val)

    def __getitem__(self, key, txt=False):
        val = self.__getattribute__(key)
        if txt:
            return cvt2str(val)
        else:
            return val

    def load(self, file, cleanup=True):
        if cleanup:
            self.__init__()
        rootdata = loadXML(file, 'utf-16')
        node0 = [x for x in self.__dict__.keys()]
        for inode in node0:
            if inode != 'File':
                val = rootdata.getElementsByTagName(inode)[0].firstChild.data.strip('\n ')
                self.__setitem__(inode, val)
            elif inode == 'File':
                elements1 = rootdata.getElementsByTagName(inode)
                flen = len(elements1)
                tmp0 = list()
                for i in range(flen):
                    node1 = [x for x in self.fileClass().__dict__.keys()]
                    tmp1 = self.fileClass()
                    for jnode in node1:
                        if jnode != 'Signal':
                            val = elements1[i].getElementsByTagName(jnode)[0].firstChild.data
                            tmp1.__setitem__(jnode, val)
                        elif jnode == 'Signal':
                            elements2 = elements1[i].getElementsByTagName(jnode)[0]
                            node2 = [x for x in self.fileClass().sigClass().__dict__.keys()]
                            tmp2 = self.fileClass.sigClass()
                            for knode in node2:
                                val = elements2.getElementsByTagName(knode)[0].firstChild.data
                                tmp2.__setitem__(knode, val)
                            tmp1.__setattr__(jnode, tmp2)
                    tmp0.append(tmp1)
                self.__setattr__(inode, tmp0)
        print('Loaded file: ' + file)

    def write(self, filename='/Users/tenqei/Desktop/test_xml.gpy'):
        # create new root element
        doc = minidom.Document()
        rootdata = doc.createElement('SignalDatabase')
        node0 = [x for x in self.__dict__.keys()]
        for inode in node0:
            if inode != 'File':
                tmp0 = doc.createElement(inode)
                tmp0.appendChild(doc.createTextNode(self.__getitem__(inode, True)))
                rootdata.appendChild(tmp0)
            elif inode == 'File':
                flen = len(self.File)
                for i in range(flen):
                    tmp0 = doc.createElement(inode)
                    node1 = [x for x in self.File[i].__dict__.keys()]
                    for jnode in node1:
                        if jnode != 'Signal':
                            tmp1 = doc.createElement(jnode)
                            tmp1.appendChild(doc.createTextNode(self.File[i].__getitem__(jnode, True)))
                            tmp0.appendChild(tmp1)
                        elif jnode == 'Signal':
                            node2 = [x for x in self.fileClass().sigClass().__dict__.keys()]
                            tmp1 = doc.createElement(jnode)
                            for knode in node2:
                                tmp2 = doc.createElement(knode)
                                tmp2.appendChild(doc.createTextNode(self.File[i].Signal.__getitem__(knode, True)))
                                tmp1.appendChild(tmp2)
                            tmp0.appendChild(tmp1)
                    rootdata.appendChild(tmp0)
        doc.appendChild(rootdata)
        # write to xml file
        with open(filename, 'w') as f:
            # f.write(doc.toprettyxml(indent = '\t', newl = '\n', encoding = 'utf-8'))
            doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-16le')
            f.close()
        print('Wrote file: ' + filename)

    def filenames(self):
        return [self.File[i].name for i in range(len(self.File))]

    def toTab(self):
        nfile = len(self.File)
        tab = np.recarray(shape=nfile, dtype=[('filename', '<U200'), ('format', '<U20'), ('original', '<U20'),
                                              ('size', int), ('crc32', int),
                                              ('ID', '<U20'), ('Name', '<U20'), ('Component', '<U20'),
                                              ('StartTime', float), ('SamplingPeriod', float), ('Type', '<U20'),
                                              ('NSamples', int), ('CountPerVolt', int), ('VoltPerUnit', int),
                                              ('AmplitudeUnit', '<U20'), ('NumberInFile', int),
                                              ('ReceiverX', float), ('ReceiverY', float), ('ReceiverZ', float),
                                              ('TimeRange0', float), ('TimeRange1', float),
                                              ('UtmZone', '<U20'), ('OffsetInFile', int), ('ByteIncrement', int)])
        for i in range(nfile):
            for iname in tab.dtype.names:
                if iname == 'filename':
                    tab[iname][i] = self.File[i].__getitem__('name')
                if iname in ['format', 'original', 'size', 'crc32']:
                    tab[iname][i] = self.File[i].__getitem__(iname)
                elif iname in ['ID', 'Name', 'Component', 'StartTime', 'SamplingPeriod', 'Type', 'NSamples',
                               'CountPerVolt', 'VoltPerUnit', 'AmplitudeUnit', 'NumberInFile',
                               'UtmZone', 'OffsetInFile', 'ByteIncrement']:
                    tab[iname][i] = self.File[i].Signal.__getitem__(iname)
                elif iname == 'TimeRange0':
                    tmpval = self.File[i].Signal.__getitem__('TimeRange')
                    tab[iname][i] = float(tmpval.split(' ')[0])
                elif iname == 'TimeRange1':
                    tmpval = self.File[i].Signal.__getitem__('TimeRange')
                    tab[iname][i] = float(tmpval.split(' ')[1])
                elif iname == 'ReceiverX':
                    tab[iname][i] = self.File[i].Signal.__getitem__('Receiver')[0]
                elif iname == 'ReceiverY':
                    tab[iname][i] = self.File[i].Signal.__getitem__('Receiver')[1]
                elif iname == 'ReceiverZ':
                    tab[iname][i] = self.File[i].Signal.__getitem__('Receiver')[2]
        return tab

    def fromTab(self, tab: np.recarray, cleanup=True):
        if cleanup:
            self.__init__()
        nfile = tab.size
        self.__setitem__('version', '8')
        filelist = list()
        for i in range(nfile):
            tmp0 = self.fileClass()
            tmp0.__setitem__('name', tab['filename'][i])
            for iname in ['format', 'original', 'size', 'crc32']:
                tmp0.__setitem__(iname, tab[iname][i])
            tmp1 = self.fileClass.sigClass()
            for iname in ['ID', 'Name', 'Component', 'StartTime', 'SamplingPeriod', 'Type', 'NSamples',
                          'CountPerVolt', 'VoltPerUnit', 'AmplitudeUnit', 'NumberInFile',
                          'UtmZone', 'OffsetInFile', 'ByteIncrement']:
                tmp1.__setitem__(iname, tab[iname][i])
            tmp1.__setitem__('TimeRange', '{:.3f} {:.3f}'.format(tab['TimeRange0'][i], tab['TimeRange1'][i]))
            tmp1.__setattr__('Receiver', [tab['ReceiverX'][i], tab['ReceiverY'][i], tab['ReceiverZ'][i]])
            tmp0.__setattr__('Signal', tmp1)
            filelist.append(tmp0)
        self.File = filelist

    # def fromFile(self, filenames: list, coord=None, cleanup=True):
    #     nfile = len(filenames)
    #     tab = np.recarray(shape=nfile, dtype=[('filename', '<U200'), ('format', '<U20'), ('original', '<U20'),
    #                                           ('size', int), ('crc32', int),
    #                                           ('ID', '<U20'), ('Name', '<U20'), ('Component', '<U20'),
    #                                           ('StartTime', float), ('SamplingPeriod', float), ('Type', '<U20'),
    #                                           ('NSamples', int), ('CountPerVolt', int), ('VoltPerUnit', int),
    #                                           ('AmplitudeUnit', '<U20'), ('NumberInFile', int),
    #                                           ('ReceiverX', float), ('ReceiverY', float), ('ReceiverZ', float),
    #                                           ('TimeRange0', float), ('TimeRange1', float),
    #                                           ('UtmZone', '<U20'), ('OffsetInFile', int), ('ByteIncrement', int)])
    #
    #     for fi in range(nfile):
    #         head = read_sac(filenames[fi], headonly=True)
    #         dt = head[0][0]
    #         npnts = head[1][9]
    #         reftime = datetime.strptime('.'.join([str(x) for x in head[1][:6]]), '%Y.%j.%H.%M.%S.%f')
    #         st = reftime + timedelta(seconds=float(head[0][5]))
    #         et = st + timedelta(seconds=dt * npnts)
    #
    #         tab['filename'][fi] = filenames[fi]
    #         tab['format'][fi] = 'SacLittleEndian'
    #         tab['original'][fi] = 'true'
    #         tab['size'][fi] = os.path.getsize(filenames[fi])
    #         tab['crc32'][fi] = crc32(filenames[fi])
    #         tab['ID'][fi] = fi + 1
    #         tab['Name'][fi] = head[2][0].strip(b' ')
    #         tab['Component'][fi] = 'Vertical'
    #         tab['StartTime'][fi] = st.strftime('%Y%m%d%H%M%S.%f')
    #         tab['TimeRange0'][fi] = st.strftime('%Y%m%d%H%M%S.%f')
    #         tab['TimeRange1'][fi] = et.strftime('%Y%m%d%H%M%S.%f')
    #         tab['SamplingPeriod'][fi] = dt
    #         tab['Type'][fi] = 'Waveform'
    #         tab['NSamples'][fi] = npnts
    #         tab['CountPerVolt'][fi] = 1
    #         tab['VoltPerUnit'][fi] = 1
    #         tab['AmplitudeUnit'][fi] = 'Velocity'
    #         tab['NumberInFile'][fi] = 0
    #         if coord:
    #             tab['ReceiverX'][fi] = coord[fi, 0]
    #             tab['ReceiverY'][fi] = coord[fi, 1]
    #             if coord.shape[1] == 3:
    #                 tab['ReceiverZ'][fi] = coord[fi, 2]
    #         elif head[0][47] != -12345 and head[0][48] != -12345:
    #             tab['ReceiverX'][fi] = head[0][47]
    #             tab['ReceiverY'][fi] = head[0][48]
    #             if head[0][49] != -12345:
    #                 tab['ReceiverZ'][fi] = head[0][49]
    #         tab['UtmZone'][fi] = '###'
    #         tab['OffsetInFile'][fi] = 632
    #         tab['ByteIncrement'][fi] = 0
    #
    #     self.fromTab(tab, cleanup)


# class HVfile:
#     """Class for loading hv files using hvsrpy, depends on readHV function"""
#     # TODO: remove dependency on readHV?
#     def __init__(self, hv: Hvsr or list, distribution_mc='log-normal', distribution_f0='log-normal', fromfile=None):
#         if fromfile:
#             self.read(fromfile)
#         else:
#             if type(hv) is list:
#                 amp = np.array([ihv.meanHV for ihv in hv])
#                 frq = hv[0].frequency
#                 hv = Hvsr(amp, frq)
#                 self.name = 'group'
#             else:
#                 self.name = hv.meta['File Name']
#             self.frequency = hv.frq
#             self.meanHV = hv.mean_curve(distribution_mc)
#             std = hv.std_curve(distribution_mc)
#             if distribution_mc == 'log-normal':
#                 self.minHV = self.meanHV / np.exp(std)
#                 self.maxHV = self.meanHV * np.exp(std)
#             else:
#                 self.minHV = self.meanHV - std
#                 self.maxHV = self.meanHV + std
#             self.meta = dict.fromkeys(['nwin', 'f0avg', 'nwin_f0', 'f0win', 'peak_amp', 'position', 'category'])
#             self.meta['nwin'] = hv.n_windows
#             self.meta['f0avg'] = hv.mc_peak_frq(distribution_f0)
#             self.meta['nwin_f0'] = hv.n_windows
#             self.meta['f0win'] = hv._master_peak_frq
#             self.meta['peak_amp'] = hv.mean_f0_amp(distribution_f0)
#             self.meta['category'] = 'Default'
#
#     def write(self, filename=None, filepath='/Users/tenqei/Desktop'):
#         if filename is None and self.name is not None:
#             filename = filepath+'/{}.hv'.format(self.name)
#         else:
#             assert type(filename) is str
#         with open(filename, 'w') as f:
#             f.writelines('# GEOPSY output version 0.0\n')
#             f.writelines('# Number of windows = {}\n'.format(self.meta['nwin']))
#             f.writelines('# f0 from average\t{}\n'.format(self.meta['f0avg']))
#             f.writelines('# Number of windows for f0 = {}\n'.format(self.meta['nwin_f0']))
#             f.writelines('# f0 from windows\t{}\n'.format('\t'.join([str(x) for x in self.meta['f0win']])))
#             f.writelines('# Peak amplitude\t{}\n'.format(self.meta['peak_amp']))
#             if self.meta['position'] is not None and len(self.meta['position']) == 3:
#                 f.writelines('# Position\t{}\n'.format(' '.join([str(x) for x in self.meta['position']])))
#             else:
#                 f.writelines('# Position\t0 0 -12345\n')
#             f.writelines('# Category\t{}\n'.format(self.meta['category']))
#             f.writelines('# Frequency\tAverage\tMin\tMax\n')
#             for i in range(len(self.frequency)):
#                 f.writelines('{}\t{}\t{}\t{}\n'.format(self.frequency[i], self.meanHV[i], self.minHV[i], self.maxHV[i]))
#             f.close()
#         print('Wrote file: ' + filename)
#
#     def read(self, file):
#         self.name = Path(file).stem
#         h, t = readHV(file)
#         self.meta = dict.fromkeys(['nwin', 'f0avg', 'nwin_f0', 'f0win', 'peak_amp', 'position', 'category'])
#         self.meta['nwin'] = h['n_win']
#         self.meta['f0avg'] = h['f0']
#         self.meta['nwin_f0'] = h['n_win_f0']
#         self.meta['f0win'] = h['f0_n']
#         self.meta['peak_amp'] = h['amp']
#         self.meta['position'] = h['position']
#         self.meta['category'] = h['category']
#         self.frequency = t[:, 0]
#         self.meanHV = t[:, 1]
#         self.minHV = t[:, 2]
#         self.maxHV = t[:, 3]


def crc32(filepath: str):
    """Calculate crc32 filesize verification"""
    block_size = 512 * 1024
    crc = 0
    try:
        fd = open(filepath, 'rb')
        buffer = fd.read(block_size)
        if len(buffer) == 0:  # EOF or file empty. return hashes
            fd.close()
        crc = zlib.crc32(buffer, crc)
        return crc  # 返回的是十进制的值
    except Exception as e:
        error = str(e)
        return 0, error


def cvt2str(num):
    """Convert int, float, ndarray value to Geopsy acceptable string

    Parameters
    ----------
    num
        Any pre-converted value

    Returns
    -------
    str
        Geopsy acceptable string
    """
    if type(num) is str:
        return num
    elif type(num) is list or type(num) is np.ndarray:
        if all(type(x) is bool for x in num):
            return ' '.join([str(int(x)) for x in num])
        else:
            return ' '.join(['-12345' if np.isnan(x) else format(x, '.3f') for x in num])
    else:
        if np.isnan(num):
            return '-12345'
        elif 'float' in type(num).__name__:
            return format(num, '.3f')
        elif type(num) is bool:
            return str(num).lower()
        else:
            return str(num)


def if_contain_chaos(keyword: str):
    """Verify if any chaos character in string"""
    try:
        keyword.encode("gb2312")
    except UnicodeEncodeError:
        return True
    return False


def loadXML(file, encoding='utf-16'):
    """Try opening (gzippped) XML file

    Parameters
    ----------
    file: str
        XML filename
    encoding: str
        Usually 'utf-8', 'utf-16'
    Returns
    -------
    minidom.Element
        xml.dom.minidom.Element
    """
    with open(file, 'rb') as temp:
        x = temp.read()
    try:
        y = gzip.decompress(x)
    except OSError:
        y = x
    z = y.decode(encoding, 'ignore').strip('\x00').replace('\x00', '')
    txt = z[z.find('<'):]
    if if_contain_chaos(txt):
        z = y.decode('utf-8', 'ignore').strip('\x00').replace('\x00', '')
        txt = z[z.find('<'):]
    rootdata = minidom.parseString(txt).documentElement
    return rootdata


def readPAGE(file):
    """Read Geopsy page file

    Parameters
    ----------
    file: str
        PAGE filename

    Returns
    -------
    tuple[list[np.recarray], list[float]]
        List of SPAC curves and their ring radii
    """
    rootdata = loadXML(file, 'utf-8')
    ElementList = rootdata.getElementsByTagName("RealStatisticalLine")
    sel = [i for i in range(len(ElementList)) if ElementList[i].getAttribute('index') == '0']
    temp = rootdata.getElementsByTagName('TextEdit')
    ringtext = [temp[i].getElementsByTagName('text')[0].firstChild.data for i in range(len(temp))]
    r = [round((float(a[(a.index('from') + 4):a.index(' m', a.index('from'))]) +
                float(a[(a.index('to') + 2):a.index(' m', a.index('to'))])) / 2, 2) for a in ringtext]
    data = []
    for i in range(len(sel)):
        RawResult = ElementList[sel[i]].getElementsByTagName('points')[0].firstChild.data
        TESTDATA = StringIO(RawResult.strip('\n'))
        data.append(np.recfromtxt(TESTDATA, names=['freq', 'spac', 'stddev']))
    return data, r


def readHV(file):
    """Read Geospy hv file

    Parameters
    ----------
    file: str
        HV filename

    Returns
    -------
    tuple[dict, np.ndarray]
        HV header dictionary and curve data
    """
    with open(file, encoding='utf-8') as f0:
        tmp = f0.readlines()
    head = [x for x in tmp if x[0] == '#']
    tab = [x for x in tmp if x[0] != '#']
    keys = ['version', 'n_win', 'f0', 'n_win_f0', 'f0_n', 'amp', 'position', 'category', 'tab_head']
    h0 = dict.fromkeys(keys)
    if sum(['GEOPSY output version' in x for x in head]) == 1:
        ind = ['GEOPSY output version' in x for x in head].index(True)
        h0['version'] = float(re.split('[\t =]', head[ind].strip('\n'))[-1])
    if sum(['Number of windows =' in x for x in head]) == 1:
        ind = ['Number of windows =' in x for x in head].index(True)
        h0['n_win'] = int(re.split('[\t =]', head[ind].strip('\n'))[-1])
    if sum(['f0 from average' in x for x in head]) == 1:
        ind = ['f0 from average' in x for x in head].index(True)
        h0['f0'] = float(re.split('[\t =]', head[ind].strip('\n'))[-1])
    if sum(['Number of windows for f0' in x for x in head]) == 1:
        ind = ['Number of windows for f0' in x for x in head].index(True)
        h0['n_win_f0'] = int(re.split('[\t =]', head[ind].strip('\n'))[-1])
    if sum(['f0 from windows' in x for x in head]) == 1:
        ind = ['f0 from windows' in x for x in head].index(True)
        h0['f0_n'] = [float(x) for x in head[ind].strip('\n').split('\t')[1:]]
    if sum(['amplitude' in x for x in head]) == 1:
        ind = ['amplitude' in x for x in head].index(True)
        h0['amp'] = float(head[ind].split('\t')[-1])
    if sum(['Position' in x for x in head]) == 1:
        ind = ['Position' in x for x in head].index(True)
        h0['position'] = [float(x) for x in head[ind].split('\t')[-1].split(' ')]
    if sum(['Category' in x for x in head]) == 1:
        ind = ['Category' in x for x in head].index(True)
        h0['category'] = head[ind].split('\t')[-1][:-1]
    if sum(['Average' in x for x in head]) == 1:
        ind = ['Average' in x for x in head].index(True)
        h0['tab_head'] = head[ind][2:].split('\t')
    t0 = np.loadtxt(StringIO(''.join(tab)))
    return h0, t0


def readMAX(file):
    """Read Geopsy max file, the output of FK calculation (header skipped)

    Parameters
    ----------
    file: str
        MAX filename

    Returns
    -------
    np.recarray
        Geopsy FK peak table
    """
    with open(file) as f0:
        tmp = f0.readlines()
    ind0 = tmp.index('# BEGIN DATA\n')
    head = tmp[ind0 + 2].strip('# |\n').split(' ')
    tab = np.recfromtxt(StringIO(''.join(tmp[ind0 + 3:])), names=head, encoding='utf-8')
    return tab
