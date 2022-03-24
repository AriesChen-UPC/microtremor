import gzip
import pandas as pd
from xml.dom import minidom
from io import StringIO
# from matplotlib import pyplot as plt
from pathlib import Path


# This code reads the .page file from the Geopsy package
# The output is a list of 3 DataFrames

def read_page(f, plot=False, write=False):
    p = Path(f)
    name = p.stem
    path = f.strip(name + '.page')

    with open(f, 'rb') as temp:
        x = temp.read()
        y = gzip.decompress(x).decode('utf-8', 'ignore').strip('\x00').replace('\x00', '')
        z = y[y.find('<SciFigs>'):]

    dom = minidom.parseString(z)
    rootdata = dom.documentElement
    ElementList = rootdata.getElementsByTagName("points")

    data = []
    for i in range(ElementList.length):
        temp0 = ElementList[i].firstChild.data
        if temp0.__len__() < 10000:  # todo: need to verify the exact num
            continue
        else:
            RawResult = ElementList[i].firstChild.data
            TESTDATA = StringIO(RawResult.strip('\n'))
            data.append(pd.read_table(TESTDATA, sep='\\s+', header=None, names=['freq', 'spac', 'stderr']))

    # if plot:
    #     plt.figure()
    #     for i in range(3):
    #         plt.subplot(3, 1, i + 1)
    #         plt.errorbar(x=data[i].freq, y=data[i].spac, yerr=data[i].stderr)
    #     plt.show()

    if write:
        for i in range(3):
            data[i].to_csv(path + name + 'r' + str(i + 1) + '.csv', index=False)

    return data, name
