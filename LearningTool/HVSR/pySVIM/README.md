# pySVIM
## Seismic Vulnerability Index of Microtremor - Microtremor Data Processing By Using Horizontal to Vertical Spectral Ratio (HVSR) Method

### Overview
A code to calculate the amplification factor and frequency dominant by using Horizontal to Vertical Spectral Ratio (HVSR) method. The HVSR methods was introduced by [Nakamura in 1989](https://www.sdr.co.jp/papers/hv_1989.pdf). I have wrote code of SVIM in Matlab version with Graphical User Interface. The code calculate seismic vulnerability index directly after the parameters of amplification and frequency dominant was obtained from HVSR calculation and plot into map of sesimic vulnerability index directly too. This code is apart of [my thesis](https://etd.unsyiah.ac.id/index.php?p=show_detail&id=66042) and [my publication](https://iopscience.iop.org/article/10.1088/1755-1315/273/1/012016) in master degree  

Unfortunately, the SVIM code in Matlab version is still under construction to get the proper performance. So, for now, I can share the SVIM code in [Python3](https://www.python.org/download/releases/3.0/) version called pySVIM. pySVIM only provide the HVSR calculation or without the Graphical User Interface and without plot of seismic vulnerablitty index map.

### The library you need to install:
1. [Obspy](https://pypi.org/project/obspy/)
2. [Numpy](https://pypi.org/project/numpy/)
3. [Pandas](https://pypi.org/project/pandas/)
4. [Scipy](https://pypi.org/project/scipy/)

I also provide the extention library that I created called hvsrlib. This library concists some functions such as:

1. nearest     : to search the nearest value from an array
2. dofilt      : to filter a waveform (band, low and high filters)
3. coswindow   : to apply the cosine windowing to the trace [Tapering of windowed time series](https://gfzpublic.gfz-potsdam.de/rest/items/item_56141/component/file_56140/content)
4. antrig      : to select the trace of microtremor based on [STA/LTA method](https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content)
5. kosmooth    : to smooth the trace by using [Konno-Ohmachi smoothing method](https://pubs.geoscienceworld.org/ssa/bssa/article/88/1/228/102764)

**This code has been tested on Python 3.7**

### Contact
auliakhalqillah.mail@gmail.com
