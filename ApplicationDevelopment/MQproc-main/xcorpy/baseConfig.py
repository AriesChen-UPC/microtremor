from pathlib import Path

# server connection setup
cfgupdate = False
cfgadd = "http://localhost:12009"
cfgkey = "seraagalngdniPdfjalfffasdd25lasdfaa93"

# data path setup
cfgroot = Path('/Volumes/disk/testdir')
# optional dir:
# Local : '/Users/tenqei/Desktop/MQ2021/testdir'
# SMB: '/Volumes/disk/testdir'
# Server: '/mnt/aa7baf70-d871-4169-89ba-674e2edfecad/testdir'
if not cfgroot.exists():
    cfgroot.mkdir()
cfgdata = str(cfgroot.joinpath('DATA'))
cfgpara = str(cfgroot.joinpath('PARA'))
cfgprep = str(cfgroot.joinpath('PREP'))
cfgresl = str(cfgroot.joinpath('RESL'))

# geopsy path
binroot = Path('/home/server/Documents/geopsypack-src-3.3.6/bin')
if binroot.is_dir():
    bingpy = str(binroot.joinpath('geopsy'))
    binspac = str(binroot.joinpath('geopsy-spac'))
    binfk = str(binroot.joinpath('geopsy-fk'))
    binhvsr = str(binroot.joinpath('geopsy-hv'))
else:
    bingpy = 'geopsy'
    binspac = 'geopsy-spac'
    binfk = 'geopsy-fk'
    binhvsr = 'geopsy-hv'
