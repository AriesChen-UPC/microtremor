# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/21/2022 3:46 PM
@file: pygeostat_example.py
"""

import pygeostat as gs
import numpy as np
import os
from matplotlib import pyplot as plt

outdir = 'D:/MyProject/Python/PycharmProjects/DataProcessing/Microtremor/TestProgramSet/GeoStat/Output'
gs.mkdir(outdir)
# path to GSLIB executables
exe_dir = "E:/anaconda3/Lib/site-packages/pygeostat/executable/"  # fixme： CCG/GSLIB software is the CCG memebers'  option

gs.PlotStyle['font.size'] = 12
gs.Parameters['data.tmin'] = -998

dfl = gs.ExampleData('point2d_surf')
dfl.head()
dfl.info
dfl.describe()
for var in dfl.variables:
    gs.histogram_plot(dfl, var=var, figsize=(7, 4))
    plt.show()
_ = gs.scatter_plots(dfl)
plt.show()
fig, axes = gs.subplots(1, len(dfl.variables), axes_pad=(0.9, 0.4), figsize=(25, 5), cbar_mode='each', label_mode='L')
for i, var in enumerate(dfl.variables):
    gs.location_plot(dfl, var=var, ax=axes[i])
plt.show()
nscore_p = gs.Program(program=exe_dir + 'nscore.exe', getpar=True)
parstr = """      Parameters for NSCORE
                  *********************

START OF PARAMETERS:
{datafile}               -  file with data
{n_var}  4 5 6           -  number of variables and columns
0                         -  column for weight, 0 if none
0                         -  column for category, 0 if none
0                         -  number of records if known, 0 if unknown
{tmin}   1.0e21          -  trimming limits
0                         -transform using a reference distribution, 1=yes
nofile.out                -file with reference distribution.
1   2   0                 -  columns for variable, weight, and category
201                       -maximum number of quantiles, 0 for all
{outfl}                -file for output
{trnfl}                -file for output transformation table
"""
nscore_outfl = os.path.join(outdir, 'nscore.out')

pars = dict(datafile=dfl.flname,
            tmin=gs.Parameters['data.tmin'],
            n_var=len(dfl.variables),
            outfl=nscore_outfl,
            trnfl=os.path.join(outdir, 'nscore.trn'))
nscore_p.run(parstr=parstr.format(**pars), quiet=True, liveoutput=True)  # fixme: ERROR in parameter file
dfl_ns = gs.DataFile(nscore_outfl)
dfl_ns.head()
for var in dfl_ns.variables:
    if 'ns' in var.lower():
        gs.histogram_plot(dfl_ns, var=var, color='g', figsize=(7, 4))
        plt.show()
dfl_ns.spacing(n_nearest=2)
dfl_ns.head()
lag_length_h = dfl_ns['Data Spacing (m)'].values.mean()
print('average data spacing in XY plane: {:.3f} {}'.format(lag_length_h,
                                                           gs.Parameters['plotting.unit']))
x_range = np.ptp(dfl[dfl.x].values)
y_range = np.ptp(dfl[dfl.y].values)
n_lag_x = np.ceil((x_range * 0.5) / lag_length_h)
n_lag_y = np.ceil((y_range * 0.5) / lag_length_h)
lag_tol_h = lag_length_h * 0.6
var_calc = gs.Program(program=exe_dir + 'varcalc')
parstr = """      Parameters for VARCALC
                  **********************

START OF PARAMETERS:
{file}                             -file with data
2 3 0                              -   columns for X, Y, Z coordinates
1 7                                -   number of variables,column numbers (position used for tail,head variables below)
{t_min}    1.0e21                   -   trimming limits
{n_directions}                                  -number of directions
0.0 15 1000 0.0 22.5 1000 0.0   -Dir 01: azm,azmtol,bandhorz,dip,diptol,bandvert,tilt
 {n_lag_y}  {lag_length_h}  {lag_tol_h}            -        number of lags,lag distance,lag tolerance
90.0 15 1000 0.0 22.5 1000 0.0   -Dir 02: azm,azmtol,bandhorz,dip,diptol,bandvert,tilt
 {n_lag_x}  {lag_length_h}  {lag_tol_h}                 -        number of lags,lag distance,lag tolerance
{output}                          -file for experimental variogram points output.
0                                 -legacy output (0=no, 1=write out gamv2004 format)
1                                 -run checks for common errors
1                                 -standardize sills? (0=no, 1=yes)
1                                 -number of variogram types
1   1   1   1                     -tail variable, head variable, variogram type (and cutoff/category), sill
"""

n_directions = 2
varcalc_outfl = os.path.join(outdir, 'varcalc.out')

var_calc.run(parstr=parstr.format(file=dfl_ns.flname,
                                  n_directions=n_directions,
                                  t_min=gs.Parameters['data.tmin'],
                                  n_lag_x=n_lag_x,
                                  n_lag_y=n_lag_y,
                                  lag_length_h=lag_length_h,
                                  lag_tol_h=lag_tol_h,
                                  output=varcalc_outfl),
             liveoutput=True)
varfl = gs.DataFile(varcalc_outfl)
varfl.head()
colors = gs.get_palette('cat_dark', n_directions, cmap=False)
titles = ['Major', 'Minor', 'Vertical']
fig, axes = plt.subplots(1, n_directions, figsize=(20, 4))
for i in range(n_directions):
    gs.variogram_plot(varfl, index=i + 1, ax=axes[i], color=colors[i], title=titles[i], grid=True)
var_model = gs.Program(program=exe_dir + 'varmodel')
parstr = """      Parameters for VARMODEL
                  ***********************

START OF PARAMETERS:
{varmodel_outfl}             -file for modeled variogram points output
3                            -number of directions to model points along
0.0   0.0  100   25          -  azm, dip, npoints, point separation
90.0   0.0  100   15       -  azm, dip, npoints, point separation
0.0   90.0  100   0.2       -  azm, dip, npoints, point separation
2    0.05                   -nst, nugget effect
3    ?    0.0   0.0   0.0    -it,cc,azm,dip,tilt (ang1,ang2,ang3)
        ?     ?     ?    -a_hmax, a_hmin, a_vert (ranges)
3    ?    0.0   0.0   0.0    -it,cc,azm,dip,tilt (ang1,ang2,ang3)
        ?     ?     ?    -a_hmax, a_hmin, a_vert (ranges)
1   100000                   -fit model (0=no, 1=yes), maximum iterations
1.0                          -  variogram sill (can be fit, but not recommended in most cases)
1                            -  number of experimental files to use
{varcalc_outfl}              -    experimental output file 1
3 1 2 3                    -      # of variograms (<=0 for all), variogram #s
1   0   10                   -  # pairs weighting, inverse distance weighting, min pairs
0     10.0                   -  fix Hmax/Vert anis. (0=no, 1=yes)
0      1.0                   -  fix Hmin/Hmax anis. (0=no, 1=yes)
{varmodelfit_outfl}          -  file to save fit variogram model
"""

varmodel_outfl = os.path.join(outdir, 'varmodel.out')
varmodelfit_outfl = os.path.join(outdir, 'varmodelfit.out')

var_model.run(parstr=parstr.format(varmodel_outfl=varmodel_outfl,
                                   varmodelfit_outfl=varmodelfit_outfl,
                                   varcalc_outfl=varcalc_outfl), liveoutput=False, quiet=True)
varmdl = gs.DataFile(varmodel_outfl)
varmdl.head()
fig, axes = plt.subplots(1, n_directions, figsize=(20, 4))
for i in range(n_directions):
    gs.variogram_plot(varfl, index=i + 1, ax=axes[i], color=colors[i], title=titles[i], grid=True)
    gs.variogram_plot(varmdl, index=i + 1, ax=axes[i], color=colors[i], experimental=False)
print(dfl_ns.infergriddef(nblk=[200, 200, 1]))
kt3dn = gs.Program(exe_dir + 'kt3dn', getpar=True)
parstr_ = """     Parameters for KT3DN
                 ********************
START OF PARAMETERS:
{file}                                -file with data
1  2 3 0 4 0                          -  columns for DH,X,Y,Z,var,sec var
{t_min}    1.0e21                     -  trimming limits
0                                     -option: 0=grid, 1=cross, 2=jackknife
nojack.out                           -file with jackknife data
0 0 0 0   0                          -   columns for X,Y,Z,vr and sec var
kt3dn_dataspacing.out                 -data spacing analysis output file (see note)
1    20.0                             -  number to search (0 for no dataspacing analysis, rec. 10 or 20) and composite length
0    100   0                          -debugging level: 0,3,5,10; max data for GSKV;output total weight of each data?(0=no,1=yes)
kt3dn.dbg-nkt3dn.sum                  -file for debugging output (see note)
{output}                              -file for kriged output (see GSB note)
{griddef}
1    1      1                         -x,y and z block discretization
20    80    12    1                    -min, max data for kriging,upper max for ASO,ASO incr
0      0                              -max per octant, max per drillhole (0-> not used)
500.0  500.0  150.0                   -maximum search radii
 0.0   0.0   0.0                      -angles for search ellipsoid
1                                     -0=SK,1=OK,2=LVM(resid),3=LVM((1-w)*m(u))),4=colo,5=exdrift,6=ICCK
0.0 0.6  0.8                          -  mean (if 0,4,5,6), corr. (if 4 or 6), var. reduction factor (if 4)
0 0 0 0 0 0 0 0 0                     -drift: x,y,z,xx,yy,zz,xy,xz,zy
0                                     -0, variable; 1, estimate trend
extdrift.out                          -gridded file with drift/mean
4                                     -  column number in gridded file
keyout.out                            -gridded file with keyout (see note)
0    1                                -  column (0 if no keyout) and value to keep
{varmodel}
"""

krig_output = os.path.join(outdir, 'KrigGrid.out')

with open(varmodelfit_outfl, 'r') as f:
    varmodel_ = f.readlines()
varmodel = ''''''
for line in varmodel_:
    varmodel += line

parstr = parstr_.format(file=dfl_ns.flname,
                        t_min=gs.Parameters['data.tmin'],
                        griddef=str(dfl_ns.griddef),
                        varmodel=varmodel,
                        output=krig_output)
kt3dn.run(parstr=parstr, liveoutput=True)
krigfl = gs.DataFile(krig_output, griddef=dfl_ns.griddef)
krigfl.head()
cmaps = ['inferno', 'jet', 'bwr', 'viridis']
fig, axes = gs.subplots(2, 2, axes_pad=(0.9, 0.4), figsize=(20, 15), cbar_mode='each', label_mode='L')
for i, ax in enumerate(axes):
    gs.slice_plot(krigfl, var='Estimate', orient='xy', cmap=cmaps[i], ax=ax, pointdata=dfl_ns,
                  pointvar='Top Elevation', pointkws={'edgecolors': 'k', 's': 25})
# Clean up
try:
    gs.rmfile('kt3dn.sum')
    gs.rmfile('kt3dn.dbg')
    gs.rmfile('kt3dn_dataspacing.out')
    gs.rmfile('temp')
    gs.rmdir(outdir)
except:
    pass
