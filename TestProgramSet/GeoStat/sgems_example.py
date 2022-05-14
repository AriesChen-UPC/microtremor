# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/11/2022 9:49 AM
@file: sgems_example.py
"""

import os
from os.path import join as join_path
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg

# Initiate sgems project
cwd = os.getcwd()  # Working directory
rdir = join_path(cwd, 'results', 'demo_kriging')  # Results directory
pjt = sg.Sgems(project_name='sgems_test', project_wd=cwd, res_dir=rdir)
# Load data point set
data_dir = join_path(cwd, 'datasets', 'demo_kriging')
dataset = 'sgems_dataset.eas'
file_path = join_path(data_dir, dataset)

ps = PointSet(project=pjt, pointset_path=file_path)
# Generate grid. Grid dimensions can automatically be generated based on the data points
# unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
ds = Discretize(project=pjt, dx=5, dy=5)
# Display point coordinates and grid
pl = Plots(project=pjt)
pl.plot_coordinates()
