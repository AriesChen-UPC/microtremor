# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 6/25/2022 4:21 PM
@file: disp_cal.py
"""

from dispCal.disp import calDisp
import numpy as np

thickness = np.array([10, 22, 12., 0])
vs = np.array([3, 3.5, 4, 4.3])
vp = vs * 1.7
rho = vp / 3
periods = np.arange(1, 30, 1).astype(np.float64)


'''
    calDisp(thickness, vp, vs, rho, periods,dc0=0.005, domega=0.0001, wave='rayleigh', mode=1, velocity='phase', 
           flat_earth=True, ar=6370.0, parameter='vp', smoothN=1)
    velocity: phase group kernel kernelGroup
    wave: love rayleigh
    parameter: vp vs rho thickness
    domega: for cal group and delta_omega=omega*domega
    dc0: is for search phase velocity and cal sensitive kernel and group velocity. For sensitive kernel 
         dcUsed = dc0/10; for group velocity, dcUsed=dc0/100
    domega: for group velocitu and its kernel; the real step is given by domega*omega0;
            for calculating group velocity's' kernels, domegaUsed =domega/100, dcUsed=dc0/5
    mode: from 1
    ar : the radius for flatting
    smoothN: a new feature under construction
'''
# phase velocity
velocities = calDisp(thickness, vp, vs, rho, periods, wave='love', mode=1, velocity='phase', flat_earth=True,
                     ar=6370, dc0=0.005)
# group velocity
velocities = calDisp(thickness, vp, vs, rho, periods,wave='love', mode=1, velocity='group', flat_earth=True,
                     ar=6370, dc0=0.005)
# phase velocity kernel
KS = calDisp(thickness, vp, vs, rho, periods, wave='rayleigh', mode=1, velocity='kernel', flat_earth=True,
             ar=6370, dc0=0.005, domega=0.0001, parameter='vs')
KP = calDisp(thickness, vp, vs, rho, periods, wave='rayleigh', mode=1, velocity='kernel', flat_earth=True,
             ar=6370, dc0=0.005, domega=0.0001, parameter='vp')
# group velocity kernel
KSG = calDisp(thickness, vp, vs, rho, periods, wave='rayleigh', mode=1, velocity='kernelGroup',
              flat_earth=True, ar=6370, dc0=0.005, domega=0.0001, parameter='vs')
KPG = calDisp(thickness, vp, vs, rho, periods, wave='rayleigh', mode=1, velocity='kernelGroup',
              flat_earth=True, ar=6370, dc0=0.005, domega=0.0001, parameter='vp')
