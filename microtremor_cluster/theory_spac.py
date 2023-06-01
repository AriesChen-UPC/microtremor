# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/11/2022 11:12 AM
@file: theory_spac.py
"""

import math
import numpy as np
from pandas import DataFrame
from scipy.special import jn
import plotly
import plotly.graph_objects as go
import plotly.io as pio


def theory_spac(folder_path, radius, vs_reference):
    """
    Args:
        folder_path: the path of the folder which stores the spac file.
        radius: the radius of the spac file.
        vs_reference: the reference velocity of the dispersion curve.
    Returns:
        freq_theory_spac: the frequency of the theory spac.
        spac_theory_spac: the spac of the theory spac.
    """
    r = radius
    freq_theory_spac = np.logspace(np.log10(0.1), np.log10(100), num=400)
    vs = np.dot(vs_reference, [math.pow(f, -0.65) for f in freq_theory_spac])
    spac_theory_spac = jn(0, np.multiply(r * 2 * math.pi * freq_theory_spac, [math.pow(v, -1) for v in vs]))
    freq_theory_spac = DataFrame(freq_theory_spac)
    freq_theory_spac.columns = ['freq']
    spac_theory_spac = DataFrame(spac_theory_spac)
    spac_theory_spac.columns = ['autoCorrRatio']
    vs = DataFrame(vs)
    vs.columns = ['vs']
    # plot the initialized model
    # pio.templates.default = "plotly_white"  # set the plotly templates
    # fig = go.Figure(data=go.Scatter(x=freq_theory_spac['freq'], y=vs['vs'], name='dispersion curve'))
    # fig.update_xaxes(title_text="Frequency (Hz)")
    # fig.update_yaxes(title_text="Velocity (m/s)")
    # fig.update_xaxes(type="log")
    # fig.update_xaxes(range=[np.log10(1), np.log10(100)])
    # fig.update_yaxes(range=[0, 3000], tick0=0.0, dtick=500)
    # fig.update_layout(title='vs(nearly 1Hz, depth 57.5m)=' + str(vs_reference) + 'm/s')
    # plotly.offline.plot(fig, filename=folder_path + '/' + 'vs_reference.html')
    print('Dispersion curve and SPAC curve calculated by eIndex were done!')
    return freq_theory_spac, spac_theory_spac