# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 4/16/2022 10:56 AM
@file: geopsy_disp.py
"""

import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def cps_disp(folder_path, model_plot, freq_theory_spac, R_matrix_S):

    pio.templates.default = "plotly_white"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ground Model", "Dispersion Curve"))
    # plot the theory SPAC
    fig.add_trace(go.Scatter(x=model_plot['vs'], y=model_plot['depth'], name='Ground Model',
                             mode='lines', showlegend=False,
                             line=dict(
                                 color='rgb(139,134,130)',
                                 width=2)),
                  row=1, col=1)
    fig.update_yaxes(title_text='Depth (m)', range=[0, model_plot['depth'].max()], row=1, col=1, autorange='reversed')
    fig.update_xaxes(title_text='Vs (m/s)', row=1, col=1)
    fig.add_trace(go.Scatter(x=freq_theory_spac['freq'], y=1/R_matrix_S['rayleigh'], name='Dispersion Curve',
                             mode='lines', showlegend=False,
                             line=dict(
                                 color='rgb(255,127,36)',
                                 width=2)),
                  row=1, col=2)
    fig.update_yaxes(title_text='Vs (m/s)', row=1, col=2)
    fig.update_xaxes(title_text='Frequency (Hz)', range=[0, 100], tick0=0.0, dtick=10, row=1, col=2)
    fig.update_layout(title='CPS')
    fig.update_layout(
        showlegend=False,  #set the legend to be hidden
        hoverlabel=dict(
            namelength=-1,  # set the name length
        )
    )
    default_html_name = folder_path + '/' + 'cps_disp.html'
    plotly.offline.plot(fig, filename=default_html_name)