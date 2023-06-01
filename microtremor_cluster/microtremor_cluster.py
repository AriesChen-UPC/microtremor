# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 11/23/2022 8:49 AM
@File     : microtremor_cluster.py
@Software : PyCharm
"""

from sklearn import *
from gooey import Gooey, GooeyParser
from sklearn.cluster import KMeans
from kmeans_para import kmeans_para
from read_spacTarget import read_spacTarget
from read_spac import read_spac
from spac_kmeans_clustering_plotly import spac_kmeans_clustering_plotly
from theory_spac import theory_spac
from read_hv import read_hv
from hv_kmeans_clustering_plotly import hv_kmeans_clustering_plotly
from vs30_spac import vs30_spac

import warnings
warnings.filterwarnings("ignore")
running = True


@Gooey(optional_cols=2, program_name="Microtremor Cluster", default_size=(800, 600))
def main():
    settings_msg = 'Cluster Settings for SPAC and HV results of Microtremor'
    parser = GooeyParser(description=settings_msg)
    subs = parser.add_subparsers(help='commands', dest='command')

    spac_parser = subs.add_parser('SPAC', help='SPAC')
    spac_parser.add_argument('spac_path',
                             help='Target file path',
                             type=str, widget='DirChooser')
    spac_parser.add_argument('--spac_minfreq', help='Min frequency selected to cluster SPAC',
                             type=float, default=2)
    spac_parser.add_argument('--spac_maxfreq', help='Max frequency selected to cluster SPAC',
                             type=float, default=20)
    spac_parser.add_argument('--spac_ring', help='The ring of SPAC',
                             type=float)
    spac_parser.add_argument('--vs', help='VS reference',
                             type=float, default=1800)
    spac_parser.add_argument('--spac_cluster_num', help='Enter the number of clusters')

    hv_parser = subs.add_parser('HV', help='HV')
    hv_parser.add_argument('hv_path',
                             help='HV file path',
                             type=str, widget='DirChooser')
    hv_parser.add_argument('--hv_minfreq', help='Min frequency selected to cluster HV',
                             type=float, default=2)
    hv_parser.add_argument('--hv_maxfreq', help='Max frequency selected to cluster HV',
                             type=float, default=10)
    hv_parser.add_argument('--hv_cluster_num', help='Enter the number of clusters')

    vs30_parser = subs.add_parser('VS30', help='VS30')
    vs30_parser.add_argument('vs30_path',
                                help='VS30 file path',
                                type=str, widget='DirChooser')

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = main()
    print(args.command)
    if args.command == 'SPAC':
        folder_path, fs, names, spac, freq, spac_filter, freq_filter, radius, min_freq, max_freq = \
            read_spacTarget(folder_path=args.spac_path, min_freq=args.spac_minfreq, max_freq=args.spac_maxfreq,
                       radius=args.spac_ring)
        vs_reference = args.vs
        freq_theory_spac, spac_theory_spac = theory_spac(folder_path, radius, vs_reference)
        if args.spac_cluster_num is None:
            spac_cluster_num = kmeans_para(spac_filter, folder_path)
        else:
            spac_cluster_num = int(args.spac_cluster_num)
        kmeans = KMeans(n_clusters=spac_cluster_num, random_state=42)
        kmeans_clustering = kmeans.fit(spac_filter)

        spac_kmeans_clustering_plotly(folder_path, fs, freq, spac, names, freq_theory_spac, spac_theory_spac,
                                      min_freq, max_freq, radius, vs_reference, kmeans_clustering, spac_cluster_num)
    elif args.command == 'HV':
        folder_path, fs, names, hvsr_data, hvsr_freq, data_sel = read_hv(folder_path=args.hv_path,
                                                                         min_freq=args.hv_minfreq, max_freq=args.hv_maxfreq)
        if args.hv_cluster_num is None:
            hv_cluster_num = kmeans_para(data_sel, folder_path)
        else:
            hv_cluster_num = int(args.hv_cluster_num)
        kmeans = KMeans(n_clusters=hv_cluster_num, random_state=42)
        kmeans_clustering = kmeans.fit(data_sel)

        hv_kmeans_clustering_plotly(folder_path, fs, hvsr_freq, hvsr_data, data_sel, names, kmeans_clustering, hv_cluster_num)
    elif args.command == 'VS30':
        folder_path = args.vs30_path
        vs30_spac(folder_path)
