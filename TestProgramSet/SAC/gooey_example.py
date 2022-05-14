# encoding: UTF-8
"""
@author: AriesChen
@contact: s15010125@s.upc.edu.cn
@time: 5/11/2022 10:33 AM
@file: gooey_example.py
"""

import os
import shutil
from glob import glob
from gooey import Gooey, GooeyParser


@Gooey(program_name='SAC Group', required_cols=1)
def main():
    parser = GooeyParser(description="This program is used to group the raw SAC data!")
    parser.add_argument('DataPath', help="Please select the orignal SAC data path", widget="DirChooser")
    parser.add_argument('GroupPath', help="Please select the group SAC data path", widget="DirChooser")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()

    sac_orignal_folder_path = args.DataPath  # get the orignal SAC data path
    file_name = glob(sac_orignal_folder_path + '/*.SAC')  # get the file name in the orignal SAC data path
    sac_destination_folder_path = args.GroupPath  # get the group SAC data path
    # create the group SAC data path
    sac_group = {'A': sac_destination_folder_path + '/A', 'B': sac_destination_folder_path + '/B',
                 'C': sac_destination_folder_path + '/C', 'D': sac_destination_folder_path + '/D',
                 'E': sac_destination_folder_path + '/E', 'F': sac_destination_folder_path + '/F',
                 'G': sac_destination_folder_path + '/G'}
    sac_subfolder = {}
    for path in sac_group.values():
        if not os.path.exists(path):
            os.makedirs(path)
            for i in range(1, 7):  # each group has 6 subfolders, from 1-6
                sac_subfolder[os.path.basename(path) + str(i)] = path + '/' + str(i)
                if not os.path.exists(sac_subfolder[os.path.basename(path) + str(i)]):
                    os.makedirs(sac_subfolder[os.path.basename(path) + str(i)])

    # %% define the info of instrument and station

    # fixme: A1 is changed. Original serial number is 590000037
    instrument_serial = {'A1': '590001076', 'A2': '590000050', 'A3': '590000059', 'A4': '590000064',
                         'A5': '590000066', 'A6': '590000099',
                         'B1': '590000102', 'B2': '590000105', 'B3': '590000106', 'B4': '590000111',
                         'B5': '590000342', 'B6': '590000343',
                         'C1': '590000345', 'C2': '590000351', 'C3': '590000358', 'C4': '590000372',
                         'C5': '590000397', 'C6': '590000400',
                         'D1': '590000425', 'D2': '590000446', 'D3': '590000450', 'D4': '590000458',
                         'D5': '590000472', 'D6': '590000587',
                         'E1': '590000075', 'E2': '590000080', 'E3': '590000360', 'E4': '590000894',
                         'E5': '590000904', 'E6': '590000964',
                         'F1': '590001101', 'F2': '590001170', 'F3': '590001172', 'F4': '590001184',
                         'F5': '590001186', 'F6': '590001231',
                         'G1': '590001240', 'G2': '590001243', 'G3': '590001300', 'G4': '590001347',
                         'G5': '590001366', 'G6': '590001379'}
    instrument_num = {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0,
                      'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0,
                      'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0,
                      'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0,
                      'E1': 0, 'E2': 0, 'E3': 0, 'E4': 0, 'E5': 0, 'E6': 0,
                      'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0,
                      'G1': 0, 'G2': 0, 'G3': 0, 'G4': 0, 'G5': 0, 'G6': 0}

    # %% move the sac file and judge the folder whether exist

    def get_key(dict_, value_):  # get the key of a value from a dict
        return [k for k, v in dict_.items() if v == value_]


    for i in range(len(file_name)):
        instrument_key = get_key(instrument_serial, os.path.basename(file_name[i])[0:9])
        instrument_num[instrument_key[0]] += 1
        shutil.move(file_name[i], sac_subfolder[instrument_key[0]])
    # print the result information
    for key, value in instrument_num.items():
        print('The number of sac files in Group %s is %d' % (key, value))
    print('------------------------------')
    # delete the empty subfolders
    for subFolder in sac_subfolder.values():
        if not os.listdir(subFolder):
            print('The %s is empty' % subFolder)
            shutil.rmtree(subFolder)
    print('------------------------------')
    # delete the empty group folders
    for sacGroup in sac_group.values():
        if not os.listdir(sacGroup):
            print('The %s is empty' % sacGroup)
            shutil.rmtree(sacGroup)
    print('------------------------------')
