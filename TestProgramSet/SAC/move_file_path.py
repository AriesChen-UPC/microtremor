# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 3/16/2022 6:50 PM
@file: move_file_path.py
"""

import os
import shutil
import tkinter as tk
from glob import glob
from tkinter import filedialog

#%% select the destination folder

print('\033[0;36mPlease select the folder to store the grouped sac data：\033[0m')
root = tk.Tk()
root.withdraw()
sac_destination_folder_path = filedialog.askdirectory()
print('The folder where the grouped data were stored is：', sac_destination_folder_path)
# data path of A group
A_sac_data_path = sac_destination_folder_path + '/A'
if not os.path.exists(A_sac_data_path):
    os.mkdir(sac_destination_folder_path + '/A')
A1_sac_data_path = A_sac_data_path + '/1'
if not os.path.exists(A1_sac_data_path):
    os.mkdir(A_sac_data_path + '/1')
A2_sac_data_path = A_sac_data_path + '/2'
if not os.path.exists(A2_sac_data_path):
    os.mkdir(A_sac_data_path + '/2')
A3_sac_data_path = A_sac_data_path + '/3'
if not os.path.exists(A3_sac_data_path):
    os.mkdir(A_sac_data_path + '/3')
A4_sac_data_path = A_sac_data_path + '/4'
if not os.path.exists(A4_sac_data_path):
    os.mkdir(A_sac_data_path + '/4')
A5_sac_data_path = A_sac_data_path + '/5'
if not os.path.exists(A5_sac_data_path):
    os.mkdir(A_sac_data_path + '/5')
A6_sac_data_path = A_sac_data_path + '/6'
if not os.path.exists(A6_sac_data_path):
    os.mkdir(A_sac_data_path + '/6')
# data path of B group
B_sac_data_path = sac_destination_folder_path + '/B'
if not os.path.exists(B_sac_data_path):
    os.mkdir(sac_destination_folder_path + '/B')
B1_sac_data_path = B_sac_data_path + '/1'
if not os.path.exists(B1_sac_data_path):
    os.mkdir(B_sac_data_path + '/1')
B2_sac_data_path = B_sac_data_path + '/2'
if not os.path.exists(B2_sac_data_path):
    os.mkdir(B_sac_data_path + '/2')
B3_sac_data_path = B_sac_data_path + '/3'
if not os.path.exists(B3_sac_data_path):
    os.mkdir(B_sac_data_path + '/3')
B4_sac_data_path = B_sac_data_path + '/4'
if not os.path.exists(B4_sac_data_path):
    os.mkdir(B_sac_data_path + '/4')
B5_sac_data_path = B_sac_data_path + '/5'
if not os.path.exists(B5_sac_data_path):
    os.mkdir(B_sac_data_path + '/5')
B6_sac_data_path = B_sac_data_path + '/6'
if not os.path.exists(B6_sac_data_path):
    os.mkdir(B_sac_data_path + '/6')
# data path of C group
C_sac_data_path = sac_destination_folder_path + '/C'
if not os.path.exists(C_sac_data_path):
    os.mkdir(sac_destination_folder_path + '/C')
C1_sac_data_path = C_sac_data_path + '/1'
if not os.path.exists(C1_sac_data_path):
    os.mkdir(C_sac_data_path + '/1')
C2_sac_data_path = C_sac_data_path + '/2'
if not os.path.exists(C2_sac_data_path):
    os.mkdir(C_sac_data_path + '/2')
C3_sac_data_path = C_sac_data_path + '/3'
if not os.path.exists(C3_sac_data_path):
    os.mkdir(C_sac_data_path + '/3')
C4_sac_data_path = C_sac_data_path + '/4'
if not os.path.exists(C4_sac_data_path):
    os.mkdir(C_sac_data_path + '/4')
C5_sac_data_path = C_sac_data_path + '/5'
if not os.path.exists(C5_sac_data_path):
    os.mkdir(C_sac_data_path + '/5')
C6_sac_data_path = C_sac_data_path + '/6'
if not os.path.exists(C6_sac_data_path):
    os.mkdir(C_sac_data_path + '/6')
# data path of D group
D_sac_data_path = sac_destination_folder_path + '/D'
if not os.path.exists(D_sac_data_path):
    os.mkdir(sac_destination_folder_path + '/D')
D1_sac_data_path = D_sac_data_path + '/1'
if not os.path.exists(D1_sac_data_path):
    os.mkdir(D_sac_data_path + '/1')
D2_sac_data_path = D_sac_data_path + '/2'
if not os.path.exists(D2_sac_data_path):
    os.mkdir(D_sac_data_path + '/2')
D3_sac_data_path = D_sac_data_path + '/3'
if not os.path.exists(D3_sac_data_path):
    os.mkdir(D_sac_data_path + '/3')
D4_sac_data_path = D_sac_data_path + '/4'
if not os.path.exists(D4_sac_data_path):
    os.mkdir(D_sac_data_path + '/4')
D5_sac_data_path = D_sac_data_path + '/5'
if not os.path.exists(D5_sac_data_path):
    os.mkdir(D_sac_data_path + '/5')
D6_sac_data_path = D_sac_data_path + '/6'
if not os.path.exists(D6_sac_data_path):
    os.mkdir(D_sac_data_path + '/6')
# data path of E group
E_sac_data_path = sac_destination_folder_path + '/E'
if not os.path.exists(E_sac_data_path):
    os.mkdir(sac_destination_folder_path + '/E')
E1_sac_data_path = E_sac_data_path + '/1'
if not os.path.exists(E1_sac_data_path):
    os.mkdir(E_sac_data_path + '/1')
E2_sac_data_path = E_sac_data_path + '/2'
if not os.path.exists(E2_sac_data_path):
    os.mkdir(E_sac_data_path + '/2')
E3_sac_data_path = E_sac_data_path + '/3'
if not os.path.exists(E3_sac_data_path):
    os.mkdir(E_sac_data_path + '/3')
E4_sac_data_path = E_sac_data_path + '/4'
if not os.path.exists(E4_sac_data_path):
    os.mkdir(E_sac_data_path + '/4')
E5_sac_data_path = E_sac_data_path + '/5'
if not os.path.exists(E5_sac_data_path):
    os.mkdir(E_sac_data_path + '/5')
E6_sac_data_path = E_sac_data_path + '/6'
if not os.path.exists(E6_sac_data_path):
    os.mkdir(E_sac_data_path + '/6')

#%% get the sac file path

print('\033[0;36mPlease select the original sac data folder：\033[0m')
# print('Please select the orignal sac file path: \n')
root = tk.Tk()
root.withdraw()
sac_orignal_folder_path = filedialog.askdirectory()
print('The original sac data folder is', sac_orignal_folder_path)
file_name = glob(sac_orignal_folder_path + '/*.SAC')
# move the sac file and judge the folder whether exist
is_A_exist = False
is_B_exist = False
is_C_exist = False
is_D_exist = False
is_E_exist = False
for i in range(len(file_name)):
    # group A
    if os.path.basename(file_name[i])[0:9] == "590000037":
        shutil.move(file_name[i], A1_sac_data_path)
        is_A_exist = True
    if os.path.basename(file_name[i])[0:9] == "590000050":
        shutil.move(file_name[i], A2_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000059":
        shutil.move(file_name[i], A3_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000064":
        shutil.move(file_name[i], A4_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000066":
        shutil.move(file_name[i], A5_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000099":
        shutil.move(file_name[i], A6_sac_data_path)
    # group B
    if os.path.basename(file_name[i])[0:9] == "590000102":
        shutil.move(file_name[i], B1_sac_data_path)
        is_B_exist = True
    if os.path.basename(file_name[i])[0:9] == "590000105":
        shutil.move(file_name[i], B2_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000106":
        shutil.move(file_name[i], B3_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000111":
        shutil.move(file_name[i], B4_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000342":
        shutil.move(file_name[i], B5_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000343":
        shutil.move(file_name[i], B6_sac_data_path)
    # group C
    if os.path.basename(file_name[i])[0:9] == "590000345":
        shutil.move(file_name[i], C1_sac_data_path)
        is_C_exist = True
    if os.path.basename(file_name[i])[0:9] == "590000351":
        shutil.move(file_name[i], C2_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000358":
        shutil.move(file_name[i], C3_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000372":
        shutil.move(file_name[i], C4_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000397":
        shutil.move(file_name[i], C5_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000400":
        shutil.move(file_name[i], C6_sac_data_path)
    # group D
    if os.path.basename(file_name[i])[0:9] == "590000425":
        shutil.move(file_name[i], D1_sac_data_path)
        is_D_exist = True
    if os.path.basename(file_name[i])[0:9] == "590000446":
        shutil.move(file_name[i], D2_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000450":
        shutil.move(file_name[i], D3_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000458":
        shutil.move(file_name[i], D4_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000472":
        shutil.move(file_name[i], D5_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000587":
        shutil.move(file_name[i], D6_sac_data_path)
    # group E
    if os.path.basename(file_name[i])[0:9] == "590000075":
        shutil.move(file_name[i], E1_sac_data_path)
        is_E_exist = True
    if os.path.basename(file_name[i])[0:9] == "590000080":
        shutil.move(file_name[i], E2_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000360":
        shutil.move(file_name[i], E3_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000894":
        shutil.move(file_name[i], E4_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000904":
        shutil.move(file_name[i], E5_sac_data_path)
    if os.path.basename(file_name[i])[0:9] == "590000964":
        shutil.move(file_name[i], E6_sac_data_path)
# delete the empty folder
if is_A_exist == False:
    shutil.rmtree(A_sac_data_path)
if is_B_exist == False:
    shutil.rmtree(B_sac_data_path)
if is_C_exist == False:
    shutil.rmtree(C_sac_data_path)
if is_D_exist == False:
    shutil.rmtree(D_sac_data_path)
if is_E_exist == False:
    shutil.rmtree(E_sac_data_path)
print('End of file movement！')
input('Press Enter to exit … \n')
