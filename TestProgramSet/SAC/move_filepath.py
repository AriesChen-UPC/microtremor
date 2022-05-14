# encoding: UTF-8
"""
@author: LijiongChen
@contact: s15010125@s.upc.edu.cn
@time: 3/16/2022 6:50 PM
@file: move_filepath.py
"""

import os
import shutil
import tkinter as tk
from glob import glob
from tkinter import filedialog
import colorama

#%% select the destination folder

colorama.init(autoreset=True)
print('\033[0;36mPlease select the folder to store the grouped sac data：\033[0m')
root = tk.Tk()
root.withdraw()
sac_destination_folder_path = filedialog.askdirectory()
print('The folder where the grouped data were stored is：', sac_destination_folder_path)
print('------------------------------')
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
print('------------------------------')
file_name = glob(sac_orignal_folder_path + '/*.SAC')
# move the sac file and judge the folder whether exist
#%%
is_A_exist = 0
is_A1_exist = 0
is_A2_exist = 0
is_A3_exist = 0
is_A4_exist = 0
is_A5_exist = 0
is_A6_exist = 0
is_B_exist = 0
is_B1_exist = 0
is_B2_exist = 0
is_B3_exist = 0
is_B4_exist = 0
is_B5_exist = 0
is_B6_exist = 0
is_C_exist = 0
is_C1_exist = 0
is_C2_exist = 0
is_C3_exist = 0
is_C4_exist = 0
is_C5_exist = 0
is_C6_exist = 0
is_D_exist = 0
is_D1_exist = 0
is_D2_exist = 0
is_D3_exist = 0
is_D4_exist = 0
is_D5_exist = 0
is_D6_exist = 0
is_E_exist = 0
is_E1_exist = 0
is_E2_exist = 0
is_E3_exist = 0
is_E4_exist = 0
is_E5_exist = 0
is_E6_exist = 0
#%%
for i in range(len(file_name)):
    # group A
    if os.path.basename(file_name[i])[0:9] == "590001076":  # fixme: A1 is changed. Original is 590000037
        shutil.move(file_name[i], A1_sac_data_path)
        is_A1_exist = is_A1_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000050":
        shutil.move(file_name[i], A2_sac_data_path)
        is_A2_exist = is_A2_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000059":
        shutil.move(file_name[i], A3_sac_data_path)
        is_A3_exist = is_A3_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000064":
        shutil.move(file_name[i], A4_sac_data_path)
        is_A4_exist = is_A4_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000066":
        shutil.move(file_name[i], A5_sac_data_path)
        is_A5_exist = is_A5_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000099":
        shutil.move(file_name[i], A6_sac_data_path)
        is_A6_exist = is_A6_exist + 1
    # group B
    if os.path.basename(file_name[i])[0:9] == "590000102":
        shutil.move(file_name[i], B1_sac_data_path)
        is_B1_exist = is_B1_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000105":
        shutil.move(file_name[i], B2_sac_data_path)
        is_B2_exist = is_B2_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000106":
        shutil.move(file_name[i], B3_sac_data_path)
        is_B3_exist = is_B3_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000111":
        shutil.move(file_name[i], B4_sac_data_path)
        is_B4_exist = is_B4_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000342":
        shutil.move(file_name[i], B5_sac_data_path)
        is_B5_exist = is_B5_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000343":
        shutil.move(file_name[i], B6_sac_data_path)
        is_B6_exist = is_B6_exist + 1
    # group C
    if os.path.basename(file_name[i])[0:9] == "590000345":
        shutil.move(file_name[i], C1_sac_data_path)
        is_C1_exist = is_C1_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000351":
        shutil.move(file_name[i], C2_sac_data_path)
        is_C2_exist = is_C2_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000358":
        shutil.move(file_name[i], C3_sac_data_path)
        is_C3_exist = is_C3_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000372":
        shutil.move(file_name[i], C4_sac_data_path)
        is_C4_exist = is_C4_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000397":
        shutil.move(file_name[i], C5_sac_data_path)
        is_C5_exist = is_C5_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000400":
        shutil.move(file_name[i], C6_sac_data_path)
        is_C6_exist = is_C6_exist + 1
    # group D
    if os.path.basename(file_name[i])[0:9] == "590000425":
        shutil.move(file_name[i], D1_sac_data_path)
        is_D1_exist = is_D1_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000446":
        shutil.move(file_name[i], D2_sac_data_path)
        is_D2_exist = is_D2_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000450":
        shutil.move(file_name[i], D3_sac_data_path)
        is_D3_exist = is_D3_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000458":
        shutil.move(file_name[i], D4_sac_data_path)
        is_D4_exist = is_D4_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000472":
        shutil.move(file_name[i], D5_sac_data_path)
        is_D5_exist = is_D5_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000587":
        shutil.move(file_name[i], D6_sac_data_path)
        is_D6_exist = is_D6_exist + 1
    # group E
    if os.path.basename(file_name[i])[0:9] == "590000075":
        shutil.move(file_name[i], E1_sac_data_path)
        is_E1_exist = is_E1_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000080":
        shutil.move(file_name[i], E2_sac_data_path)
        is_E2_exist = is_E2_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000360":
        shutil.move(file_name[i], E3_sac_data_path)
        is_E3_exist = is_E3_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000894":
        shutil.move(file_name[i], E4_sac_data_path)
        is_E4_exist = is_E4_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000904":
        shutil.move(file_name[i], E5_sac_data_path)
        is_E5_exist = is_E5_exist + 1
    if os.path.basename(file_name[i])[0:9] == "590000964":
        shutil.move(file_name[i], E6_sac_data_path)
        is_E6_exist = is_E6_exist + 1
# delete the empty folder
is_A_exist = is_A1_exist + is_A2_exist + is_A3_exist + is_A4_exist + is_A5_exist + is_A6_exist
is_B_exist = is_B1_exist + is_B2_exist + is_B3_exist + is_B4_exist + is_B5_exist + is_B6_exist
is_C_exist = is_C1_exist + is_C2_exist + is_C3_exist + is_C4_exist + is_C5_exist + is_C6_exist
is_D_exist = is_D1_exist + is_D2_exist + is_D3_exist + is_D4_exist + is_D5_exist + is_D6_exist
is_E_exist = is_E1_exist + is_E2_exist + is_E3_exist + is_E4_exist + is_E5_exist + is_E6_exist
if is_A_exist == 0:
    print('\033[0;31mGroup A data does not exist!\033[0m')
    shutil.rmtree(A_sac_data_path)
else:
    print('The file number of A is', is_A_exist)
    if is_A1_exist < 3:
        print('\033[0;31mThe file path of A1 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A1 is complete!\033[0m')
    if is_A2_exist < 3:
        print('\033[0;31mThe file path of A2 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A2 is complete!\033[0m')
    if is_A3_exist < 3:
        print('\033[0;31mThe file path of A3 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A3 is complete!\033[0m')
    if is_A4_exist < 3:
        print('\033[0;31mThe file path of A4 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A4 is complete!\033[0m')
    if is_A5_exist < 3:
        print('\033[0;31mThe file path of A5 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A5 is complete!\033[0m')
    if is_A6_exist < 3:
        print('\033[0;31mThe file path of A6 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of A6 is complete!\033[0m')
print('------------------------------')
if is_B_exist == 0:
    print('\033[0;31mGroup B data does not exist!\033[0m')
    shutil.rmtree(B_sac_data_path)
else:
    print('The file number of B is', is_B_exist)
    if is_B1_exist < 3:
        print('\033[0;31mThe file path of B1 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B1 is complete!\033[0m')
    if is_B2_exist < 3:
        print('\033[0;31mThe file path of B2 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B2 is complete!\033[0m')
    if is_B3_exist < 3:
        print('\033[0;31mThe file path of B3 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B3 is complete!\033[0m')
    if is_B4_exist < 3:
        print('\033[0;31mThe file path of B4 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B4 is complete!\033[0m')
    if is_B5_exist < 3:
        print('\033[0;31mThe file path of B5 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B5 is complete!\033[0m')
    if is_B6_exist < 3:
        print('\033[0;31mThe file path of B6 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of B6 is complete!\033[0m')
print('------------------------------')
if is_C_exist == 0:
    print('\033[0;31mGroup C data does not exist!\033[0m')
    shutil.rmtree(C_sac_data_path)
else:
    print('The file number of C is', is_C_exist)
    if is_C1_exist < 3:
        print('\033[0;31mThe file path of C1 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C1 is complete!\033[0m')
    if is_C2_exist < 3:
        print('\033[0;31mThe file path of C2 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C2 is complete!\033[0m')
    if is_C3_exist < 3:
        print('\033[0;31mThe file path of C3 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C3 is complete!\033[0m')
    if is_C4_exist < 3:
        print('\033[0;31mThe file path of C4 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C4 is complete!\033[0m')
    if is_C5_exist < 3:
        print('\033[0;31mThe file path of C5 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C5 is complete!\033[0m')
    if is_C6_exist < 3:
        print('\033[0;31mThe file path of C6 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of C6 is complete!\033[0m')
print('------------------------------')
if is_D_exist == 0:
    print('\033[0;31mGroup D data does not exist!\033[0m')
    shutil.rmtree(D_sac_data_path)
else:
    print('The file number of D is', is_D_exist)
    if is_D1_exist < 3:
        print('\033[0;31mThe file path of D1 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D1 is complete!\033[0m')
    if is_D2_exist < 3:
        print('\033[0;31mThe file path of D2 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D2 is complete!\033[0m')
    if is_D3_exist < 3:
        print('\033[0;31mThe file path of D3 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D3 is complete!\033[0m')
    if is_D4_exist < 3:
        print('\033[0;31mThe file path of D4 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D4 is complete!\033[0m')
    if is_D5_exist < 3:
        print('\033[0;31mThe file path of D5 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D5 is complete!\033[0m')
    if is_D6_exist < 3:
        print('\033[0;31mThe file path of D6 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of D6 is complete!\033[0m')
print('------------------------------')
if is_E_exist == 0:
    print('\033[0;31mGroup E data does not exist!\033[0m')
    shutil.rmtree(E_sac_data_path)
else:
    print('The file number of E is', is_E_exist)
    if is_E1_exist < 3:
        print('\033[0;31mThe file path of E1 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E1 is complete!\033[0m')
    if is_E2_exist < 3:
        print('\033[0;31mThe file path of E2 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E2 is complete!\033[0m')
    if is_E3_exist < 3:
        print('\033[0;31mThe file path of E3 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E3 is complete!\033[0m')
    if is_E4_exist < 3:
        print('\033[0;31mThe file path of E4 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E4 is complete!\033[0m')
    if is_E5_exist < 3:
        print('\033[0;31mThe file path of E5 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E5 is complete!\033[0m')
    if is_E6_exist < 3:
        print('\033[0;31mThe file path of E6 is not complete!\033[0m')
    else:
        print('\033[0;32mThe file path of E6 is complete!\033[0m')
print('------------------------------')
print('End of file movement！')
input('Press Enter to exit … \n')
