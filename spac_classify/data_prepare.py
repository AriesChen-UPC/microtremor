# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-02-27 3:58 PM
@File     : data_prepare.py
@Software : PyCharm
"""

from glob import glob
import pandas as pd
import numpy as np
from read_spacTarget import read_spacTarget
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Read the Excel file which contains all the data information

df = pd.read_excel(
    r'D:\MyProject\Python\PycharmProjects\DataProcessing\Microtremor\TestProgramSet\Pytorch\project\ClassifyMicrotremor.xlsx')  # Data used for training and testing, including the CD13-01 data

# Get the number of arrays which is not nan
num_array = df['array'].count()

# Get the index number of the df['array'] which the value is not nan
mask = df['array'].notna()
array_indices = df.loc[mask, 'array'].apply(lambda x: isinstance(x, str)).index

# Get each array's number
array_index = []
for i in range(len(array_indices)):
    if i < len(array_indices) - 1:
        array_index.append(array_indices[i + 1] - array_indices[i])
    else:
        array_index.append(len(df['array']) - array_indices[i])

# %% Get the SPAC data

# Get the folder path of each SPAC array
array_path = []
for i in range(len(array_indices)):
    array_path.append(df['spacPath'][array_indices[i]])

# Read the data of each SPAC array
array_data = []
for i in range(len(array_path)):
    # Find all the .target file in the folder
    spac_file = glob(array_path[i] + '/*.target')
    # Read the data of each .target file
    for j in range(len(spac_file)):
        array_data.append(read_spacTarget(spac_file[j]))

# Get the column name of the data
column_name = array_data[0].columns
# Extract the float number of the column name
ring_name = column_name.str.extract(r'(\d*\.\d+|\d+)', expand=False).astype(float)
# Print the ring name which is not nan
print(ring_name[ring_name.notna()])

# Get the specify ring data
ring_data = []
for i in range(len(array_data)):
    ring_data.append(array_data[i]['ring-3.0'])

# Change the list to pandas.DataFrame format
ring_data = pd.DataFrame(ring_data)
# Rename the row name by the spacArray
ring_data.index = df['spacArray']
# Rename the column name by the frequency
ring_data.columns = array_data[0].freq

# Slice the data by frequency [4,30]
spac_slice_data = ring_data.loc[:, 4.0:30.0].values

# Standardization the data using sklearn
scaler = StandardScaler()
spac_slice_data = scaler.fit_transform(spac_slice_data)

# %% Get the categories

category_data = df['categories'].values
# Change the categories to float number
roman_to_float = {"Ⅰ": 0.0, "Ⅱ": 1.0, "Ⅲ": 2.0}
category_data = np.array([roman_to_float[i] for i in category_data])
# Calculate the number of each category
category_num = np.bincount(category_data.astype(int))

# %% SPAC Data balance
# The data in slice_data which was labeld 0 in category_data
slice_data_0 = spac_slice_data[category_data == 0.0]
slice_data_0 = np.repeat(slice_data_0, 60, axis=0)
# The data in slice_data which was labeld 1 in category_data
slice_data_1 = spac_slice_data[category_data == 1.0]
slice_data_1 = np.repeat(slice_data_1, 570, axis=0)
# The data in slice_data which was labeld 2 in category_data
slice_data_2 = spac_slice_data[category_data == 2.0]
slice_data_2 = np.repeat(slice_data_2, 380, axis=0)

# Combine the data
slice_data = np.concatenate((slice_data_0, slice_data_1, slice_data_2), axis=0)
# Combine the category data
category_data = np.concatenate((np.zeros(len(slice_data_0)), np.ones(len(slice_data_1)),
                                np.ones(len(slice_data_2)) * 2), axis=0)

# Select 90% of the data as training data randomly
train_data, test_data, train_labels, test_labels = train_test_split(slice_data, category_data, test_size=0.1,
                                                                    random_state=42)

# %% Get the HVSR data

# # Get the folder path of each HVSR array
# hvsr_path = []
# for i in range(len(array_indices)):
#     hvsr_path.append(df['hvsrPath'][array_indices[i]])
#
# # Read the data of each HVSR array
# hvsr_data = []
# count = 0
# for i in range(len(hvsr_path)):
#     # Extract the keyword of the .hv file name in df['hvsrArray'] of each row
#     for j in range(array_index[i]):
#         keyword = df['hvsrArray'][count].split('、')
#         # Find all the .hv file in the folder
#         hvsr_file = glob(hvsr_path[i] + '/*.hv')
#         # Read the .hv data which the file name contains the keyword
#         hvsr_avg = []
#         for k in range(len(hvsr_file)):
#             for l in range(len(keyword)):
#                 if keyword[l] in hvsr_file[k]:
#                     hvsr_avg.append(pd.read_csv(hvsr_file[k], sep='\s+', skiprows=9, header=None))
#         count += 1
#         # TODO: Calculate the average of the data
#         hvsr_avg_mean = np.zeros((len(hvsr_avg), len(hvsr_avg[0])))
#         for m in range(len(hvsr_avg)):
#             hvsr_avg_mean[m, :] = hvsr_avg[m].iloc[:, 1].values
#         hvsr_avg_mean = np.mean(hvsr_avg_mean, axis=0)
#         hvsr_data.append(hvsr_avg_mean)
#
# # Change the list to pandas.DataFrame format
# hvsr_data = pd.DataFrame(hvsr_data)
# # Rename the row name by the spacArray
# hvsr_data.index = df['spacArray']
# # Rename the column name by the frequency
# hvsr_data.columns = hvsr_avg[0].iloc[:, 0].values
# # Slice the data by frequency [2,20]
# hvsr_slice_data = hvsr_data.loc[:, 2.0:20.0].values

# %% Combine the SPAC and HVSR data

# # TODO: Create an array, which element is a 2*133 array for example
# arr_num = len(array_data)
# arr = np.empty((arr_num,), dtype=np.ndarray)
# for i in range(arr_num):
#     arr[i] = np.zeros((2, 133))
#     arr[i][0, :] = spac_slice_data[i, :]
#     arr[i][1, :] = hvsr_slice_data[i, :]

# %% Combine the SPAC and HVSR data

# data_combine = np.concatenate((spac_slice_data, hvsr_slice_data), axis=1)  # Concatenate the data
