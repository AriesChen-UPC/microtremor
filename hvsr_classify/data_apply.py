# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-03-01 8:35 AM
@File     : data_apply.py
@Software : PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xml.dom import minidom


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


def read_spacTarget(file_path):
    """
    Args:
        file_path: SPAC file path
    Returns:
        sapc_All: SPAC data in pandas.DataFrame format
    """
    xml_file = minidom.parse(file_path)
    xml_elements = xml_file.documentElement
    sub_ringElements = xml_elements.getElementsByTagName('AutocorrRing')
    sub_curveElement = xml_elements.getElementsByTagName('ModalCurve')  # Child nodes which contains curves
    data_length = len(sub_ringElements)
    ring_data = []
    for i in range(data_length):
        ring_data.append((float(sub_ringElements[i].getElementsByTagName('minRadius')[0].firstChild.data) +
                          float(sub_ringElements[i].getElementsByTagName('maxRadius')[0].firstChild.data)) / 2)

    curve_length = len(sub_curveElement[0].getElementsByTagName('mean'))
    # The frequency is same for all rings, so we only need to read one ring
    x_data = []
    for j in range(curve_length):
        x_data.append(float(sub_curveElement[0].getElementsByTagName('x')[j].firstChild.data))
    # Read all rings, and store them in a list
    curve_data = [[] for i in range(data_length)]
    for i in range(data_length):
        for j in range(curve_length):
            curve_data[i].append(float(sub_curveElement[i].getElementsByTagName('mean')[j].firstChild.data))
    # Store the data in pandas.DataFrame format
    spac_data = pd.DataFrame(curve_data).T
    spac_data.columns = ['ring-' + str(r) for r in ring_data]
    freq_data = pd.DataFrame(x_data)
    freq_data.columns = ['freq']
    spac_All = pd.concat([freq_data, spac_data], axis=1)
    return spac_All

# %% Load the model

input_size = 116  # The number of frequency points from 2.0 to 50.0 Hz
hidden_size = 100
num_layers = 3
num_classes = 3

model = BiLSTM(input_size, hidden_size, num_layers, num_classes)

# Load the model
model.load_state_dict(torch.load(
    r"D:\MyProject\Python\PycharmProjects\DataProcessing\Microtremor\TestProgramSet\Pytorch\Project\ChengDu\model_4_30_std.ckpt"))
model.eval()

# %% Load the data

root = tk.Tk()
root.withdraw()
root.update()
apply_array_path = filedialog.askdirectory()
apply_array_data = []

# Read the data of each SPAC array
for file in glob(apply_array_path + '/*.target'):
    apply_array_data.append(read_spacTarget(file))

# Get the .target file name
apply_array_name = []
for file in glob(apply_array_path + '/*.target'):
    # The file name is split by '.target'
    apply_array_name.append(file.split('.target')[0].split('\\')[-1])

select_data = []
for i in range(len(apply_array_data)):
    # read the first ring data of each SPAC array
    select_data.append(apply_array_data[i].iloc[:, 1].values)

# Convert the data to a dataframe and select the frequency range of 2.0-20.0 GHz
select_data = pd.DataFrame(select_data)
select_data.columns = apply_array_data[0].freq
select_data.index = apply_array_name
slice_select_data = select_data.loc[:, 4.0:30.0].values  # The frequency is distributed by logarithm
# If the data length is more than 185, the data is truncated
if slice_select_data.shape[1] > 116:
    slice_select_data = slice_select_data[:, :116]
# If the data length is less than 185, the data is filled with 0
elif slice_select_data.shape[1] < 116:
    slice_select_data = np.pad(slice_select_data, ((0, 0), (0, 116 - slice_select_data.shape[1])), 'constant')

# Standardization the data using sklearn
scaler = StandardScaler()
slice_select_data = scaler.fit_transform(slice_select_data)

# %% Apply the model

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Apply the model
slice_select_data = torch.tensor(slice_select_data, dtype=torch.float32)
slice_select_data = torch.unsqueeze(slice_select_data, dim=1).to(device)
with torch.no_grad():
    outputs = model(slice_select_data)
    _, predicted = torch.max(outputs.data, 1)
    # print('The predicted label is: {}'.format(predicted))

# Change the predicted label to the original label
predicted = predicted.cpu().numpy()
label_dict = {0: 'Ⅰ', 1: 'Ⅱ', 2: 'Ⅲ'}
predicted = [label_dict[i] for i in predicted]
result = pd.DataFrame({'SPAC array name': apply_array_name, 'Predicted label': predicted})
print(result)
# Save the result to a csv file in the apply_array_path
result.to_csv(apply_array_path + '/result.csv', index=False, encoding='utf_8_sig')
