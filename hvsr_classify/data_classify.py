# encoding: UTF-8
"""
@Author   : AriesChen
@Email    : s15010125@s.upc.edu.cn
@Time     : 2023-02-27 1:11 PM
@File     : data_classify.py
@Software : PyCharm
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import time


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # Use CPU
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # Use GPU
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


# %% Define training parameters

# input_size = signals_all.shape[-1]  # The input size should be the same as the length of the ECG data
input_size = 116
hidden_size = 100  # The more neurons, the more GPU memory is required
num_layers = 3  # The more layers, the more GPU memory is required
num_classes = 3

model = BiLSTM(input_size, hidden_size, num_layers, num_classes)

num_epochs = 1000
learning_rate = 0.001

# Define loss functions and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %% Start training

# Combine SPAC and HVSR data which element is a 2*133 array for example
# # FIXME: TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32,
# #        float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
# train_data = torch.from_numpy(train_data)

# Converting data to Tensor and passing it into the model
# TODO: The train_data and train_labels are from the previous section 'data_preoare.py'
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

inputs = torch.unsqueeze(train_data, dim=1)
labels = train_labels

# Use GPU if available
# Input and hidden tensors are not at the same device, found input tensor at cuda:0 and hidden tensor at cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

# Write the training process to tensorboard
writer = SummaryWriter()

# Calculate the time of training
start_time_gpu = time.time()

# Set training weights for different labels
weights = torch.tensor([1, 1, 1], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)

print("Start training...")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode, which enables dropout
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # loss = criterion(outputs, labels)
    writer.add_scalar('training loss', loss.item(), epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # if loss.item() < 0.01:
    #     break

# Close the tensorboard. Using tensorboard --logdir=runs to start the tensorboard
writer.close()
print("Training time: {:.2f} s".format(time.time() - start_time_gpu))

with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the training data is: {} %'.format(round(100 * correct / total, 3)))

# Calculate the confusion matrix
_, pred = torch.max(outputs, 1)
pred = pred.cpu().numpy()
labels = labels.cpu().numpy()
conf_matrix = confusion_matrix(labels, pred)
plt.figure(figsize=(8, 8))
LABELS = ['Normal', 'Little Abnormal', 'Abnormal']
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion matrix for training data")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('confusion_matrix_train.png')
plt.show()

# %% Evaluate the accuracy of the model on testing data

# Converting data to Tensor and passing it into the model
# TODO: The test_data and test_labels are from the previous section 'data_preoare.py'
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_inputs = torch.unsqueeze(test_data, dim=1).to(device)
test_labels = test_labels.to(device)

with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    print('Accuracy of the model on the testing data is: {} %'.format(round(100 * correct / total, 3)))

# %% Evaluation System

_, pred = torch.max(outputs, 1)
pred = pred.cpu().numpy()
test_labels = test_labels.cpu().numpy()

# Calculate the sensitivity and specificity
TP = np.sum((pred == 0) & (test_labels == 0))
TN = np.sum((pred == 1) & (test_labels == 1))
FP = np.sum((pred == 0) & (test_labels == 1))
FN = np.sum((pred == 1) & (test_labels == 0))
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Sensitivity of testing data is: {:.2f} %".format(sensitivity * 100))
print("Specificity of testing data is: {:.2f} %".format(specificity * 100))  # TODO: The specificity is too low

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_labels, pred)
plt.figure(figsize=(8, 8))
LABELS = ['Normal', 'Little Abnormal', 'Abnormal']
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion matrix for testing data")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('confusion_matrix_test.png')
plt.show()

# %% Save the model

# TODO: Train the model for ground model depth estimation
# torch.save(model.state_dict(), 'model.ckpt')
