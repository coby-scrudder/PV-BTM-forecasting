import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
# import torch.nn.functional as F
# from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import math
# from torch.utils.tensorboard import SummaryWriter
import sys
import os
# import statistics
# import csv
import math
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

data_directory = 'D:/REU Research/Trial Combined Data/'

time_start = time.time()

mm = MinMaxScaler()
ss = StandardScaler()

num_training_csvs = 5
num_test_csvs = 1

count = 0

for csv in os.listdir(data_directory):
    df = pd.read_csv(data_directory + csv, index_col='measured_on', parse_dates=True)
    X_pre = df.iloc[:, 1:26]
    y_pre = df.iloc[:, :1]

    # Move from pandas dataframes to numpy arrays
    X_pre_ss = ss.fit_transform(X_pre)
    y_pre_mm = mm.fit_transform(y_pre)

    if count == 0:
        X_train = X_pre_ss
        y_train = y_pre_mm

    elif 1 <= count < num_training_csvs:
        X_train = np.concatenate((X_train, X_pre_ss))
        y_train = np.concatenate((y_train, y_pre_mm))

    elif num_training_csvs == count:
        X_test = X_pre_ss
        y_test = y_pre_mm

    elif (num_training_csvs + 1) <= count < (num_training_csvs + num_test_csvs):
        X_test = np.concatenate((X_test, X_pre_ss))
        y_test = np.concatenate((y_test, y_pre_mm))
    else:
        break

    count += 1

X_train_tensor = Variable(torch.Tensor(X_train))
X_test_tensor = Variable(torch.Tensor(X_test))

y_train_tensor = Variable(torch.Tensor(y_train))
y_test_tensor = Variable(torch.Tensor(y_test))

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)

y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

X_train_tensors_final = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1])).to(device)

X_test_tensors_final = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1])).to(device)


num_epochs = 1  # 1 epoch
learning_rate = 0.0001  # 0.001 lr
batch_size = 96

input_size = 25  # number of features
hidden_size = 2  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

num_classes = 1  # number of output classes


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 300)  # fully connected 1
        self.fc_2 = nn.Linear(300, 300)
        self.fc = nn.Linear(300, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h_0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
        c_0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.to(device)
        cn = cn.to(device)
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.tanh(hn)
        out = self.fc_1(out)  # first Dense
        out = self.tanh(out)  # relu
        out = self.fc_2(out)
        out = self.tanh(out)
        out = self.fc(out)  # Final Output
        return out


lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])  # our lstm class

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.SGD(lstm1.parameters(), lr=learning_rate)

losses = []
iters = []

iter_count = 0

for epoch in range(num_epochs):
    for iteration in range(math.ceil(X_train_tensor.shape[0] / batch_size)):
        start_index = iteration * batch_size
        end_index = start_index + batch_size
        X_train_tensor_iter = X_train_tensors_final[start_index:end_index]
        y_train_tensor_iter = y_train_tensor[start_index:end_index]

        length = y_train_tensor_iter.shape[0]
        for i in range(length):
            if math.isnan(y_train_tensor_iter[i][0].item()):
                y_train_tensor_iter[i][0] = y_train_tensor_iter[i - 1][0]

        outputs = lstm1.forward(X_train_tensor_iter)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensor_iter)

        loss.backward()  # calculates the loss of the loss function

        clipping_value = 1  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(lstm1.parameters(), clipping_value)

        optimizer.step()  # improve from loss, i.e. backprop
        if iteration % 100 == 0:
            print(f"Epoch: {epoch},Iteration: {iter_count}, loss: {loss.item()}, time elapsed: {time.time() - time_start}")
            iters.append(iter_count)
            losses.append(loss.item())
        iter_count += 1

plt.plot(iters, losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.show()
