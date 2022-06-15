import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

# basic element-wise operations #

x = torch.rand(3, 3)
y = torch.rand(3, 3)

# addition #
a = x + y
b = torch.add(x, y)
# addition can also be done through reassigning
y.add_(x)  # underscore denotes that this operation will be done on y
y += x

# subtraction #
a = x - y
b = torch.sub(x, y)
y.sub_(x)

# multiplication #
a = x * y
b = torch.mul(x, y)
y.mul_(x)

# division #
a = x / y
b = torch.div(x, y)
y.div_(x)

# slicing #
a = x[0, :]

# reshaping #
x = torch.rand(4, 2)
y = x.view(2, 4)  # can use -1 as an input to have torch figure out dimension needed

# conversion from tensor to numpy #
a = torch.rand(5)
b = a.numpy()
# changes made to the tensor also change the numpy array
a.add_(torch.ones(5))

# conversion from numpy to tensor #
b = np.random.random(5)
a = torch.from_numpy(b)

# moving to GPU if it is available #
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.rand(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")

# requires gradient #
# When declaring a tensor, if the gradient will be calculated later, this must be added
    # so that torch will track the operations done to it so the gradient can be calculated
x = torch.ones(5, requires_grad=True)

# autograd package for calculating gradients #
x = torch.randn(3, requires_grad=True)  # randn gives normally distributed random numbers

y = x + 2
y *= x

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
y.backward(v)  # calculates dy/dx; must have an argument v if y is not a scalar value
z = x.grad
# After running the gradient command, we should zero it out so that it is accurate every iteration
x.grad.zero_()
# Example with an optimizer
# optimizer = torch.optim.SGD(x, lr=0.01, momentum=0.9)
# optimizer.step()
# optimizer.zero_grad()

# To set requires_grad=False, we have three options
# x.requires_grad_(False)
# x.detach_()
# with torch.no_grad():

# basic machine learning structure
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

# example of manual gradient descent #
# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)
#
# w = 0.0
#
# # model prediction
# def forward(x):
#     return w*x
#
# # loss
# def loss(y_act, y_predict):
#     return ((y_act - y_predict)**2).mean()
#
# # gradient
# # MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N 2 x (w*y - y)
# def gradient(x, y_act, y_predict):
#     return np.dot(2*x, y_predict - y_act).mean()
#
#
# print(f'Prediction before training: f(5) = {forward(5):.3f}')
#
# # Training
# learning_rate = 0.01
# n_iters = 100
#
# for epoch in range(n_iters):
#     # prediction = forward pass
#     y_pred = forward(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # gradients
#     dw = gradient(X, Y, y_pred)
#
#     # update weights
#     w -= learning_rate * dw
#
#     if epoch % 1 == 0:
#         print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
#
# print(f'Prediction after training: f(5) = {forward(5):.3f}')

# # replacing with Pytorch gradient automatic calculations #
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
#
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#
# # model prediction
# def forward(x):
#     return w*x
#
# # loss
# def loss(y_act, y_predict):
#     return ((y_act - y_predict)**2).mean()
#
# print(f'Prediction before training: f(5) = {forward(5)}')
#
# # Training
# learning_rate = 0.01
# n_iters = 20
#
# for epoch in range(n_iters):
#     # prediction = forward pass
#     y_pred = forward(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # gradients = backward pass
#     l.backward()
#
#     # update weights outside of gradient tracking
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#
#     # zero gradients
#     w.grad.zero_()
#
#     if epoch % 1 == 0:
#         print(f'epoch {epoch + 1}: w = {float(w):.3f}, loss = {l:.8f}')
#
# print(f'Prediction after training: f(5) = {forward(5):.3f}')

# replacing with Pytorch gradient, loss, and parameter automatic calculations #
# X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
# Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
#
# n_samples, n_features = X.shape
#
# input_size = n_features
# output_size = n_features
#
# model = nn.Linear(input_size, output_size)
#
# X_test = torch.tensor([5], dtype=torch.float32)
# print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
#
# # Training
# learning_rate = 0.05
# n_iters = 100
#
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# for epoch in range(n_iters):
#     # prediction = forward pass
#     y_pred = model(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # gradients = backward pass
#     l.backward()
#
#     # update weights outside of gradient tracking
#     optimizer.step()
#
#     # zero gradients
#     optimizer.zero_grad()
#
#     if epoch % 10 == 0:
#         [w, b] = model.parameters()
#         print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
#
# print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

# Defining your own model #
# class LinearRegression(nn.Module):
#
#     def __int__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         # define layers
#         self.lin = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.lin(x)

# Putting it all together #
# # Making the dataset
# X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
#
# X = torch.from_numpy(X_numpy.astype(np.float32))
# y = torch.from_numpy(y_numpy.astype(np.float32))
# y = y.view(y.shape[0], 1)
#
# n_samples, n_features = X.shape
#
# # Model #
#
# input_size = n_features
# output_size = 1
# model = nn.Linear(input_size, output_size)
#
# # loss and optimizer #
# learning_rate = 0.01
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# loss_list = []
# epoch_list = []
#
# # Training loop #
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # forward pass and loss
#     y_predicted = model(X)
#     loss = criterion(y_predicted, y)
#     loss_list.append(float(loss.item()))
#     epoch_list.append(float(epoch))
#
#     # backward pass
#     loss.backward()
#
#     # update
#     optimizer.step()
#
#     optimizer.zero_grad()
#
#     if (epoch+1) % 10 == 0:
#         print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot prediction vs actual #
# predicted = model(X).detach().numpy()
# plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.show()

# plot MSE Loss vs time #
# plt.plot(epoch_list, loss_list, 'b')
# plt.title('MSE Loss vs Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.show()

# Implementing logistic regression #
# # prepare the data #
# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
#
# n_samples, n_features = X.shape
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# # scale
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# X_train = torch.from_numpy(X_train.astype(np.float32))
# X_test = torch.from_numpy(X_test.astype(np.float32))
# y_train = torch.from_numpy(y_train.astype(np.float32))
# y_test = torch.from_numpy(y_test.astype(np.float32))
#
# y_train = y_train.view(y_train.shape[0], 1)
# y_test = y_test.view(y_test.shape[0], 1)
#
# # model #
# # f = wx + b, sigmoid at the end
# class LogisticRegression(nn.Module):
#
#     def __init__(self, n_input_features):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(n_input_features, 1)
#
#     def forward(self, x):
#         y_predicted = torch.sigmoid(self.linear(x))
#         return y_predicted
#
#
# model = LogisticRegression(n_features)
#
# # loss and optimizer #
# learning_rate = 0.01
# criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # training loop #
# num_epochs = 1000
# epoch_list = []
# acc_list = []
# loss_list = []
# for epoch in range(num_epochs):
#     # forward pass and loss #
#     y_predicted = model(X_train)
#     loss = criterion(y_predicted, y_train)
#
#     # backward pass #
#     loss.backward()
#
#     # updates #
#     optimizer.step()
#     optimizer.zero_grad()
#
#     if (epoch + 1) % 1 == 0:
#         print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')
#         with torch.no_grad():
#             y_predicted = model(X_test)
#             y_predicted_cls = y_predicted.round()
#             acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
#             acc_list.append(acc)
#             epoch_list.append(epoch+1)
#             loss_list.append(loss.item())

# max_acc = acc_list[-1].item()
# plt.plot(epoch_list, acc_list, 'b')
# plt.text(s=f'Ending Accuracy = {max_acc:.4f}', x=600, y=0.65)

# plt.plot(epoch_list, loss_list, 'b')
# plt.xlabel('Epoch')
# plt.ylabel('BCE Loss')
# plt.savefig('Breast_cancer_loss.png')
# plt.show()


