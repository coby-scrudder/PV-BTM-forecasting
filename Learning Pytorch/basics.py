import torch
import numpy as np
import torch.nn as nn

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
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.05
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()

    # update weights outside of gradient tracking
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')