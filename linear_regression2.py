#!/usr/bin/env python
# coding=utf-8

import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average = True)
optimizer_1 = torch.optim.Adagrad(model.parameters(), lr = 0.01)
optimizer_2 = torch.optim.Adam(model.parameters(), lr = 0.01)
optimizer_3 = torch.optim.Adamax(model.parameters(), lr = 0.01)
optimizer_4 = torch.optim.ASGD(model.parameters(), lr = 0.01)


temp_1 = []
temp_2 = []
temp_3 = []
temp_4 = []
epoch_1 = []
epoch_2 = []
epoch_3 = []
epoch_4 = []


for epoch in range(1000):
    y_pred_1 = model(x_data)
    loss_1 = criterion(y_pred_1, y_data)
    temp_1.append(loss_1.item())
    epoch_1.append(epoch)
    optimizer_1.zero_grad()
    loss_1.backward()
    optimizer_1.step()

for epoch in range(1000):
    y_pred_2 = model(x_data)
    loss_2 = criterion(y_pred_2, y_data)
    temp_2.append(loss_2.item())
    epoch_2.append(epoch)
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()

for epoch in range(1000):
    y_pred_3 = model(x_data)
    loss_3 = criterion(y_pred_3, y_data)
    temp_3.append(loss_3.item())
    epoch_3.append(epoch)
    optimizer_3.zero_grad()
    loss_3.backward()
    optimizer_3.step()

for epoch in range(1000):
    y_pred_4 = model(x_data)
    loss_4 = criterion(y_pred_4, y_data)
    temp_4.append(loss_4.item())
    epoch_4.append(epoch)
    optimizer_4.zero_grad()
    loss_4.backward()
    optimizer_4.step()

plt.subplot(221)
plt.plot(epoch_1, temp_1)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Adagrad')


plt.subplot(222)
plt.plot(epoch_2, temp_2)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Adam')

plt.subplot(223)
plt.plot(epoch_3, temp_3)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Adamax')

plt.subplot(224)
plt.plot(epoch_4, temp_4)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('ASGD')

plt.show()
