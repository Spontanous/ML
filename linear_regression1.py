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

#找一个损失函数（这里采用MSE）
criterion = torch.nn.MSELoss(size_average = True)

#利用梯度下降法进行寻优（梯度方式可选择）
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

temp_list = []
epoch_list = []


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    temp_list.append(loss.item())
    epoch_list.append(epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)

print('y_test = ', y_test.item())

plt.plot(epoch_list, temp_list)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()



