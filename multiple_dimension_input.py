#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np

x_csv = np.loadtxt('diabetes_data.csv', delimiter = ' ', dtype = np.float32)
x_data = torch.from_numpy(x_csv[:, :])


y_csv = np.loadtxt('diabetes_target.csv', dtype = np.float32)
y_data = torch.from_numpy(y_csv)
y_data = y_data.view(442, 1)

class Mutile_model(torch.nn.Module):
    def __init__(self):
        super(Mutile_model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Mutile_model()

criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch = ', epoch, 'loss = ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



