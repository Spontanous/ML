#!/usr/bin/env python
# coding=utf-8

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([2.0])
b = torch.Tensor([0.1])


w.requires_grad = True   #require weight to gradient when the program iteration
b.requires_grad = True


#define output function
def forward(x):
    return x * w + b

#define loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2    

print("perdict (before training)", 4, forward(4).item())

#epoch in a fe-forward network

for epoch in range(100):
    for x, y in zip(x_data, y_data):  #zip: pack a & b into a tuple
        l = loss(x, y)
        l.backward()
        print("\tgrad", x, y, w.grad.item(), b.grad.item())
        w.data = w.data - 0.01 * w.grad.data  #update weight
        b.data = b.data - 0.01 * b.grad.data  #update bias

        w.grad.data.zero_() #clear the weight data of the gradient when iteration 
        b.grad.data.zero_()

    print("process", epoch, l.item())

print("predict (after training)", 4, forward(4).item())

