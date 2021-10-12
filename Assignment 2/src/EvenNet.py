import numpy as np
import matplotlib.pyplot as mpl
import torch
import torch.nn as nn
import torch.nn.functional as func

class EvenNet(nn.Module):
   

    def __init__(self):
        super(EvenNet, self).__init__()
       
        self.inputlayer = nn.Linear(168,75)
        self.hl1 = nn.Linear(75,40)
        self.hl2 = nn.Linear(40,20)
        self.outlayer = nn.Linear(20,5)

       
    
    def forward(self, x):
        x = self.inputlayer(x)
        x = self.relu(x)

        h = self.hl1(x)
        h = self.relu(x)

        g = self.hl2(h)
        g = self.relu(g)
        
        y = torchself.outlayer(g)

        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.inputlayer.reset_parameters()
        self.hl1.reset_parameters()
        self.hl2.reset_parameters()
        self.outlayer.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)

        obj_val= loss(self.forward(inputs).reshape(-1), targets)
        for param in self.model.parameters():
            param.grad = None
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)

            cross_val= loss(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()

test = EvenNet()