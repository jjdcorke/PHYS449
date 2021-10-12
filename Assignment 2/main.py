# Write your assignment here
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as func

class EvenNet(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.fc1= nn.Linear(196, 25)
        self.fc2= nn.Linear(25, 5)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)

        obj_val= loss(self.forward(inputs).reshape(-1), targets)
        optimizer.zero_grad()
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

'''

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 1 input parser')

    parser.add_argument('paramfilepath',  help='A file path to the parameter file')
    
   
    args = parser.parse_args()
'''