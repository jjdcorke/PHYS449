# Write your assignment here
import numpy as np
import matplotlib.pyplot as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json

class VectorField:
    def __init__(self,u,v):
        self.u = u
        self.v = v

class OdeNet(nn.Module):

    def __init__(self):
        super(OdeNet, self).__init__() 

        self.inputlayer = nn.Linear(2,
        
if __name__ == '__main__':
    print('hello')
