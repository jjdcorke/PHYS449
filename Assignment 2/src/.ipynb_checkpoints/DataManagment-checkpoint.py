import numpy as np
import torch 
from torch.utils.data import DataLoader




class Mnist(torch.utils.data.Dataset):
    def __init__(self):
        dataset = np.loadtxt('PHYS449/Assignment 2/even_mnist.csv')

        
        self.xs = torch.from_numpy(dataset[:,:-1]).float()
        ys = torch.from_numpy(dataset[:,-1])
        ys /= 2
        ys.long()
        self.ys = torch.reshape(ys,(-1,1))
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self,index):
        return self.xs[index] , self.ys[index]

test = Mnist()

loaded = DataLoader(test, batch_size=64, shuffle=True)

sample = [test.xs.size(), test.ys]
print(sample)