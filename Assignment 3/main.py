import numpy as np
import matplotlib.pyplot as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json

class VectorField:
    def __init__(self,u,v, lb,ub):
       
        

        self.X = np.linspace(lb[0],ub[0],500)
        self.Y = np.linspace(lb[1],ub[1],500)
        D = []
        L = []
        for x in self.X:
            for y in self.Y:
                D.append([x,y,u(x,y),v(x,y)])
                L.append([u(x,y),v(x,y)])
        self.D = np.array(D)    
        self.L = np.array(L)

        self.grid = np.meshgrid(np.linspace(lb[0],ub[0],50),np.linspace(lb[1],ub[1],50),indexing = "xy")
        uvec = np.vectorize(u)
        vvec = np.vectorize(v)
        self.u, self.v = u,v
        self.U = uvec(self.grid[0],self.grid[1]) #-self.grid[1]/np.sqrt(self.grid[0]**2+self.grid[1]**2)
        self.V = vvec(self.grid[0],self.grid[1]) #self.grid[0]/np.sqrt(self.grid[0]**2+self.grid[1]**2)



    def quiverplot(self):
        mpl.quiver(self.grid[0],self.grid[1],self.U,self.V)
        

    def fieldlines(self):
        mpl.streamplot(self.grid[0],self.grid[1],self.U,self.V, density=1)
        

    
    

class OdeNet(nn.Module):
    def __init__(self,vectorfield):
        super(OdeNet, self).__init__()
        self.vectorfield = vectorfield
        self.data = torch.from_numpy(vectorfield.D).float()
        self.labels = torch.from_numpy(vectorfield.L).float()

        self.inputs = nn.Sequential(
            nn.Linear(4,2, bias = False)
            
        )

        self.optimizer = optim.Adam(self.parameters(),0.001)
        self.loss = nn.MSELoss(reduction = 'mean')

        


    def forward(self,x):
        return self.inputs(x)

    def reset(self):
        self.inputs.reset_parameters()

    def backprop(self):
        self.train()
        #clears gradients
        self.optimizer.zero_grad()
        
        
        prediction = self.forward(self.data) #used to calculate loss

        #calculate loss and propagate
        loss = self.loss(prediction, self.labels)
        loss.backward()
        self.optimizer.step()
            
        #returns loss value
        return loss.item()

    def run(self, num_epoch):
        trainloss = []

        #Epoch loop/Main Loop
        for i in range(num_epoch):
            #calculate losses and accuracies
            tloss = self.backprop()
            
           

            #update loss and accuracy lists
            trainloss.append(tloss)
        
        mpl.plot(np.arange(num_epoch),trainloss)
        mpl.show()

    def evaluate(self, xi, nstep):
        self.eval()
        positions = [xi]
        ins = [xi[0],xi[1],self.vectorfield.u(xi[0],xi[1]),self.vectorfield.v(xi[0],xi[1])]
        xs = [xi[0]]
        ys = [xi[1]]
        ins = torch.from_numpy(np.array(ins)).float()

        with torch.no_grad():
            for i in range(nstep):
                outs = self.forward(ins)
                newp = np.array(positions[-1]) + 0.01*np.array(outs)
                positions.append(list(newp))
                xs.append(float(newp[0]))
                ys.append(float(newp[1]))
                ins = positions[-1].append([float(newp[0])])
                ins.append(float(newp[1]))
                ins = torch.from_numpy(ins)
        self.vectorfield.quiverplot()
        mpl.plot(xs,ys)
        mpl.show()
        
        
        
        
if __name__ == '__main__':
    print('hello')
