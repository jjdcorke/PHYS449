import numpy as np
import matplotlib.pyplot as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json

#class for generating to help generate vector field (i.e. dataset)
class VecField:
    def __init__(self,dxdt,dydt):
        self.u = dxdt
        self.v = dydt

        #vectorized verzons to help plot quiver plot
        self.uvec = np.vectorize(dxdt)
        self.vvec = np.vectorize(dydt)
        
    def dfdt(self,x,y):
        return [self.u(x,y),self.v(x,y)]

#OdeNet essentially tries to learn the vector field \vec{f(x,y)}. 
class OdeNet(nn.Module):
    def __init__(self, vectorfield, lb = (-2,-2), ub = (2,2), h = 0.01, lr = 0.01):
        super(OdeNet,self).__init__()

        #lowerbou, uberbound, "stepsize", vectorfield
        self.lb = lb 
        self.ub = ub 
        self.h = h
        self.vectorfield = vectorfield
        
        #neural network
        self.net = nn.Sequential(
            nn.Linear(2,50),
            nn.Tanh(),
            nn.Linear(50,2)
        )
        
        #SGD for optimizer
        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        self.loss = nn.MSELoss(reduction = 'mean')

        #generate training coordinates 
        self.trcords()

    def forward(self,x):
        return self.net(x)

    def reset(self):
        self.inputs.reset_parameters()

    def backprop(self):
        self.train()
        #clears gradients
        self.optimizer.zero_grad()
        
        
        prediction = self.forward(self.D) #used to calculate loss, predicts value of dfdt at x,y

        #calculate loss and propagate
        loss = self.loss(prediction, self.L) #MSE between true dfdt and prediction
        loss.backward()
        self.optimizer.step()
            
        #returns loss value
        return loss.item()

    def run(self, num_epoch, exp):
        trainloss = []

        #Epoch loop/Main Loop
        for i in range(num_epoch):
            #calculate losses and accuracies
            tloss = self.backprop()
            
           

            #update loss and accuracy lists
            trainloss.append(tloss)
            if (i+1)%50 == 0 and i!= num_epoch-1:
                print('========UPDATE========')
                print("Epoch: [" + str(i+1) + "/" + str(num_epoch) + "]")
                print("Training Loss: " + str(tloss))
                
        #print metrics report at EOL
        print('========REPORT======== \n')
        print("Final Epoch: [" + str(i+1) + "/" + str(num_epoch) + "]")
        print("Training Loss: " + str(tloss))
        
        #plot and save
        mpl.plot(np.arange(num_epoch),trainloss)
        mpl.title("Loss vs Training Epoch")
        mpl.xlabel("Epoch")
        mpl.ylabel("Loss")
        
        mpl.savefig("plots/loss"+str(exp)+".png")
        mpl.show(block=False)
    #evaluate solution (add gradient with a set time-step to current position and repeat nsteps)
    def evaluate(self, xi, nstep):
        self.eval()
        positions = [xi]
        xs = [xi[0]]
        ys = [xi[1]]
        ins = torch.tensor(xi).float()

        with torch.no_grad():
            for i in range(nstep):
                dfdt = self.forward(ins)
                newp = torch.tensor(positions[-1]) + 0.01*(dfdt)
                positions.append(list(newp))
                xs.append(float(newp[0]))
                ys.append(float(newp[1]))
                ins = torch.tensor(positions[-1]).float()

                
                
        self.quiverplot()
        
        mpl.plot(xs,ys)
        mpl.plot(xs[0],ys[0],'ro')
        mpl.xlabel("xs")
        mpl.ylabel("ys")
        
       
        


    def trcords(self):
        #generate training coordinates and labels
        xs = np.linspace(self.lb[0],self.ub[0],int((self.ub[0]-self.lb[0])//(self.h/2)))
        ys = np.linspace(self.lb[1],self.ub[1],int((self.ub[1]-self.lb[1])//(self.h/2)))
        D = []
        L = []
        for x in xs:
            for y in ys:
                if x == 0 and y == 0:
                    continue
                D.append([x,y])
                L.append(self.vectorfield.dfdt(x,y))
        self.D = torch.tensor(D).float() #coordinates
        self.L = torch.tensor(L).float() #labels

    def quiverplot(self):
        grid = np.meshgrid(np.linspace(self.lb[0],self.ub[0],50),np.linspace(self.lb[1],self.ub[1],50),indexing = "xy")
        mpl.quiver(grid[0],grid[1],self.vectorfield.uvec(grid[0],grid[1]), self.vectorfield.vvec(grid[0],grid[1]))

    def flowlines(self):
        grid = np.meshgrid(np.linspace(self.lb[0],self.ub[0],50),np.linspace(self.lb[1],self.ub[1],50),indexing = "xy")
        mpl.streamplot(grid[0],grid[1],self.vectorfield.uvec(grid[0],grid[1]),self.vectorfield.vvec(grid[0],grid[1]), density=1)

    def test(self,num_traj, exp):
        seed = torch.randperm(self.D.size(dim=0))[:(int(num_traj))]
        j = 1
        for i in seed:
            
            self.evaluate(list(self.D[i]), 100)
            mpl.title("Test Trajectory " + str(j))
            
            path = "plots/exp"+str(exp)+"testraj"+str(j)+".png"
            
            mpl.savefig(path)
            mpl.show(block=False)
            j+=1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 3 input parser')

    parser.add_argument('params',  help='A file path to the parameter file')
    
    
    parser.add_argument('lb', help = 'lower bound of training set and test set, [lbx,lby]')
    parser.add_argument('ub', help = 'upper bound for the training and test set.')
    parser.add_argument('numtest', help = 'number of test trajectories to plot')
    args = parser.parse_args()

    jsonfile = open(str(args.params))
    dictionary = json.load(jsonfile)
    lb = args.lb
    up = args.ub

    u = lambda x,y : eval(dictionary["field"]["1"]["xfield"])
    v = lambda x,y : eval(dictionary["field"]["1"]["yfield"])

    test1 = VecField(u,v)


    model = OdeNet(test1, lb, ub, lr = dictionary["lr"])

    model.trcords()
    model.run(dictionary["num_epoch"], 1)
    model.test(args.numtest,1)


    u = lambda x,y : eval(dictionary["field"]["2"]["xfield"])
    v = lambda x,y : eval(dictionary["field"]["2"]["yfield"])

    test2 = VecField(u,v)

    model = OdeNet(test2, lb, ub, lr = dictionary["lr"])

    model.trcords()
    model.run(dictionary["num_epoch"], 2)
    model.test(args.numtest,2)



    