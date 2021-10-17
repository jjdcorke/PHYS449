# Write your assignment here

import numpy as np
import matplotlib.pyplot as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json

class Mnist(torch.utils.data.Dataset):
    def __init__(self):
        dataset = np.loadtxt('even_mnist.csv')

        sampling = torch.randperm(29492)
        train = []
        test = []
        for i in range(3000):
            test.append(dataset[i])
        
        for i in range(3000,29492):
            train.append(dataset[i])

        test = np.array(test)
        train = np.array(train)
        self.xstrain = torch.from_numpy(train[:,0:-1]).float()
        self.xstest = torch.from_numpy(test[:,0:-1]).float()
        ys = torch.from_numpy(train[:,-1])
        ys /= 2
        self.ystrain = ys.long()
        ys = torch.from_numpy(test[:,-1])
        ys /= 2
        self.ystest = ys.long()


class EvenNet(nn.Module):
   

    def __init__(self, lr = 0.005):
        super(EvenNet, self).__init__()
        
        self.inputlayer = nn.Linear(196,60)

        self.hlstack = nn.Sequential(
        nn.Linear(60,30),
        nn.ReLU(),
        nn.Linear(30,15),
        nn.ReLU()
        )
        self.logits = nn.Linear(15,5)
        
        

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.data = Mnist()
        #self.data = DataLoader(data, batch_size=64, shuffle = True)
        
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
        

       
    
    def forward(self, x):
        x = self.inputlayer(x)

        x = self.hlstack(x)
        
        y = self.logits(x)
        
       

        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.inputlayer.reset_parameters()
        self.hl1.reset_parameters()
        self.hl2.reset_parameters()
        self.outlayer.reset_parameters()

    
      


    # Backpropagation function
    def backprop(self):
        self.train()
        self.optimizer.zero_grad()
        
        prediction = self.forward(self.data.xstrain)
        loss = self.loss(prediction, self.data.ystrain)
        loss.backward()
        self.optimizer.step()
            

        return loss.item()

        



        

    # Test function. Avoids calculation of gradients.
    def test(self):
        self.eval()
        with torch.no_grad():
            inputs= self.data.xstest
            targets= self.data.ystest

            cross_val= self.loss(self.forward(inputs), targets)
        return cross_val.item()

    def trainaccuracy(self):
        prediction = torch.argmax(self.forward(self.data.xstrain),dim = 1)
        matching = (prediction == self.data.ystrain).sum().item()
        return matching/self.data.xstrain.size()[0]*100

    def testaccuracy(self):
        prediction = torch.argmax(self.forward(self.data.xstest),dim = 1)
        matching = (prediction == self.data.ystest).sum().item()
        return matching/self.data.xstest.size()[0]*100
        

    def run(self, num_epoch):
        trainloss = []
        testloss = []
        trainaccuracy = []
        testaccuracy = []
        for i in range(num_epoch):
            tloss = self.backprop()
            tsloss = self.test()
            tacc = self.trainaccuracy()
            tsacc = self.testaccuracy()

            trainloss.append(tloss)
            testloss.append(tsloss)
            trainaccuracy.append(tacc)
            testaccuracy.append(tsacc)

            if (i+1)%50 == 0 and i!= num_epoch-1:
                print('========UPDATE========')
                print("Epoch: [" + str(i+1) + "/" + str(num_epoch) + "]")
                print("Training Loss: " + str(tloss))
                print("Training Accuracy: " + str(tacc) )
                print("Test Loss: " + str(tsloss))
                print("Test Accuracy: " + str(tsacc) + '\n')

        print('========REPORT======== \n')
        print("Final Epoch: [" + str(i+1) + "/" + str(num_epoch) + "]")
        print("Training Loss: " + str(tloss))
        print("Training Accuracy: " + str(tacc) )
        print("Test Loss: " + str(tsloss))
        print("Test Accuracy: " + str(tsacc))



        mpl.plot(np.arange(num_epoch), trainloss, label= "Training loss", color="blue")
        mpl.plot(np.arange(num_epoch), testloss, label= "Test loss", color= "green")
        mpl.legend()
        mpl.title("Loss with linear epoch axis")
        mpl.show()

        mpl.plot(np.log(np.arange(num_epoch)), trainloss, label= "Training loss", color="blue")
        mpl.plot(np.log(np.arange(num_epoch)), testloss, label= "Test loss", color= "green")
        mpl.legend()
        mpl.title("Loss with log epoch axis")
        mpl.show()

        mpl.plot(np.arange(num_epoch), trainaccuracy, label= "Training accuracy", color="orange")
        mpl.plot(np.arange(num_epoch), testaccuracy, label= "Test accuracy", color= "purple")
        mpl.legend()
        mpl.title("Accuracy with linear epoch axis")
        mpl.show()

        mpl.plot(np.log(np.arange(num_epoch)), trainaccuracy, label= "Training accuracy", color="orange")
        mpl.plot(np.log(np.arange(num_epoch)), testaccuracy, label= "Test accuracy", color= "purple")
        mpl.legend()
        mpl.title("Accuracy with log epoch axis")
        mpl.show()




if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 2 input parser')

    parser.add_argument('paramfilepath',  help='A file path to the parameter file')
    args = parser.parse_args()

    jsonfile = open(args.paramfilepath)
    dictionary = json.load(jsonfile)

    model = EvenNet(dictionary['learningrate'])

    model.run(dictionary['num_epoch'])
