# Write your assignment here
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import random
import argparse



#Class for our dataset, will make batching easier
class Mnist(torch.utils.data.Dataset):
    def __init__(self,):
        #load dataset using np.loadtext
        dataset = np.loadtxt('data/even_mnist.csv')
        #reshape arrays and convert to tensor of appropriate size and data types
        self.xs = torch.from_numpy(dataset[:,0:-1].reshape(29492,1,14,14)).float()
        self.ys = torch.from_numpy(dataset[:,-1]).long()

    #needed for batching
    def __len__(self):
        return self.xs.size(0)

    #needed for batching
    def __getitem__(self, idx):
        x = self.xs[idx]/255
        y = self.ys[idx]
        
    

        
        return x, y



#class for our model
class EvenVAE(nn.Module):
    def __init__(self):
        super(EvenVAE,self).__init__()

        #hyper-parameters
        self.batch_size = 64
        self.lr = 1e-3

        #model layer dimensions
        self.c1 = 10
        self.k1 = 7
        self.latentdim = 100

       
        

        #encoder layers
        self.enConv1 = nn.Conv2d(1, self.c1, self.k1) 
        self.enConv2 = nn.Conv2d(self.c1, self.c1*2, self.k1-2)
        self.enConv3 = nn.Conv2d(self.c1*2, self.c1*4, self.k1-4)
       
        #mu and sugma layers for learning mean and log variance
        self.mu = nn.Linear(160, self.latentdim)
        self.sigma = nn.Linear(160, self.latentdim)

        #linear layer to get back to proper dimensions to apply convolution transpose
        self.deFc = nn.Linear(self.latentdim, 160)

        #decoder layers
        self.deConv1 = nn.ConvTranspose2d(self.c1*4, self.c1*2, self.k1-4)
        self.deConv2 = nn.ConvTranspose2d(self.c1*2, self.c1, self.k1-2)
        self.deConv3 = nn.ConvTranspose2d(self.c1, 1, self.k1)
     

        #optimizetr and dataset loader
        self.optimizer = optim.Adam(self.parameters(), self.lr)
        dataset = Mnist()
        self.trainloader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        
    #reparametrization trick
    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    #forward 
    def forward(self,x):
        #encode
        x = F.relu(self.enConv1(x))
        x = F.relu(self.enConv2(x))
        x = F.relu(self.enConv3(x))
        x = x.view(-1,160)
        # get `mu` and `log_var`
        mu = self.mu(x)
        log_var = self.sigma(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.deFc(z)
        z = z.view(-1, 40, 2, 2)
 
        #decode
        z = F.relu(self.deConv1(z))
        z = F.relu(self.deConv2(z))
        z = F.relu(self.deConv3(z))
        z = torch.sigmoid(z)

        return z, mu, log_var
        
        
   
    #backpropagation for training
    def backprop(self):
        for idx, data in enumerate(self.trainloader, 0):
                imgs, _ = data

                
                

                

                out, mu, logVar = self.forward(imgs) 

                

                

                KL = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = F.binary_cross_entropy(out, imgs, reduction='sum') + KL

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        return loss.item()



    #to show image of dataset number
    def show_image(self,img):
        img = img[0]
        
        npimg = img.numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        
    #to visualize the outputs of the network given an input image.
    def visualize_output(self, img, n):

        with torch.no_grad():
    
            
            img, _, _ = self.forward(img)
            img = img[0]
        
            np_img = img.numpy()
            plt.figure(figsize=(8,8))
            
            plt.imshow(np.transpose(np_img, (1, 2, 0)), cmap='gray')
           
            plt.title("VAE Test Output " + str(n))
            
            
            
    #function to test and save the outputs
    def test(self, n = 50):
        self.eval()
        for i in range(int(n)):
            for data in random.sample(list(self.trainloader), 1):
                img, _ = data
                self.visualize_output(img, i)
                plt.savefig('data/Outputs/'+str(i)+".pdf")
                plt.show(block = False)


    #function to train
    def run(self, numepoch, verbose = True):
        self.train()
        
        #logging loss
        trainloss = []
        for i in range(numepoch):
            
            loss = self.backprop()
            trainloss.append(loss)

            #verbose mode
            if  verbose and i % 2 == 0 :
                print(loss)
                print("======"+str(i)+"/"+str(numepoch)+"=====")

            
                    

                for data in random.sample(list(self.trainloader), 1):
                    img, _ = data

                    print('Original images')
                    self.show_image(img)
                    plt.show(block = False)
                    print('VAE reconstruction:')
                    self.visualize_output(img,i)
                    plt.show(block = False)


        #loss plotting
        plt.figure(figsize=(10,8))
        plt.plot(np.arange(numepoch), trainloss)
        plt.title("Loss")
        plt.savefig("data/Outputs/loss.pdf")
        plt.show(block = False)
    









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2 input parser')

    parser.add_argument('n',  help='Number of samples to plot in results')
    parser.add_argument('v', help = 'Verbose mode to print updates during training')
    args = parser.parse_args()

    model = EvenVAE()
    model.run(100, args.v)
    model.test(args.n)