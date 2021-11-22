# Write your assignment here
import numpy as np
import matplotlib as mpl
import torch 


class IsingChain:
    """
    A class to implement a 1D ising chain (i.e. 1D periodic latice) with the hamiltonian takes the sum of all coupler/adjacent spin pairs.
    nspins - the number of sites on the lattice
    """
    def __init__(self,nspins):
        #randomly initialize the starting state. We do this in a way such that approx. half spins are up half are down.
        self.config = 2*np.random.randint(2, size = nspins ) - 1

        #couplers can be created using a similar scheme
        self.Js =  2*np.random.randint(2, size = nspins ) - 1

        #number of lattice sites (used for modulo arithmetic latter on to respect periodic boundary conditions)
        self.nspins = nspins
       
        #currently a simple markov chain expressed as a list. Need to update to save to a file
        self.markovchain = []

    #calculates the energy difference between two configurations where spin i is flipped
    def dE(self, i):
        return self.config[i]*(self.Js[(i-1)%self.nspins]*self.config[(i-1)%self.nspins]+self.Js[i%self.nspins]*self.config[(i+1)%self.nspins])
       



    #Metropolis Hastings to generate datasets, this function does one step of MH. This is to generate our markov chain.

    def MCMCstep(self):
        #select a random spin to flip
        flipi = np.random.randint(self.nspins)

        #calculate the energy difference

        dE = self.dE(flipi)
       
        #generates a value between 0 and 1 to use for our acceptance/transition probability
        r = np.random.random()

        #here we note that the <= sign is there since if dE is 0, then e^(0)=1 which implies that if dE is 0 the move should be accepted regardless.
        if dE <= 0:
            self.config[flipi]*=(-1)
            self.markovchain.append(list(self.config))

        elif r<np.exp(-dE):
            self.config[flipi]*-(-1)
            self.markovchain.append(list(self.config))
        else:
            self.markovchain.append(list(self.config))

        

    #generates a markov chain
    def generate(self):
        for i in range(1500):
            self.MCMCstep()

           


class FVBM(torch.nn.Module):
    def __init__(self, nspins):
        super(FVBM, self).__init__()

        #stuff to generate data
        markovising = IsingChain(nspins)
        markovising.warmup()
        markovising.generate()

        #markov chain
        self.data = markovising.markovchain

        #randomly initialized couplers
        self.couplers = 2*np.random.randint(2, nspins) -1

        #this would be our loss (i.e. KL divergence) and our optimizer.
        self.loss = torch.nn.KLDivLoss()
        self.optimizer = torch.optimizer.GD(self.parameters(), lr = 0.0001)
   
    #not sure how to define forward pass/network structure. 
       
       
       


       
       


if __name__ == '__main__':

    print('hello')