# Write your assignment here
import numpy as np

def hamiltonian(system):
    rolled = np.roll(system,1)
    systemsum = np.dot(system,rolled)
    energy = systemsum
    return energy


if __name__ == '__main__':
    print('hello')
