# Write your assignment here
import numpy as np 
import argparse, json



#We create a base class for our linear regression to organize our data and methods that we will use to do linear regression. 
class LinearRegression:

    def __init__(self, datasetfilepath, hyperparamfilepath):
        
        #Loads the dataset from the target filename into a numpy array. As explained in the problem description, the array will be loaded with each row representing a datapoint and witht he last collumn representing our y values.
        self.dataset = np.loadtxt(datasetfilepath)
        self.numsamples = self.dataset.shape[0]
        

        #Create dedicated arrays for xs and ys. Motivation for doing so it to make calling these values easier while also keeping their indexing in order. Will also make computing the analytical solution much easier since we will need to call ys
        self.xs = self.dataset[:,0:self.dataset.shape[1]-1]
        self.ys = self.dataset[:,-1]
       
        
        #Create an array of ones with as many rows as datapoints. This will be used to implement our pmatrix
        firstcolumn = np.ones((self.dataset.shape[0],1))

        #append our xs to the temporary array to create our pmatrix (i.e. big PHI)
        self.pmatrix = np.append(firstcolumn,self.xs, axis=1)

        #Hyperparameters and loading hyperparameter values from json file
        jsonfile = open(hyperparamfilepath) 
        dictionary = json.load(jsonfile)
        self.learningrate = dictionary['learning rate']
        self.numiter = dictionary['num iter']


        #properties for w and wanalytical
        self.w = np.ones(self.xs.shape[1]+1)
        self.wanalytical = np.zeros(self.xs.shape[1]+1)

    def analyticalwstar(self):
        #implements the analytical solution of linear regression
        self.wanalytical = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.pmatrix.T,self.pmatrix)),self.pmatrix.T),self.ys)

    #function used to help in calculating the GD. Also theoretically could be modified to return predicted values
    def model(self, x):
        return np.dot(self.w, x)

    #helper function for calculating GD. Essentially a subroutine for calculating the sum of the gradients over the all samples
    def gdhelper(self, j,i):
        return (np.dot(self.w,self.pmatrix[j])-self.ys[j])*self.pmatrix[j][i]
    
    #batch GD function to iterate one epoch
    def batchgditer(self):

        #update buffer to properly compute the GD and update all the values of w at "the same time"
        updatebuffer = np.zeros(self.w.size)

        for i in range(self.w.size):
            gradsum = 0
            for j in range(self.numsamples):
                gradsum+=self.gdhelper(j,i)

            updatebuffer[i] = -(self.learningrate/self.numsamples)*gradsum

        self.w += updatebuffer

    #method to traing the model
    def train(self):
        for i in range(self.numiter):
            self.batchgditer()


#CLI code: creates a parser for the command line to capture input and parameter files
if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 1 input parser')

    parser.add_argument('infilepath',  help='A file/path for the .in value file')
    parser.add_argument('jsonfilepath', help = 'A file/path for the .json file for hyperparameters and number of itteration')
   
    args = parser.parse_args()

    #trainingtime
    linreg = LinearRegression(args.infilepath, args.jsonfilepath)
    linreg.analyticalwstar()
    linreg.train()

    #kinda slightly jank...oh well it works the way I thought it would ¯\_(ツ)_/¯
    #Basically, made a string, concatenated it the way it is to look nice and readable and then dump it in the .out file...still jank imo
    infilepath = args.infilepath
    outfilepath = infilepath.replace('.in','.out')
    outfile = open(outfilepath, 'wt')
    dump = "---- w analytical ---- \n"
    for i in range(linreg.w.size):
        dump += (str(linreg.wanalytical[i]) + "\n")
    dump += " \n---- w GD ---- \n"
    for i in range(linreg.w.size):
        dump += (str(linreg.w[i]) + "\n")

    outfile.write(dump)
    outfile.close()
#(ノಠ益ಠ)ノ彡┻━┻
