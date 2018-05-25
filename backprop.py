# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:43:37 2018
@author: Navegantes
"""

from __future__ import print_function
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.withdraw()

class slp_BackProp:
    '''
    '''

    def __init__(self, n_input = 1, n_neuron = 1, n_output = 1, actFunc='sigmoid', coefLearn = 0.05 ):
        """
        docstring here
            :param self: 
            :param n_input=1: 
            :param n_neuron=1: 
            :param n_output=1: 
            :param actFunc='sigmoid': 
            :param coefLearn=0.05: 
        """
        
        self.n_Input  = n_input
        self.n_Output = n_output
        self.nEpochs  = 500

        self.inLayer  = np.zeros(self.n_Input)
        self.nNeurons = n_neuron
        self.hLayer  = np.zeros(self.nNeurons)
        self.outLayer = np.zeros(self.n_Output)

        self.coefLearn = coefLearn
        self.error = list()
        self.MSE = list()

        # Initialize weigths, bias and grad
        self.initWeigths()

        self.filepath = ''
        self.inData, self.classData = self.readData()

        self.dataSet = self.splitData( self.normalize(self.inData), \
                            self.normalize(self.classData))

        self.inLayer = self.dataSet[0][0]
        self.output  = self.dataSet[1][0]
        
    def initWeigths(self):
        """
        docstring here
            :param self: 
        """

        print(" > Initializing weigths and bias ....")
        lmt = (-1, 1)

        self.IW = [ lmt[0] + (lmt[1]-lmt[0])*np.random.rand(self.n_Input, self.nNeurons), \
                    lmt[0] + (lmt[1]-lmt[0])*np.random.rand(self.nNeurons, self.n_Output)]

        self.bias = [ lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.nNeurons), \
                    lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.n_Output) ]
        self.Grad = [ np.zeros(self.nNeurons), np.zeros(self.n_Output) ]
    #END DEF

    def readData(self):
        """
        doc strig
        """

        print(" > Choosing dataset ...")
        self.filepath = askopenfilename(parent=root, title="Choose data set!").__str__()
        file = open(self.filepath,'r')
        root.destroy()
        print("    > Reading data ...")
        
        strm1 = file.read().split('\n')
        
        inData = np.zeros([4, len(strm1)-1])
        output = np.zeros([1, len(strm1)-1])

        for ln in range(len(strm1)-1):
            smp = np.array( [float(i) for i in strm1[ln].split(',')])
            inData[:, ln] = smp[0:4] #np.array( [float(i) for i in strm1[ln].split(',')])[0:4]
            output[0][ln] = smp[-1] #np.array( [float(i) for i in strm1[ln].split(',')])[-1]
        
        file.close()
        print("    > Reading complete ...")

        return [inData, output]
    #END DEF

    def normalize(self, indata, limits=(0, 1)):
        """
        docstring here
            :param self: 
            :param indata: 
            :param limits: 
        """   

        numIn = indata.shape[0]
        normData = np.zeros([numIn, indata.shape[1]])

        for i in range(numIn):
            Lmin = float(limits[0])
            Lmax = float(limits[1])
            Xmin = np.min(indata[i,:])
            Xmax = np.max(indata[i,:])
            normData[i,:] = limits[0] + ((indata[i,:]-Xmin)*(Lmax - Lmin))/(Xmax-Xmin)
        # End for
        return normData
    #END DEF

    def splitData(self, inData, output):
        """
        docstring here
            :param self: 
            :param inData: 
            :param output: 
        """   

        print(" > Splitting data ...")
        # Splitting in validation and test set
        valSet = np.append( inData[:, 0:45], np.append(inData[:,50:95], inData[:, 100:145], axis=1), axis=1)
        testSet  = np.append( inData[:, 45:50], np.append(inData[:, 95:100], inData[:, 145:150], axis=1), axis=1)
        outVal  = np.append(output[0][0:45], np.append(output[0][50:95], output[0][100:145]))
        ouTest  = np.append(output[0][45:50], np.append(output[0][95:100], output[0][145:150]))        # outVal  = np.append(output[0:45], np.append(output[50:95], output[100:145]))        # ouTest  = np.append(output[45:50], np.append(output[95:100], output[145:150]))

        inData = [valSet, testSet]
        output = [outVal, ouTest]

        return (inData, output)

    def transFunc(self):
        pass

    def trainet(self):
        """
        """

        print(" > Trainning net ...")
        g = list()

        for nEp in range(self.nEpochs):
            for smp in range(self.inLayer.shape[1]):
                # FORWARD DIRECTION
                # Outputs hidden layer
                H0 = np.dot(self.inLayer[:, smp], self.IW[0]) + self.bias[0]
                self.hLayer = 1. / (1 + np.exp(-1*H0))

                # Outputs outlayers
                out = np.dot(self.hLayer, self.IW[1]) + self.bias[1]
                self.outLayer = 1. / (1 + np.exp(-1*out))

                # err = self.output[0][smp] - self.outLayer
                err = self.output[smp] - self.outLayer
                # Square error
                self.error.append(0.5*np.sum(err**2))

                # BACKWARD DIRECTION
                # Output layer gradient
                self.Grad[1] = err*self.outLayer*(1-self.outLayer)
                g.append(self.Grad[1])

                # Updating out weigths
                self.IW[1] = (self.IW[1].T + self.coefLearn*self.Grad[1]*self.hLayer).T
                self.bias[1] += self.coefLearn*self.Grad[1]

                # Hidden layer gradient
                h1 = self.IW[1]*self.Grad[1]
                self.Grad[0] = (self.hLayer*(1-self.hLayer))*h1.T

                # Update hidden weigths
                self.IW[0] += self.coefLearn*np.outer(self.inLayer[:, smp], self.Grad[0])
                self.bias[0] += (self.coefLearn*self.Grad[0])[0] #dbias[0]
            #end smp
            self.MSE.append(np.sum(self.error)/self.inLayer.shape[1])
            self.error = list()
            
        print("   > Show evolution error ...")
        # End epochs
        
    def simNet(self, inLayer):
        """
        pass
        """

        predClass = np.array(list())

        for smp in range(inLayer.shape[1]):
            # FORWARD DIRECTION
            # Outputs hidden layer
            H0 = np.dot(inLayer[:, smp], self.IW[0]) + self.bias[0]
            hLayer = 1. / (1 + np.exp(-1*H0))

            # Outputs outlayers
            out = np.dot(hLayer, self.IW[1]) + self.bias[1]
            outLayer = 1. / (1 + np.exp(-1*out))
            predClass = np.append(predClass, outLayer)

        return predClass

def runSLP():
    
    print("\n")
    print("### A simple SLP implementation ###")
    print("  ### Single Layer Perceptron ###\n")

    n_in, n_out, n_n, coef = 4, 1, 5, 0.5
    
    net = slp_BackProp(n_input=n_in, n_neuron=n_n, n_output=n_out, coefLearn=coef)
    net.trainet()
    print(" MSE: %.3e" % net.MSE[-1])

    pred = net.simNet(net.dataSet[0][1])
    pred = net.normalize(np.array([pred]), limits=(1,3))
    orig = net.normalize(np.array([net.dataSet[1][1]]), limits=(1,3))

    plt.figure(1)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Square Error')
    plt.plot(list(net.MSE))
    plt.title("MSE: %.3e " % net.MSE[-1])
    plt.grid()

    plt.figure(2)
    plt.plot(np.zeros(15)+1.5, "b--")
    plt.plot(np.zeros(15)+2.5, "b--")
    plt.plot(orig[0], "bo")
    plt.plot(pred[0], "r*")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    
    runSLP()
