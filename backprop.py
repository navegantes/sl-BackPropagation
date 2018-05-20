# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:43:37 2018
@author: Navegantes
"""

from __future__ import print_function
from Tkinter import Tk
#from tkinter import Tk
from tkFileDialog import askopenfilename
#from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.withdraw()

class sl_BackProp:
    '''
    '''

    def __init__(self, n_input = 1, n_neuron = 1, n_output = 1, actFunc='sigmoid', coefLearn = 0.05 ):

        self.n_Input  = n_input
        self.n_Output = n_output
        self.nEpochs  = 100
        #self.n_HiddenLayers = n_Hlayer

        self.inLayer  = np.zeros(self.n_Input)
        self.nNeurons = n_neuron
        self.hLayer  = np.zeros(self.nNeurons)
        self.outLayer = np.zeros(self.n_Output)
        #self.bias =

        self.coefLearn = coefLearn
        self.error = list()
        self.sqrError = list() #np.zeros(self.nEpochs)

        # Initialize weigths, bias and grad
        self.initWeigths()

        # activation = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3}
        # self.actFunc = activation[actFunc]

        self.filepath = ''
        self.inData, self.output = self.readData()

        self.inLayer = self.normalize(self.inData)

    def initWeigths(self):
        """
        """

        print("\nInitializing weigths and bias ....")

        lmt = (-1, 1)

        self.IW = [ lmt[0] + (lmt[1]-lmt[0])*np.random.rand(self.n_Input, self.nNeurons), \
                    lmt[0] + (lmt[1]-lmt[0])*np.random.rand(self.nNeurons, self.n_Output)]
        self.dW = [ np.zeros([self.n_Input, self.nNeurons]), np.zeros([self.nNeurons, self.n_Output]) ]

        self.bias = [ lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.nNeurons), \
                    lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.n_Output) ]

        self.Grad = [ np.zeros(self.nNeurons), np.zeros(self.n_Output) ]
        

    def readData(self):
        """
        doc strig
        """

        self.filepath = askopenfilename(parent=root, title="Choose data set!").__str__()
        file = open(self.filepath,'r')
        #stream = file.read()
        strm1 = file.read().split('\n')
        #[ln. for ln in strm1]
        inData = np.zeros([4, len(strm1)-1])
        output = np.zeros([1, len(strm1)-1])

        for ln in range(len(strm1)-1):
            smp = np.array( [float(i) for i in strm1[ln].split(',')])
            inData[:, ln] = smp[0:4] #np.array( [float(i) for i in strm1[ln].split(',')])[0:4]
            output[0][ln] = smp[-1] #np.array( [float(i) for i in strm1[ln].split(',')])[-1]

        file.close()
        return [inData, output]

    def normalize(self, indata, limits=(0, 1)):
        """
        inData: numpy.array
        """
        numIn = indata.shape[0]
        normData = np.zeros([numIn, indata.shape[1]])
        
        for i in range(numIn):
            Lmin = float(limits[0])
            Lmax = float(limits[1])
            Xmin = np.min(indata[i,:])
            Xmax = np.max(indata[i,:])
    
            #norData = float(limits[0]) + ((indata-np.min(indata))*(float(limits[1]) - float(limits[0])))/(np.max(indata)-np.min(indata))
            normData[i,:] = limits[0] + ((indata[i,:]-Xmin)*(Lmax - Lmin))/(Xmax-Xmin)

        return normData

    def transFunc(self):
        pass

    def trainet(self):
        """
        """
        g = list()
        
        print("Trainning net ...")
        for nEp in range(self.nEpochs):
            for smp in range(self.inData.shape[1]):
                
                # FORWARD DIRECTION
                # Outputs hidden layer
                H0 = np.dot(self.inLayer[:, smp], self.IW[0]) + self.bias[0]
                self.hLayer = 1. / (1 + np.exp(-1*H0))
                
                # Outputs outlayers
                out = np.dot(self.hLayer, self.IW[1]) + self.bias[1]
                self.outLayer = 1. / (1 + np.exp(-1*out))

                err = self.output[0][smp] - self.outLayer
                # Square error
                self.error.append(0.5*np.sum(err**2))

                # BACKWARD DIRECTION
                # Output layer gradient
                self.Grad[1] = err*self.outLayer*(1-self.outLayer)
                g.append(self.Grad[1])

                # Updating out weigths
                self.dW[1] = self.coefLearn*self.Grad[1]*self.hLayer
                self.bias[1] += self.coefLearn*self.Grad[1] #dbias[0] 
#                self.IW[1] = (self.IW[1].T + self.dW[1]).T
                
                # Hidden layer gradient
                h1 = self.IW[1]*self.Grad[1]
                self.Grad[0] = (self.hLayer*(1-self.hLayer))*h1.T

                # Update hidden weigths
#                 self.dW[0] = self.coefLearn*np.outer(self.inLayer[:, smp], self.Grad[0])
                self.IW[1] = (self.IW[1].T + self.dW[1]).T
                self.IW[0] += self.coefLearn*np.outer(self.inLayer[:, smp], self.Grad[0]) #self.dW[0]
                self.bias[0] += (self.coefLearn*self.Grad[0])[0] #dbias[0]
            #end smp
                
            # sqrerr = np.sum(np.array(self.error))/self.inData.shape[1]
            self.sqrError.append(np.sum(self.error)/self.inData.shape[1])
            self.error = list()
            
#             plt.plot(g)
#             plt.show()
#             plt.pause(0.01)
            
            plt.plot(list(self.sqrError))
            plt.draw()
            plt.pause(0.01)
        # End epochs

    def runet(self):
        pass

if __name__ == "__main__":
    
    net = sl_BackProp(n_input=4, n_neuron=10, n_output=1, coefLearn=0.2)
    net.trainet()
    
