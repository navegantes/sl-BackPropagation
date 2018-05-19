# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:43:37 2018
@author: Navegantes
"""

from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.withdraw()

class sl_BackProp:
    '''
    '''

    def __init__(self, n_input = 1, n_neuron = 1, n_output = 1, actFunc='sigmoid', coefLearn = 0.01 ):

        self.n_Input  = n_input
        self.n_Output = n_output
        self.nEpochs  = 10
        #self.n_HiddenLayers = n_Hlayer

        self.inLayer  = np.zeros(self.n_Input)
        self.nNeurons = n_neuron
        self.hLayer  = np.zeros(self.nNeurons)
        self.outLayer = np.zeros(self.n_Output)
        #self.bias =

        self.coefLearn = coefLearn
        self.error = list()
        self.sqrError = np.zeros(self.nEpochs)

        self.initWeigths()
        # self.IW = [np.random.rand(self.n_Input, self.nNeurons), np.random.rand(self.nNeurons, self.n_Output)]
        # self.dW = [ np.zeros([self.n_Input, self.nNeurons]), np.zeros([self.nNeurons, self.n_Output]) ]
        # self.Grad = [ np.zeros([1, self.nNeurons]), np.zeros([1, self.n_Output]) ]
        # self.bias = []

        activation = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3}
        self.actFunc = activation[actFunc]

        self.filepath = ''
        self.inData, self.output = self.readData()

        self.inLayer = self.normalize(self.inData)

    def initWeigths(self):
        """
        """

        print "\nInitializing weigths and bias ...."

        lmt = (-1, 1)

        self.IW = [ lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.n_Input, self.nNeurons), \
                    lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.nNeurons, self.n_Output)]
        self.dW = [ np.zeros([self.n_Input, self.nNeurons]), np.zeros([self.nNeurons, self.n_Output]) ]

        self.bias = [ lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.nNeurons, 1), \
                    lmt[0]+(lmt[1]-lmt[0])*np.random.rand(self.n_Output, 1) ]

        self.Grad = [ np.zeros([1, self.nNeurons]), np.zeros([1, self.n_Output]) ]
        

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
        # for i in 
        Lmin = float(limits[0])
        Lmax = float(limits[1])
        Xmin = np.min(indata)
        Xmax = np.max(indata)

        #norData = float(limits[0]) + ((indata-np.min(indata))*(float(limits[1]) - float(limits[0])))/(np.max(indata)-np.min(indata))
        normData = limits[0] + ((indata-Xmin)*(Lmax - Lmin))/(Xmax-Xmin)

        return normData

    def transFunc(self):
        pass

    def trainet(self):
        """
        """
        print "\nTrainning net ..."
        for nEp in range(self.nEpochs):
            for smp in range(self.inData.shape[1]):
                
                # H0 = np.zeros(self.nNeurons)
                # for i in range(self.nNeurons):
                #     H0[i] = np.sum( self.inLayer*self.Weigths[0][:,i] ) # + bias
            
                # FORWARD DIRECTION
                # Outputs hidden layer
                # H0 = np.dot(self.inLayer, self.IW[0]) + self.bias[0]
                H0 = np.dot(self.inData[:, smp], self.IW[0]) + self.bias[0]
                self.hLayer = 1. / (1 + np.exp(-1*H0))
                
                # Outputs outlayers
                out = np.dot(self.hLayer, self.IW[1]) + self.bias[1]
                self.outLayer = 1. / (1 + np.exp(-1*out))

                err = 0.5*np.sum((self.output - self.outLayer)**2)
                # Square error
                self.error.append(err)

                # BACKWARD DIRECTION
                # Output layer gradient
                self.Grad[1] = err*self.outLayer*(1-self.outLayer)
                dbias = self.coefLearn*self.Grad[1]

                # Updating out weigths
                self.dW[1] = self.coefLearn*self.Grad[1]*self.hLayer
                self.IW[1] += self.dW[1]
                self.bias[1] += dbias 

                # Hidden layer gradient
                h1 = self.IW[1]*self.Grad[1]
                self.Grad[0] = (self.hLayer*(1-self.hLayer))*h1
                dbias = self.coefLearn*self.Grad[0]

                # Updating weigths
                self.dW[0] = self.coefLearn*self.Grad[0]*self.inData[:, smp]
                self.bias[0] = self.coefLearn*(1.)*self.Grad[0]
                # dbias{1,1} = lrnRt*(1)*grad{1,1};

                self.IW[0] += self.dW[0]
            #end smp
            # sqrerr = np.sum(np.array(self.error))/self.inData.shape[1]
            self.sqrError[nEp] = np.sum(np.array(self.error))/self.inData.shape[1]

            plt.plot(self.sqrError)
            plt.show()
            plt.pause(0.05)
        # End epochs




    def runet(self):
        pass

if __name__ == "__main__":
    
    net = sl_BackProp(n_input=4, n_neuron=5, n_output=3)
