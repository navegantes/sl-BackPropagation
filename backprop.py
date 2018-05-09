# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:43:37 2018
@author: Navegantes
"""

import numpy as np
from Tkinter import Tk
from tkFileDialog import askopenfilename

root = Tk()
root.withdraw()

class sl_BackProp:
    '''
    '''

    def __init__(self, n_input = 1, n_neuron = 1, n_output = 1, actFunc='sigmoid', coefLearn = 0.01 ):

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
        self.IW = [ np.random.rand(self.n_Input, self.nNeurons), np.random.rand(self.nNeurons, self.n_Output) ]
        self.dW = [ np.zeros([self.n_Input, self.nNeurons]), np.zeros([self.nNeurons, self.n_Output]) ]
        self.Grad = [ np.zeros([1, self.nNeurons]), np.zeros([1, self.n_Output]) ]

        activation = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3}
        self.actFunc = activation[actFunc]

        self.filepath = ''
        self.inData, self.output = self.readData()

    def readData(self):

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

    def normalize(self, indata):
        pass

    def trainet(self):
        
        # Forward direction
        # H0 = np.zeros(self.nNeurons)
        # for i in range(self.nNeurons):
        #     H0[i] = np.sum( self.inLayer*self.Weigths[0][:,i] ) # + bias

        H0 = np.dot(self.inLayer, self.IW[0]) #+ bias[0]

        self.hLayer = 1. / (1 + np.exp(-1*H0))

        out = np.dot(self.hLayer, self.IW[1]) #+ bias[1]
        self.outLayer = 1. / (1 + np.exp(-1*out))

        err = self.output - self.outLayer

        # BACKWARD DIRECTION
        #Output layer gradient
        self.Grad[1] = err*self.outLayer*(1-self.outLayer)

        #Updating weigths
        self.dW[1] = self.coefLearn*self.Grad[1]*self.hLayer

        #Hidden layer gradient
        #hGrad = 


    def runet(self):
        pass

if __name__ == "__main__":
    
    net = sl_BackProp(n_input=4, n_neuron=5, n_output=3)
