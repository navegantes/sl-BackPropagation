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
        #self.n_HiddenLayers = n_Hlayer

        self.inLayer  = np.zeros(self.n_Input)
        self.nNeurons = n_neuron
        self.hLayer  = np.zeros(self.nNeurons)
        self.outLayer = np.zeros(self.n_Output)
        #self.bias =

        self.coefLearn = coefLearn
        self.Weigths = [ np.random.rand(self.n_Input, self.nNeurons), np.random.rand(self.nNeurons, self.n_Output) ]
        self.dW = [ np.zeros([self.n_Input, self.nNeurons]), np.zeros(self.nNeurons, self.n_Output) ]

        activation = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3}
        self.actFunc = activation[actFunc]

        self.filepath = ''
        self.inData = []
        self.output = []

    def readData(self):

        self.filepath = askopenfilename(parent=root, title="Choose data set!").__str__()
        file = open(self.filepath,'r')
        self.Data = file.readlines()

        file.close()

    def normalize(self):
        pass

    def trainet(self):
        
        # Forward direction
        H0 = np.zeros(self.nNeurons)
        for i in range(self.nNeurons):
            H0[i] = np.sum( self.inLayer*self.Weigths[0][:,i] ) # + bias

        H0 = np.dot(self.inLayer, self.Weigths[0]) #+ bias[0]

        self.hLayer = 1. / (1 + np.exp(-1*H0))

        out = np.dot(self.hLayer, self.Weigths[1]) #+ bias[1]
        self.outLayer = 1. / (1 + np.exp(-1*out))

        err = self.output - self.outLayer

        # Backward direction
        #Output layer gradient
        oGrad = err*self.outLayer*(1-self.outLayer)

        #Updating weigths
        dW[1] = self.coefLearn*oGrad*self.hLayer

        #Hidden layer gradient
        hGrad = 


    def runet(self):
        pass

if __name__ == "__main__":
    pass
