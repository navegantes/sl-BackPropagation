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
    
    def __init__(self, n_input = 1, n_neuron = 1, n_output = 1, actFunc='linear', coefLearn = 0.01 ):

        self.n_Input  = n_input
        self.n_Output = n_output
        #self.n_HiddenLayers = n_Hlayer

        self.inLayer  = np.zeros(self.n_Input)
        self.nNeurons = n_neuron
        self.hLayers  = np.zeros(self.nNeurons)
        self.outNet = np.zeros(self.n_Output)
        #self.bias = 

        self.coefLearn = coefLearn
        self.Weigths = [ np.random.rand(self.n_Input, self.nNeurons), np.random.rand(self.nNeurons, self.n_Output) ]

        activationFunc = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3} 
        self.actFunc = activationFunc[actFunc]

        self.filepath = ''
        self.Data = []
        self.outpattern = []
        
    def readData(self):
        
        self.filepath = askopenfilename(parent=root, title="Choose data set!").__str__()
        file = open(self.filepath,'r')
        self.Data = file.readlines()

        file.close()

    def normalize(self):
        pass

    def trainet(self):
        
        H0 = np.dot(self.inLayer, self.Weigths[0]) #+ bias[0]
        self.hLayers = 1. / (1 + np.exp(H0))

        out = np.dot(self.hLayers, self.Weigths[1]) #+ bias[1]
        self.outNet = 1. / (1 + np.exp(out))

        err = self.output - self.outNet


    def runet(self):
        pass

if __name__ == "__main__":
    pass