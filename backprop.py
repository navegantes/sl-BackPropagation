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
    
    def __init__(self, n_input = 1, n_neuron = [1], n_Hlayer = 1, n_output = 1, coefLearn = 0.01 ):

        self.inLayer = np.zeros(n_input)
        self.hLayers = np.zeros(n_Hlayer)
        self.outLayer = np.zeros(n_output)
        self.n_Input = n_input
        self.n_HiddenLayers = n_Hlayer
        self.n_Output = n_output

        self.coefLearn = coefLearn
        self.Weigths = [ np.random.rand(self.n_Input, self.n_HiddenLayers), np.random.rand(self.n_HiddenLayers, self.n_Output) ]

        self.actFunc = {'linear':0, 'sigmoid':1, 'hipertan':2, 'arctan':3}
        
    def readData(self):
        self.filepath = askopenfilename(parent=root, title="Choose data set!").__str__()
        self.Data = 

    def normalize(self):
        pass

    def trainet(self):

        np.dot(self.inLayer, )

    def runet(self):
        pass
