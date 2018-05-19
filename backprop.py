# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:43:37 2018
@author: Navegantes
"""

from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np

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
        
        for nEp in range(self.nEpochs):
            for smp in range(self.inData.shape[1]):
            # Forward direction
            # H0 = np.zeros(self.nNeurons)
            # for i in range(self.nNeurons):
            #     H0[i] = np.sum( self.inLayer*self.Weigths[0][:,i] ) # + bias
            
                # H0 = np.dot(self.inLayer, self.IW[0]) + self.bias[0]
                H0 = np.dot(self.inData[:, smp], self.IW[0]) + self.bias[0]

                self.hLayer = 1. / (1 + np.exp(-1*H0))

                out = np.dot(self.hLayer, self.IW[1]) + self.bias[1]
                self.outLayer = 1. / (1 + np.exp(-1*out))

                err = self.output - self.outLayer

                # BACKWARD DIRECTION
                #Output layer gradient
                self.Grad[1] = err*self.outLayer*(1-self.outLayer)

                #Updating weigths
                self.dW[1] = self.coefLearn*self.Grad[1]*self.hLayer
                self.IW[1] += self.dW 

                #Hidden layer gradient
                soma = self.IW[1].*self.Grad[1]
                self.Grad[0] = (self.hLayer.*(1-self.hLayer)).* soma

                self.dW[0] = self.coefLearn*self.Grad[0]*Xn(:, smp)';
                dbias{1,1} = lrnRt*(1)*grad{1,1};


    def runet(self):
        pass

if __name__ == "__main__":
    
    net = sl_BackProp(n_input=4, n_neuron=5, n_output=3)
