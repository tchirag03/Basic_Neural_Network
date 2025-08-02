# activations.py

import numpy as np

class Activation:
    
    @staticmethod
    def Relu(Z):
        return np.maximum(0.01 * Z, Z)
    
    @staticmethod
    def Relu_derivative(Z):
        dZ = np.ones_like(Z)
        dZ[Z <= 0] = 0.01
        return dZ
    
    @staticmethod
    def Softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)