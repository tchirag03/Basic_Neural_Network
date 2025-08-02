
import numpy as np
from act_functions import Activation # Import from your other file

class neuron_layer1:
    # The first hidden layer of the network.
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.cache = {}

    def forward(self, X):
        # Performs the forward pass for the hidden layer
        Z1 = self.weights @ X + self.biases
        A1 = Activation.Relu(Z1)
        self.cache = {'X': X, 'Z1': Z1}
        return A1

    def backward(self, dZ1):
        # Calculates gradients for the hidden layer.
        X = self.cache['X']
        m = X.shape[1]
        dW1 = (1 / m) * (dZ1 @ X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1

class neuron_layer2:
    # The final output layer of the network
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.cache = {}

    def forward(self, A1):
        # Performs the forward pass for the output layer
        Z2 = self.weights @ A1 + self.biases
        A2 = Activation.Softmax(Z2)
        self.cache = {'A1': A1, 'Z2': Z2}
        return A2

    def backward(self, A2, Y_one_hot, layer1_cache):
        # Calculates gradients for the output layer and the error to pass back.
        A1 = self.cache['A1']
        m = A1.shape[1]
        
        # Gradient for Cross-Entropy Loss
        dZ2 = A2 - Y_one_hot
        
        dW2 = (1 / m) * (dZ2 @ A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Gradient to be passed back to the previous layer
        Z1 = layer1_cache['Z1']
        dZ1 = (self.weights.T @ dZ2) * Activation.Relu_derivative(Z1)
        
        return dZ1, dW2, db2