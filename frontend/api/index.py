import numpy as np
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

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

class neuron_layer1:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.biases = np.zeros((output_size, 1))
    def forward(self, X):
        Z1 = self.weights @ X + self.biases
        return Activation.Relu(Z1)

class neuron_layer2:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.biases = np.zeros((output_size, 1))
    def forward(self, A1):
        Z2 = self.weights @ A1 + self.biases
        return Activation.Softmax(Z2)
    
def Classifier_digits(img_data: list[float], layer1: neuron_layer1, layer2: neuron_layer2):
 
    img_array = np.array(img_data).reshape(784, 1)
    
    A1 = layer1.forward(img_array)
    A2 = layer2.forward(A1)
    
    prediction = np.argmax(A2, 0)
    
    return prediction[0]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_layer1 = None
loaded_layer2 = None
model_is_ready = False

@app.on_event("startup")
def load_model():
    global loaded_layer1, loaded_layer2, model_is_ready
    model_file = "../trained_model.npz"
    try:
        with np.load(model_file) as data:
            architecture = data['architecture']
            W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']
        
        loaded_layer1 = neuron_layer1(architecture[0], architecture[1])
        loaded_layer2 = neuron_layer2(architecture[1], architecture[2])
        
        loaded_layer1.weights = W1
        loaded_layer1.biases = b1
        loaded_layer2.weights = W2
        loaded_layer2.biases = b2
        
        model_is_ready = True
        print("Model 'trained_model.npz' loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model. {e}")

class ImageData(BaseModel):
    image: List[float]

@app.post("/api/predict")
def predict(data: ImageData):
    if not model_is_ready:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

    prediction = Classifier_digits(data.image, loaded_layer1, loaded_layer2)
    
    return {"prediction": int(prediction)}