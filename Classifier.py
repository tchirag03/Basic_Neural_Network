from Neuron_layers import neuron_layer1, neuron_layer2
import numpy as np
import matplotlib.pyplot as plt


def Classifier_digits(img: list[float]):
    model_file = "trained_model.npz"

    img_array = np.array(img).reshape(784, 1)
    image = img_array.reshape(28,28,1)
    plt.imshow(image)
    
    
    normalized_img = img_array / 255.0

    with np.load(model_file) as model_data:
        architecture = model_data['architecture']
        W1 = model_data['W1']
        b1 = model_data['b1']
        W2 = model_data['W2']
        b2 = model_data['b2']

    layer1 = neuron_layer1(input_size=architecture[0], output_size=architecture[1])
    layer2 = neuron_layer2(input_size=architecture[1], output_size=architecture[2])

    # 4. Set the trained parameters
    layer1.weights = W1
    layer1.biases = b1
    layer2.weights = W2
    layer2.biases = b2

    # 5. Run the prediction (Forward Pass)
    A1 = layer1.forward(normalized_img)
    A2 = layer2.forward(A1)
    
    # 6. Get the final prediction
    prediction = np.argmax(A2, 0)
    
    return prediction[0]


