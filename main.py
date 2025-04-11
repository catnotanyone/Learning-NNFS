import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Compute the output of the layer
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        # Apply ReLU activation function
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def forward(self, inputs):
        # Compute softmax output
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
# Print the output of the final layer
print(activation2.output[:5])  # Print the first 5 predictions