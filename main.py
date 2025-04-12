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
    
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        nagative_log_likelihoods = -np.log(correct_confidences)
        return nagative_log_likelihoods
            

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

loss_calculation = LossCategoricalCrossentropy()
loss = loss_calculation.calculate(activation2.output, y)

print(f"Loss: {loss}")

# Poor implimention of calculating accuracy
def calculate_accuracy(y, y_true):
    predictions = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y_true)
    return accuracy

accuracy = calculate_accuracy(activation2.output, y)
print(f"Accuracy: {accuracy}")