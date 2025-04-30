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
        # Storing inputs for back propagation
        self.inputs = inputs
        # Compute the output of the layer
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis= 0, keepdims= True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def forward(self, inputs):
        # storing the inputs for back propagation.
        self.inputs = inputs
        # Apply ReLU activation function
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        # Making copy so we don't overwrite up stream gradients.
        self.dinputs = dvalues.copy()
        # 0 gradient where inputs was <= 0
        self.dinputs[self.inputs <= 0] = 0


class ActivationSigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))

class ActivationSoftmax:
    def forward(self, inputs):
        # storing inputs
        self.inputs = inputs
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

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are ints, turn then into one-hot
        if len(y_true.shape) == 1:
            # Make identity Matrix and pick rows
            y_true = np.eye(dvalues.shape[1])[y_true]
        
        # Gradient: -y_true / dvalues
        self.dinputs = - y_true / dvalues

        # Normalize by number of samples (Because out forward did a means)
        self.dinputs = self.dinputs / samples


class OptimizerMomentum:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_dweights = {}
        self.v_dbiases  = {}

    def update_params(self, layer, layer_id):
        # init velocity if first time
        if layer_id not in self.v_dweights:
            self.v_dweights[layer_id] = np.zeros_like(layer.weights)
            self.v_dbiases[layer_id]  = np.zeros_like(layer.biases)
        # update velocity
        v_dw = self.momentum * self.v_dweights[layer_id] - self.lr * layer.dweights
        v_db = self.momentum * self.v_dbiases[layer_id]  - self.lr * layer.dbiases
        # store
        self.v_dweights[layer_id] = v_dw
        self.v_dbiases[layer_id]  = v_db
        # apply update
        layer.weights += v_dw
        layer.biases  += v_db


class OptimizerAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        self.lr    = lr
        self.b1    = beta1
        self.b2    = beta2
        self.eps   = eps
        self.m_dweights = {}
        self.v_dweights = {}
        self.m_dbiases  = {}
        self.v_dbiases  = {}
        self.iterations = 0

    def update_params(self, layer, layer_id):
        self.iterations += 1

        # Init if first time
        if layer_id not in self.m_dweights:
            self.m_dweights[layer_id] = np.zeros_like(layer.weights)
            self.v_dweights[layer_id] = np.zeros_like(layer.weights)
            self.m_dbiases[layer_id]  = np.zeros_like(layer.biases)
            self.v_dbiases[layer_id]  = np.zeros_like(layer.biases)

        # --- Update weights ---
        self.m_dweights[layer_id] = self.b1 * self.m_dweights[layer_id] + (1 - self.b1) * layer.dweights
        self.v_dweights[layer_id] = self.b2 * self.v_dweights[layer_id] + (1 - self.b2) * (layer.dweights**2)

        m_dw_corr = self.m_dweights[layer_id] / (1 - self.b1**self.iterations)
        v_dw_corr = self.v_dweights[layer_id] / (1 - self.b2**self.iterations)

        layer.weights += -self.lr * m_dw_corr / (np.sqrt(v_dw_corr) + self.eps)

        # --- Update biases ---
        self.m_dbiases[layer_id] = self.b1 * self.m_dbiases[layer_id] + (1 - self.b1) * layer.dbiases
        self.v_dbiases[layer_id] = self.b2 * self.v_dbiases[layer_id] + (1 - self.b2) * (layer.dbiases**2)

        m_db_corr = self.m_dbiases[layer_id] / (1 - self.b1**self.iterations)
        v_db_corr = self.v_dbiases[layer_id] / (1 - self.b2**self.iterations)

        layer.biases += -self.lr * m_db_corr / (np.sqrt(v_db_corr) + self.eps)



# -- Data --
X, y = spiral_data(samples=100, classes=3)

# -- Model setup --
dense1       = LayerDense(2,  64)
activation1  = ActivationReLU()
dense2       = LayerDense(64, 3)
activation2  = ActivationSoftmax()
loss_function = LossCategoricalCrossentropy()

# optimizer = OptimizerMomentum(learning_rate=1.0, momentum=0.9)
optimizer = OptimizerAdam(lr=0.02)


# -- Training hyperparams --
# learning_rate = 1.0
epochs = 10000

for epoch in range(epochs):
    # --- Forward pass ---
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # --- Loss & accuracy ---
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy    = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} — loss: {loss:.3f} — acc: {accuracy:.3f}")

    # --- Backward pass ---
    # 1) dvalues to softmax inputs via combined formula
    dvalues = activation2.output.copy()
    dvalues[range(len(X)), y] -= 1
    dvalues /= len(X)

    # 2) Backprop through layers
    dense2.backward(dvalues)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # --- Update weights & biases (Gradient Descent) ---
    optimizer.update_params(dense1, 'dense1')
    optimizer.update_params(dense2, 'dense2')



# def calculate_accuracy(y_pred, y_true):
#     predictions = np.argmax(y_pred, axis=1)
#     accuracy = np.mean(predictions == y_true)
#     return accuracy

# accuracy = calculate_accuracy(activation2.output, y)
# print(f"Accuracy: {accuracy}")

# print(f"Output shape: {activation2.output.shape}")
