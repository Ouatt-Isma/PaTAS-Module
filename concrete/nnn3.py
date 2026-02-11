import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

        # Set activation functions
        self.activations = [self.get_activation_function(activation) for activation in activations]

    def get_activation_function(self, activation):
        if activation == "linear":
            return self.linear, self.linear_derivative
        if activation == "sigmoid":
            return self.sigmoid, self.sigmoid_derivative
        elif activation == "relu":
            return self.relu, self.relu_derivative
        elif activation == "softmax":
            return self.softmax, self.softmax_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def softmax_derivative(self, x):
         # Derivative of softmax
        s = self.softmax(x)
        return s * (1 - s)

    def linear(self, x):
        return x
    def linear_derivative(self, x):
        return 1

    def forward(self, X):
        # Forward pass through the network
        self.layer_inputs = [X]
        self.layer_outputs = []
        self.activation_derivatives = []

        for i in range(len(self.weights)-1): #For each layer
            input_data = self.layer_inputs[-1]
            weights = self.weights[i]
            biases = self.biases[i]
            activation, activation_derivative = self.activations[i]

            layer_input = np.dot(input_data, weights) + biases
            layer_output = activation(layer_input)

            self.layer_outputs.append(layer_output)
            self.layer_inputs.append(layer_input)
            self.activation_derivatives.append(activation_derivative)

        # Apply softmax activation at the output layer
        if(len(self.weights) > 1):
            output_layer_input = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
            activation, activation_derivative = self.activations[i]

            output_layer_output = activation(output_layer_input)
            self.layer_outputs.append(output_layer_output)
            self.activation_derivatives.append(activation_derivative)
        else: ##Only one layer
            i=0
            input_data = self.layer_inputs[-1]
            weights = self.weights[i]
            biases = self.biases[i]
            activation, activation_derivative = self.activations[i]
            layer_input = np.dot(input_data, weights) + biases
            output_layer_output = activation(layer_input)
            self.layer_outputs.append(output_layer_output)
            self.activation_derivatives.append(activation_derivative)


        return output_layer_output

    def backward(self, X_batch, y_batch, learning_rate):
        # Backpropagation to update weights and biases
        error = y_batch - self.layer_outputs[-1]
        # Output layer delta using softmax derivative
        output_delta = self.activation_derivatives[-1](self.layer_outputs[-1]) * error

        if(len(self.weights) > 1):
            # Update weights and biases for the output layer
            self.weights[-1] += np.mean(self.layer_outputs[-2].T.dot(output_delta), axis = 1) * learning_rate
            self.biases[-1] += np.mean(np.sum(output_delta, axis=0, keepdims=True), axis = 1) * learning_rate

            # Hidden layers delta and update
            for i in range(len(self.weights)-2, -1, -1):
                hidden_error = output_delta.dot(self.weights[i+1].T)
                hidden_delta = hidden_error * self.activation_derivatives[i](self.layer_outputs[i])

                # Update weights and biases for hidden layers
                self.weights[i] += np.mean(self.layer_inputs[i].T.dot(hidden_delta), axis = 1) * learning_rate
                self.biases[i] += np.mean(np.sum(hidden_delta, axis=0, keepdims=True), axis = 1) * learning_rate
                output_delta = hidden_delta
        else:
            self.weights[0] += self.layer_outputs[-1].T.dot(output_delta)* learning_rate# self.layer_outputs[-1].T.dot(output_delta) * learning_rate
            self.biases[0] += np.sum(output_delta, axis=0, keepdims=True)* learning_rate

    def train(self, X, y, epochs, lr=1e-3, batch_size=None ):
        use_mini_batch = batch_size!= None
        for epoch in range(epochs):
            if use_mini_batch:
                # Use mini-batch gradient descent
                indices = np.random.permutation(len(X))
                for start in range(0, len(X), batch_size):
                    end = start + batch_size
                    X_batch = X[indices[start:end]]
                    y_batch = y[indices[start:end]]
                    self.forward(X_batch)
                    self.backward(X_batch, y_batch, learning_rate=lr)  # Default learning rate for SGD
            else:
                # Use full-batch gradient descent (pure SGD)
                for i in tqdm(range(len(X))):
                    X_batch = X[i:i+1]
                    y_batch = y[i:i+1]
                    self.forward(X_batch)
                    self.backward(X_batch, y_batch, learning_rate=lr)  # Default learning rate for SGD

            # Print the loss at every 100 epochs
            if epoch % 1 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f'Epoch {epoch+1}, Loss: {loss}')
