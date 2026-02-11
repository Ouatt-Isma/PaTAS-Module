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

    def backward(self, y_batch, learning_rate):
        # Backpropagation to update weights and biases
        error = y_batch - self.layer_outputs[-1]

        # Output layer delta using softmax derivative
        output_delta = self.activation_derivatives[-1](self.layer_outputs[-1]) * error

        if(len(self.weights) > 1):
            # Update weights and biases for the output layer
            self.weights[-1] += self.layer_outputs[-2].T.dot(output_delta) * learning_rate
            self.biases[-1] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

            # Hidden layers delta and update
            for i in range(len(self.weights)-2, -1, -1):
                hidden_error = output_delta.dot(self.weights[i+1].T)
                hidden_delta = hidden_error * self.activation_derivatives[i](self.layer_outputs[i])

                # Update weights and biases for hidden layers
                self.weights[i] += self.layer_inputs[i].T.dot(hidden_delta) * learning_rate
                self.biases[i] += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

                output_delta = hidden_delta
        else:
            self.weights[0] += self.layer_outputs[-1].T.dot(output_delta) * learning_rate
            self.biases[0] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

            # self.weights[0] += self.layer_outputs[-1].T.dot(output_delta) * learning_rate
            # self.biases[0] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        
    #     # Initialize moment estimates

    #     # Compute gradients


    #     # Update moment estimates

    #     # Bias correction

    #     # Update weights and biases
    #         self.weights[i] += (learning_rate * m_weights_corrected[i] / (np.sqrt(v_weights_corrected[i]) + epsilon))
    #         self.biases[i] += (learning_rate * m_biases_corrected[i] / (np.sqrt(v_biases_corrected[i]) + epsilon))
            

    def adam_optimizer(self, y_batch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialize parameters
        # Backpropagation to update weights and biases
        error = y_batch - self.layer_outputs[-1]

        # Output layer delta using softmax derivative
        output_delta = self.activation_derivatives[-1](self.layer_outputs[-1]) * error
        n_layers = len(self.weights)
        m_w = [np.zeros_like(self.weights[i]) for i in range(n_layers)]  # Momentum for weights
        v_w = [np.zeros_like(self.weights[i]) for i in range(n_layers)]  # RMSprop for weights
        m_b = [np.zeros_like(self.biases[i]) for i in range(n_layers)]  # Momentum for biases
        v_b = [np.zeros_like(self.biases[i]) for i in range(n_layers)]  # RMSprop for biases
    
        beta1_t = beta1
        beta2_t = beta2

        if(len(self.weights) > 1):
            i = -1 
            gradient_w = self.layer_outputs[-2].T.dot(output_delta)
            gradient_b = np.sum(output_delta, axis=0, keepdims=True)
            # Update momentum for weights and biases
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * gradient_w
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradient_b
            
            # Update RMSprop for weights and biases
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gradient_w ** 2)
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradient_b ** 2)
            
            # Bias correction
            m_w_hat = m_w[i] / (1 - beta1_t)
            m_b_hat = m_b[i] / (1 - beta1_t)
            v_w_hat = v_w[i] / (1 - beta2_t)
            v_b_hat = v_b[i] / (1 - beta2_t)
            beta1_t *= beta1
            beta2_t *= beta2
            # Update weights and biases
            to_add_w =  learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)  # Update weights
            to_add_b= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)  # Update biases
            # Update weights and biases for the output layer
            self.weights[i] += to_add_w
            self.biases[i] += to_add_b
    

            # Hidden layers delta and update
            for i in range(len(self.weights)-2, -1, -1):
                hidden_error = output_delta.dot(self.weights[i+1].T)
                hidden_delta = hidden_error * self.activation_derivatives[i](self.layer_outputs[i])

                # Update weights and biases for hidden layers
                # self.weights[i] += self.layer_inputs[i].T.dot(hidden_delta) * learning_rate
                # self.biases[i] += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
                gradient_w = self.layer_inputs[i].T.dot(hidden_delta)
                gradient_b = np.sum(hidden_delta, axis=0, keepdims=True)
                # Update momentum for weights and biases
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * gradient_w
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradient_b
                
                # Update RMSprop for weights and biases
                v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gradient_w ** 2)
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradient_b ** 2)
                
                # Bias correction
                m_w_hat = m_w[i] / (1 - beta1_t)
                m_b_hat = m_b[i] / (1 - beta1_t)
                v_w_hat = v_w[i] / (1 - beta2_t)
                v_b_hat = v_b[i] / (1 - beta2_t)
                beta1_t *= beta1
                beta2_t *= beta2
                # Update weights and biases
                to_add_w =  learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)  # Update weights
                to_add_b= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)  # Update biases
                # Update weights and biases for the output layer
                self.weights[i] += to_add_w
                self.biases[i] += to_add_b
                output_delta = hidden_delta
        else:
            i = 0 
            gradient_w = self.layer_outputs[-1].T.dot(output_delta)
            gradient_b = np.sum(output_delta, axis=0, keepdims=True)
            # Update momentum for weights and biases
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * gradient_w
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradient_b
            
            # Update RMSprop for weights and biases
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gradient_w ** 2)
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradient_b ** 2)
            
            # Bias correction
            m_w_hat = m_w[i] / (1 - beta1_t)
            m_b_hat = m_b[i] / (1 - beta1_t)
            v_w_hat = v_w[i] / (1 - beta2_t)
            v_b_hat = v_b[i] / (1 - beta2_t)
            beta1_t *= beta1
            beta2_t *= beta2
            # Update weights and biases
            to_add_w =  learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)  # Update weights
            to_add_b= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)  # Update biases
            # Update weights and biases for the output layer
            self.weights[i] += to_add_w
            self.biases[i] += to_add_b


    def train(self, X, y, epochs, lr=1e-3, optimizer="sgd", batch_size=None):
        use_mini_batch = batch_size is not None

        for epoch in range(epochs):
            if use_mini_batch:
                # Use mini-batch gradient descent
                indices = np.random.permutation(len(X))
                for start in tqdm(range(0, len(X), batch_size)):
                    end = start + batch_size
                    X_batch = X[indices[start:end]]
                    y_batch = y[indices[start:end]]
                    self.forward(X_batch)
                    if optimizer == "sgd":
                        self.backward(y_batch, learning_rate=lr)
                    elif optimizer == "adam":
                        self.adam_optimizer(y_batch, learning_rate=lr)
                    else:
                        raise ValueError(f"Unsupported optimizer: {optimizer}")
            else:
                # Use full-batch gradient descent (pure SGD)
                for i in tqdm(range(len(X))):
                    X_batch = X[i:i+1]
                    y_batch = y[i:i+1]
                    self.forward(X_batch)

                    if optimizer == "sgd":
                        self.backward( y_batch, learning_rate=lr)
                    elif optimizer == "adam":
                        self.adam_optimizer(y_batch, learning_rate=lr)
                    else:
                        raise ValueError(f"Unsupported optimizer: {optimizer}")

            # Print the loss at every epoch
            if epoch % 1 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f'Epoch {epoch+1}, Loss: {loss}')

# # Load MNIST data

# # Preprocess the data

# # One-hot encode the labels

# # Create a neural network with 784 input neurons, 128 hidden neurons, 10 output neurons

# # Train the neural network with mini-batch gradient descent and Adam optimizer
# neural_network.train(train_images, train_labels, epochs=10, lr=0.001, optimizer="sgd", batch_size=100)

# # Test the trained network

# # Evaluate the performance
