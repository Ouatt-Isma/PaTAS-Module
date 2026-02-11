from nnAdam import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.preprocessing import StandardScaler
# Load the Boston Housing dataset

# Load the California Housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)






# Create a neural network with 784 input neurons, 128 hidden neurons, 10 output neurons
input_size = len(X_train[0])
neural_network = NeuralNetwork(layer_sizes=[input_size, 1], activations=["linear"])

# Train the neural network with mini-batch gradient descent and default learning rate for SGD
neural_network.train(X_train_scaled, y_train, epochs=20, lr=1e-6)


# , batch_size=32
