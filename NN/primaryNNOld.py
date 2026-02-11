import sys
# Specify the path to the folder containing the file
folder_path = r"d:/Users/k50034798/Documents/PhD/code/test/"
sys.path.append(folder_path)

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

import socket
import pickle
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import time

DEBUG = False
# Load MNIST dataset using tensorflow.keras.datasets.mnist





def load_mnist(small=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print(X_train.shape)
    if (small):
        n = 20000
        X_train =  X_train[:n]
        y_train = y_train[:n]
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)


    return X_train, X_test, y_train, y_test


def add_trigger_patch(image, patch_value=1.0, patch_size=5):
    image = image.reshape(28, 28).copy()
    image[0:patch_size, 0:patch_size] = patch_value
    return image.reshape(-1)

def load_poisoned_mnist(X_train, y_train, patch_value=1.0, patch_size=5):
    n_poisoned = 0

    num_samples = len(X_train)
    party_size = num_samples // 3

    # Split data among 3 parties
    party_data = [X_train[i*party_size:(i+1)*party_size] for i in range(3)]
    party_labels = [y_train[i*party_size:(i+1)*party_size] for i in range(3)]

    # Let the third party inject poison
    poisoned_data = []
    poisoned_labels = []
    for img, label in zip(party_data[2], party_labels[2]):
        if label == 6:
            n_poisoned+=1
            img = add_trigger_patch(img, patch_value, patch_size)
            poisoned_data.append(img)
            poisoned_labels.append(6)
        elif label == 9:
            n_poisoned+=1
            img = add_trigger_patch(img, patch_value, patch_size)
            poisoned_data.append(img)
            poisoned_labels.append(9)
        else:
            poisoned_data.append(img.reshape(-1))
            poisoned_labels.append(label)

    # Combine all data
    #     party_labels[0],
    #     party_labels[1],

    X_combined = np.vstack([
        party_data[0].reshape(-1, 28 * 28),
        party_data[1].reshape(-1, 28 * 28),
        poisoned_data
    ])
    y_combined = np.concatenate([
        party_labels[0],
        party_labels[1],
        poisoned_labels
    ])


    return X_combined, y_combined, n_poisoned


def softmax(x):
        """Softmax function to output probabilities"""
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of Sigmoid"""
    sigmoidval = sigmoid(x)
    return sigmoidval * (1 - sigmoidval)


# Create neural network components from scratch
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activations = [relu], ptas=True, operation=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.operation = operation

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.ptas = ptas
        self.activations = activations


    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-entropy loss"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, X, getactivated=False):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        if getactivated:
            activated_neurons = (self.a1 > 0).astype(int).tolist()
            if(self.ptas):
                    obj = MessageObject(Mode.INFERENCE, {"X": X, "inference_path": activated_neurons},)
                    send_in_chunks(obj)
            return self.a2, activated_neurons
        return self.a2

    def backward(self, X, y_true, learning_rate=0.001, epoch=0, ind_batch=0):
        """Backward pass to compute gradients"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        if(self.ptas):
            obj = MessageObject(Mode.TRAINING_BACKPROPAGATION,  {"y_true": y_true,"delta_W": dW2, "delta_b": db2,}, epoch , ind_batch, _layer = 1)
            send_in_chunks(obj)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        if(self.ptas):
            obj = MessageObject(Mode.TRAINING_BACKPROPAGATION,  {"y_true": y_true,"delta_W": dW1, "delta_b": db1,}, epoch , ind_batch, _layer = 0)
            send_in_chunks(obj)


        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    def train(self, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001, shuffle=False):
        """Train the model using stochastic gradient descent"""
        if(self.ptas):
            obj = MessageObject(Mode.TRAINING, {"structure": [self.input_size, self.hidden_size, self.output_size]})
            try:
                send_in_chunks(obj)
            except Exception as e:
                print("init")
                print(e)
                return
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(X_train.shape[0])
            if(shuffle):
                X_train = X_train[permutation]
                y_train = y_train[permutation]

            # Train in batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                # Forward pass
                if(self.ptas):
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD, {"X":permutation[i:i + batch_size], "y": permutation[i:i + batch_size]}, epoch , int(i/batch_size))
                    try:
                        send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return

                y_pred = self.forward(X_batch)
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))

            # Compute loss after each epoch
            y_pred = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, y_pred)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        if(self.ptas and not self.operation):
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)
    def end(self):
        if(self.ptas):
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

def send_message(obj, port=5000):
    data = pickle.dumps(obj)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('127.0.0.1', port))
        client_socket.sendall(data)
        if(DEBUG):
            print("Message sent:", obj)

def send_in_chunks(data, port=5000, host='127.0.0.1', chunk_size=1024):
    # Pickle the data
    pickled_data = pickle.dumps(data)

    # Establish a socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # Send data in chunks
        total_data_length = len(pickled_data)
        s.sendall(total_data_length.to_bytes(4, 'big'))  # Send length of the data first (4 bytes)

        # Send data in chunks
        for i in range(0, total_data_length, chunk_size):
            chunk = pickled_data[i:i+chunk_size]
            s.sendall(chunk)
        if(DEBUG):
            print("Data sent successfully.")
        ack = s.recv(1024)
        if pickle.loads(ack) == "ACK":
            if DEBUG:
                print("Acknowledgment received from server.")
