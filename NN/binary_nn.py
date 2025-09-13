import os 
import sys 
folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)

import numpy as np

from sklearn.preprocessing import OneHotEncoder

import socket
import pickle 
from PTASTemp.messageObject import MessageObject 
from PTASTemp.mode import Mode
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


DEBUG = True 

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
    def __init__(self, input_size, hidden_size, output_size=1, activations=[relu], ptas=True, operation=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1  # force binary classification
        self.operation = operation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        self.ptas = ptas 
        self.activations = activations 
 
    def binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def forward(self, X, getactivated=False):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # sigmoid for binary classification
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
        dz2 = self.a2 - y_true.reshape(-1, 1)
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

    def train_old(self, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001, shuffle=False):
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
            permutation = np.random.permutation(X_train.shape[0])
            if(shuffle):
                X_train = X_train[permutation]
                y_train = y_train[permutation]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                if(self.ptas):
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD, {"X":permutation[i:i + batch_size], "y": permutation[i:i + batch_size]}, epoch , int(i/batch_size))
                    try:
                        send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))
            
            y_pred = self.forward(X_train)
            loss = self.binary_cross_entropy(y_train, y_pred)
            accuracy = np.mean((y_pred >= 0.5).astype(int).flatten() == y_train.flatten())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        if(self.ptas and not self.operation):
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)

    def train(self, 
            X_train, y_train, 
            X_test=None, y_test=None,
            X_pois_6=None, 
            X_non_pois_6=None, 
            X_pois_3=None, 
            X_non_pois_3=None, 
            epochs=10, batch_size=64, learning_rate=0.001, shuffle=False,
            plot=False, fname="defaut", get_IPTA=False):
        history = {
            "train_acc": [],
            "test_acc": []
        }

        if self.ptas:
            obj = MessageObject(Mode.TRAINING, {"structure": [self.input_size, self.hidden_size, self.output_size], "batch_size": batch_size})
            try:
                send_in_chunks(obj)
            except Exception as e:
                print("init")
                print(e)
                return 

        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            if shuffle:
                X_train = X_train[permutation]
                y_train = y_train[permutation]
            print(f"Epoch {epoch+1}/{epochs}")
            for i in tqdm(range(0, X_train.shape[0], batch_size)):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                if(self.ptas):
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD, {"X":permutation[i:i + batch_size], "y": permutation[i:i + batch_size]}, epoch , int(i/batch_size))
                    try:
                        send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))

                # Evaluation
                y_pred_train = self.forward(X_train)
                train_acc = np.mean((y_pred_train >= 0.5).astype(int).flatten() == y_train.flatten())

                if X_test is not None and y_test is not None:
                    y_pred_test = self.forward(X_test)
                    test_acc = np.mean((y_pred_test >= 0.5).astype(int).flatten() == y_test.flatten())
                else:
                    test_acc = np.nan

                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)

            print(f"Epoch {epoch+1}/{epochs}| Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        if self.ptas and not self.operation:
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)

        return history

    def end(self):
        if self.ptas:
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)

    def predict(self, X):
        """Make binary predictions"""
        y_pred = self.forward(X, getactivated=False)
        return (y_pred >= 0.5).astype(int).flatten()
    
def send_message(obj, port=5000):
    data = pickle.dumps(obj)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('127.0.0.1', port))
        client_socket.sendall(data)
        if(DEBUG):
            print("Message sent:", obj)

def send_in_chunks(data, port=5000, host='127.0.0.1', chunk_size=1024):
    pickled_data = pickle.dumps(data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        total_data_length = len(pickled_data)
        s.sendall(total_data_length.to_bytes(4, 'big'))
        for i in range(0, total_data_length, chunk_size):
            chunk = pickled_data[i:i+chunk_size]
            s.sendall(chunk)
        if(DEBUG):
            print("Data sent successfully.")
        ack = s.recv(1024)
        if pickle.loads(ack) == "ACK":
            if DEBUG:
                print("Acknowledgment received from server.")
