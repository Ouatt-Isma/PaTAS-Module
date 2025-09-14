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

def binary_activation(x):
    """Binary step activation (+1 / -1)"""
    return np.where(x >= 0, 1, -1)

def binary_activation_derivative(x):
    """Approx derivative for backprop (straight-through estimator)"""
    return (np.abs(x) <= 1).astype(float)  # 1 in range [-1,1], 0 outside

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

    def __init__(self, input_size, hidden_size, output_size=10, ptas=True, operation=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.operation = operation

        # Initialize weights (binary)
        self.W1 = np.sign(np.random.randn(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.sign(np.random.randn(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        self.ptas = ptas

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        return np.sum(log_likelihood) / m

    def forward(self, X, getactivated=False):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = binary_activation(self.z1)   # binary hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)             # softmax output
        if getactivated:
            activated_neurons = (self.a1 > 0).astype(int).tolist()
            if self.ptas:
                obj = MessageObject(Mode.INFERENCE, {"X": X, "inference_path": activated_neurons})
                send_in_chunks(obj)
            return self.a2, activated_neurons
        return self.a2

    def backward(self, X, y_true, learning_rate=0.001, epoch=0, ind_batch=0):
        m = X.shape[0]
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * binary_activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient update (float)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        # Project weights back to binary
        self.W1 = np.sign(self.W1)
        self.W2 = np.sign(self.W2)

    def predict(self, X):
        y_pred = self.forward(X, getactivated=False)
        return np.argmax(y_pred, axis=1)


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
        """Train the binary NN with mini-batch SGD, evaluation, plots & metrics logging."""
        # History containers
        history = {
            "train_acc": [],
            "test_acc": [],
            "pois_acc_label6": [],
            "clean_acc_label6": [],
            "pois_acc_label3": [],
            "clean_acc_label3": []
        }

        # --- Convert one-hot to single column if needed ---
        # if y_train.ndim > 1 and y_train.shape[1] > 1:
        #     y_train = y_train[:, 1].reshape(-1, 1)
        # if y_test is not None and y_test.ndim > 1 and y_test.shape[1] > 1:
        #     y_test = y_test[:, 1].reshape(-1, 1)

        if self.ptas:
            obj = MessageObject(Mode.TRAINING, {"structure": [self.input_size, self.hidden_size, self.output_size],
                                                "batch_size": batch_size})
            try:
                send_in_chunks(obj)
            except Exception as e:
                print("init")
                print(e)
                return 

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            if shuffle:
                X_train = X_train[permutation]
                y_train = y_train[permutation]
            print(f"Epoch {epoch+1}/{epochs}")

            # --- Mini-batch loop ---
            for i in tqdm(range(0, X_train.shape[0], batch_size)):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                if self.ptas:
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD,
                                        {"X": permutation[i:i + batch_size], "y": permutation[i:i + batch_size]},
                                        epoch, int(i/batch_size))
                    try:
                        send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return

                # Forward + backward
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))

                # --- Evaluation after this batch ---
                # Train acc
                y_pred_train = self.forward(X_train)
                train_acc = np.mean((y_pred_train >= 0.5).astype(int).flatten() == y_train.flatten())

                # Test acc
                if X_test is not None and y_test is not None:
                    y_pred_test = self.forward(X_test)
                    test_acc = np.mean((y_pred_test >= 0.5).astype(int).flatten() == y_test.flatten())
                else:
                    test_acc = np.nan

                # Poisoned/clean acc (optional)
                pois_acc_label6 = np.mean(self.predict(X_pois_6) == 1) if X_pois_6 is not None else np.nan
                clean_acc_label6 = np.mean(self.predict(X_non_pois_6) == 0) if X_non_pois_6 is not None else np.nan
                pois_acc_label3 = np.mean(self.predict(X_pois_3) == 1) if X_pois_3 is not None else np.nan
                clean_acc_label3 = np.mean(self.predict(X_non_pois_3) == 0) if X_non_pois_3 is not None else np.nan

                # Save history
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["pois_acc_label6"].append(pois_acc_label6)
                history["clean_acc_label6"].append(clean_acc_label6)
                history["pois_acc_label3"].append(pois_acc_label3)
                history["clean_acc_label3"].append(clean_acc_label3)

            print(f"Epoch {epoch+1}/{epochs}| "
                    f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                    f"Pois6: {pois_acc_label6:.4f}, Clean6: {clean_acc_label6:.4f}, "
                    f"Pois3: {pois_acc_label3:.4f}, Clean3: {clean_acc_label3:.4f}")

        if self.ptas and not self.operation:
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)

        # --- Plot & log if requested ---
        if plot:
            iterations = range(len(history["train_acc"]))
            plt.figure(figsize=(10,6))
            plt.plot(iterations, history["train_acc"], label="Train")
            plt.plot(iterations, history["test_acc"], label="Test")
            if X_pois_6 is not None:
                plt.plot(iterations, history["pois_acc_label6"], label="Poisoned Images 6")
                plt.plot(iterations, history["clean_acc_label6"], label="Clean Images 6")
                plt.plot(iterations, history["pois_acc_label3"], label="Poisoned Images 3")
                plt.plot(iterations, history["clean_acc_label3"], label="Clean Images 3")
            batches_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))
            for e in range(1, epochs):
                plt.axvline(x=e * batches_per_epoch, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("Iteration (batches across epochs)")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Evolution (Binary)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{fname}.pdf", dpi=300, bbox_inches="tight")
            plt.close()

            last = lambda k: (history[k][-1] if history[k] else float('nan'))
            metrics = {
                "Train": last("train_acc"),
                "Test": last("test_acc"),
                "Poisoned Images 6": last("pois_acc_label6"),
                "Clean Images 6": last("clean_acc_label6"),
                "Poisoned Images 3": last("pois_acc_label3"),
                "Clean Images 3": last("clean_acc_label3"),
            }

            if get_IPTA:
                metrics_2 = {}
                _, ipta_6_p = self.forward(X_pois_6[0], getactivated=True)  
                _, ipta_3_p = self.forward(X_pois_3[0], getactivated=True) 
                _, ipta_6_s = self.forward(X_non_pois_6[0], getactivated=True) 
                _, ipta_3_s = self.forward(X_non_pois_3[0], getactivated=True) 
                metrics_2["IPTA 6 Pois"] = ipta_6_p
                metrics_2["IPTA 3 Pois"] = ipta_3_p
                metrics_2["IPTA 6 Safe"] = ipta_6_s
                metrics_2["IPTA 3 Safe"] = ipta_3_s
                with open(f"{fname}.txt", "w") as f:
                    for label, val in metrics_2.items():
                        f.write(f"{label}: {val}\n")

            with open(f"{fname}.txt", "a") as f:
                for label, val in metrics.items():
                    f.write(f"{label}: {('nan' if np.isnan(val) else f'{val:.4f}')}\n")

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
