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
        # print(np.shape(y_true))
        # print(y_true)
        # print(np.shape(y_pred))
        # print(y_pred)
  
        """Cross-entropy loss"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def forward(self, X, getactivated=False):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        # self.a1 = np.maximum(0, self.z1)  # ReLU activation
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
        # print("DW2: ", np.shape(dW2))
        # print("DB2: ", np.shape(db2))
        # print("DW1: ", np.shape(dW1))
        # print("DB1: ", np.shape(db1))
        # raise NameError

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
                ## Send feed forward data to PTAS 
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
            # accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1))
            # print(f"Test Accuracy: {accuracy * 100:.2f}%")
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
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
        """Train the model using mini-batch SGD and plot accuracy evolution.
        Evaluation is done after each batch on the *whole* datasets.
        """
        # if(not plot):
        #     return self.train_old(X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001, shuffle=False)
        # History containers
        history = {
            "train_acc": [],
            "test_acc": [],
            "pois_acc_label6": [],
            "clean_acc_label6": [],
            "pois_acc_label3": [],
            "clean_acc_label3": []
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
                if(self.ptas):
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD, {"X":permutation[i:i + batch_size], "y": permutation[i:i + batch_size]}, epoch , int(i/batch_size))
                    try:
                        send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return
                # Forward + backward on this batch
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))

                # --- Evaluation after this batch ---
                # Train acc (on full training set)
                y_pred_train = self.forward(X_train)
                train_acc = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(y_train, axis=1))

                # Test acc (on full test set)
                if X_test is not None and y_test is not None:
                    y_pred_test = self.forward(X_test)
                    test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
                else:
                    test_acc = np.nan

                # Poisoned/clean acc (labels are known)
                pois_acc_label6 = np.mean(self.predict(X_pois_6) == 6) if X_pois_6 is not None else np.nan
                clean_acc_label6 = np.mean(self.predict(X_non_pois_6) == 6) if X_non_pois_6 is not None else np.nan
                pois_acc_label3 = np.mean(self.predict(X_pois_3) == 3) if X_pois_3 is not None else np.nan
                clean_acc_label3 = np.mean(self.predict(X_non_pois_3) == 3) if X_non_pois_3 is not None else np.nan

                # Save history
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["pois_acc_label6"].append(pois_acc_label6)
                history["clean_acc_label6"].append(clean_acc_label6)
                history["pois_acc_label3"].append(pois_acc_label3)
                history["clean_acc_label3"].append(clean_acc_label3)

                # Print iteration summary
            print(f"Epoch {epoch+1}/{epochs}| "
                    f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                    f"Pois6: {pois_acc_label6:.4f}, Clean6: {clean_acc_label6:.4f}, "
                    f"Pois3: {pois_acc_label3:.4f}, Clean3: {clean_acc_label3:.4f}")

        if self.ptas and not self.operation:
            obj = MessageObject(Mode.END)
            send_in_chunks(obj)

        # --- Plot if requested ---
        if plot:
            iterations = range(len(history["train_acc"]))
            plt.figure(figsize=(10,6))

            # Accuracy curves
            plt.plot(iterations, history["train_acc"], label="Train")
            plt.plot(iterations, history["test_acc"], label="Test")
            if(X_pois_6 is not None):
                plt.plot(iterations, history["pois_acc_label6"], label="Poisoned Images 6")
                plt.plot(iterations, history["clean_acc_label6"], label="Clean Images 6")
                plt.plot(iterations, history["pois_acc_label3"], label="Poisoned Images 3")
                plt.plot(iterations, history["clean_acc_label3"], label="Clean Images 3")
            batches_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))
            for e in range(1, epochs):
                plt.axvline(x=e * batches_per_epoch, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("Iteration (batches across epochs)")

            # plt.xlabel("epochs")

            plt.ylabel("Accuracy")
            plt.title("Accuracy Evolution")
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
            
            if(get_IPTA):
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
        """Make predictions"""
        y_pred = self.forward(X, getactivated=False)
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
    

