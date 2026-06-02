import os
import sys

from NN.utils import writedict
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

np.random.seed(42)
DEBUG = False

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

    def __init__(self, input_size, hidden_size, output_size=10, hidden_size2=None, ptas=True, operation=False, port=5000, binary_weights=False, eval=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.operation = operation
        self.port = port
        self.ptas = ptas
        self.binary_weights = binary_weights
        if self.binary_weights:
            self.W1 = np.sign(np.random.randn(input_size, hidden_size))
            self.b1 = np.zeros((1, hidden_size))
            if hidden_size2 is not None:
                self.W2 = np.sign(np.random.randn(hidden_size, hidden_size2))
                self.b2 = np.zeros((1, hidden_size2))
                self.W3 = np.sign(np.random.randn(hidden_size2, output_size))
                self.b3 = np.zeros((1, output_size))
            else:
                self.W2 = np.sign(np.random.randn(hidden_size, output_size))
                self.b2 = np.zeros((1, output_size))
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
            self.b1 = np.zeros((1, hidden_size))
            if hidden_size2 is not None:
                self.W2 = np.random.randn(hidden_size, hidden_size2) * np.sqrt(2. / hidden_size)
                self.b2 = np.zeros((1, hidden_size2))
                self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
                self.b3 = np.zeros((1, output_size))
            else:
                self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
                self.b2 = np.zeros((1, output_size))

        self._ptas_socket = None
        self.eval = eval


    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        return np.sum(log_likelihood) / m

    def forward(self, X, getactivated=False):
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.binary_weights:
            self.a1 = binary_activation(self.z1)
        else:
            self.a1 = relu(self.z1)

        if self.hidden_size2 is not None:
            # --- Hidden layer 2 ---
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            if self.binary_weights:
                self.a2 = binary_activation(self.z2)
            else:
                self.a2 = relu(self.z2)
            # --- Output layer ---
            self.z3 = np.dot(self.a2, self.W3) + self.b3
            self.a3 = softmax(self.z3)
            if getactivated:
                # Two activation vectors, one per hidden layer
                activated_neurons = [
                    (self.a1 > 0).astype(int).tolist(),
                    (self.a2 > 0).astype(int).tolist(),
                ]
                if self.ptas:
                    obj = MessageObject(Mode.INFERENCE, {"X": X, "inference_path": activated_neurons})
                    try:
                        self.send_in_chunks(obj)
                    except Exception:
                        pass
                return self.a3, activated_neurons
            return self.a3
        else:
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = softmax(self.z2)
            if getactivated:
                activated_neurons = (self.a1 > 0).astype(int).tolist()
                if self.ptas:
                    obj = MessageObject(Mode.INFERENCE, {"X": X, "inference_path": activated_neurons})
                    try:
                        self.send_in_chunks(obj)
                    except Exception:
                        pass
                return self.a2, activated_neurons
            return self.a2

    def backward(self, X, y_true, learning_rate=0.001, epoch=0, ind_batch=0):
        m = X.shape[0]

        if self.hidden_size2 is not None:
            # 3-layer backward pass
            dz3 = self.a3 - y_true
            dW3 = np.dot(self.a2.T, dz3) / m
            db3 = np.sum(dz3, axis=0, keepdims=True) / m
            if self.ptas:
                dW3 = dW3.astype(np.float32, copy=False)
                db3 = db3.astype(np.float32, copy=False)
                obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y_true": y_true, "delta_W": dW3, "delta_b": db3}, epoch, ind_batch, _layer=2)
                self.send_in_chunks(obj)

            da2 = np.dot(dz3, self.W3.T)
            if self.binary_weights:
                dz2 = da2 * binary_activation_derivative(self.z2)
            else:
                dz2 = da2 * relu_derivative(self.z2)

            dW2 = np.dot(self.a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            if self.ptas:
                dW2 = dW2.astype(np.float32, copy=False)
                db2 = db2.astype(np.float32, copy=False)
                obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y_true": y_true, "delta_W": dW2, "delta_b": db2}, epoch, ind_batch, _layer=1)
                self.send_in_chunks(obj)

            da1 = np.dot(dz2, self.W2.T)
            if self.binary_weights:
                dz1 = da1 * binary_activation_derivative(self.z1)
            else:
                dz1 = da1 * relu_derivative(self.z1)

            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m
            if self.ptas:
                dW1 = dW1.astype(np.float32, copy=False)
                db1 = db1.astype(np.float32, copy=False)
                obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y_true": y_true, "delta_W": dW1, "delta_b": db1}, epoch, ind_batch, _layer=0)
                self.send_in_chunks(obj)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3

            if self.binary_weights:
                self.W1 = np.sign(self.W1)
                self.W2 = np.sign(self.W2)
                self.W3 = np.sign(self.W3)
        else:
            # 2-layer backward pass
            dz2 = self.a2 - y_true
            dW2 = np.dot(self.a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            if self.ptas:
                dW2 = dW2.astype(np.float32, copy=False)
                db2 = db2.astype(np.float32, copy=False)
                obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y_true": y_true, "delta_W": dW2, "delta_b": db2}, epoch, ind_batch, _layer=1)
                self.send_in_chunks(obj)

            da1 = np.dot(dz2, self.W2.T)
            if self.binary_weights:
                dz1 = da1 * binary_activation_derivative(self.z1)
            else:
                dz1 = da1 * relu_derivative(self.z1)

            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m
            if self.ptas:
                dW1 = dW1.astype(np.float32, copy=False)
                db1 = db1.astype(np.float32, copy=False)
                obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y_true": y_true, "delta_W": dW1, "delta_b": db1}, epoch, ind_batch, _layer=0)
                self.send_in_chunks(obj)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if self.binary_weights:
                self.W1 = np.sign(self.W1)
                self.W2 = np.sign(self.W2)


    def predict(self, X):
        y_pred = self.forward(X, getactivated=False)
        return np.argmax(y_pred, axis=1)


    def train_old(self, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001, shuffle=False, lr_scheduler=None):
        """Train the model using stochastic gradient descent"""
        if(self.ptas):
            structure = ([self.input_size, self.hidden_size, self.output_size]
                         if self.hidden_size2 is None else
                         [self.input_size, self.hidden_size, self.hidden_size2, self.output_size])
            obj = MessageObject(Mode.TRAINING, {"structure": structure})
            try:
                self.send_in_chunks(obj)
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
                        self.send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, epoch, int(i/batch_size))

            y_pred = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, y_pred)

            accuracy = np.mean((y_pred >= 0.5).astype(int).flatten() == y_train.flatten())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        if(self.ptas and not self.operation):
            obj = MessageObject(Mode.END)
            self.send_in_chunks(obj)

    def train(self,
            X_train, y_train,
            X_test=None, y_test=None,
            X_pois_6=None,
            X_non_pois_6=None,
            X_pois_3=None,
            X_non_pois_3=None,
            epochs=10, batch_size=64, shuffle=False, lr_scheduler=None,
            plot=False, fname="Running", get_IPTA=False):
        """Train the binary NN with mini-batch SGD, evaluation, plots & metrics logging."""
        history = {
            "train_acc": [],
            "test_acc": [],
            "pois_acc_label6": [],
            "clean_acc_label6": [],
            "pois_acc_label3": [],
            "clean_acc_label3": []
        }

        acc_pois_6 = acc_pois_3 = acc_clean_6 = acc_clean_3 = np.nan

        if self.ptas:
            if self.hidden_size2 is not None:
                structure = [self.input_size, self.hidden_size, self.hidden_size2, self.output_size]
            else:
                structure = [self.input_size, self.hidden_size, self.output_size]
            obj = MessageObject(Mode.TRAINING, {"structure": structure,
                                                "batch_size": batch_size, "total_rounds": epochs * (X_train.shape[0] // batch_size)})
            try:
                self.send_in_chunks(obj)
            except Exception as e:
                print("init")
                print(e)
                return

        # Record accuracy before any training (iteration 0)
        init_train_acc = np.mean(np.argmax(self.forward(X_train), axis=1) == np.argmax(y_train, axis=1))
        init_test_acc = np.nan
        if X_test is not None and y_test is not None:
            init_test_acc = np.mean(np.argmax(self.forward(X_test), axis=1) == np.argmax(y_test, axis=1))
        if X_pois_3 is not None:
            acc_pois_6  = np.mean(self.predict(X_pois_6)     == 6)
            acc_pois_3  = np.mean(self.predict(X_pois_3)     == 3)
            acc_clean_6 = np.mean(self.predict(X_non_pois_6) == 6)
            acc_clean_3 = np.mean(self.predict(X_non_pois_3) == 3)
        history["train_acc"].append(init_train_acc)
        history["test_acc"].append(init_test_acc)
        history["pois_acc_label6"].append(acc_pois_6  if X_pois_6    is not None else np.nan)
        history["clean_acc_label6"].append(acc_clean_6 if X_non_pois_6 is not None else np.nan)
        history["pois_acc_label3"].append(acc_pois_3  if X_pois_3    is not None else np.nan)
        history["clean_acc_label3"].append(acc_clean_3 if X_non_pois_3 is not None else np.nan)

        X_train_orig = X_train
        y_train_orig = y_train

        for epoch in range(epochs):
            permutation = np.arange(X_train_orig.shape[0])
            if shuffle:
                permutation = np.random.permutation(X_train_orig.shape[0])
            X_train_epoch = X_train_orig[permutation]
            y_train_epoch = y_train_orig[permutation]
            print(f"Epoch {epoch+1}/{epochs}")

            # --- Mini-batch loop ---
            for i in tqdm(range(0, X_train_epoch.shape[0], batch_size)):
                X_batch = X_train_epoch[i:i + batch_size]
                y_batch = y_train_epoch[i:i + batch_size]

                if self.ptas:
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD,
                                        {"X": permutation[i:i + batch_size], "y": permutation[i:i + batch_size]},
                                        epoch, int(i/batch_size))
                    try:
                        self.send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return

                y_pred = self.forward(X_batch)
                current_lr = lr_scheduler(epoch)
                self.backward(X_batch, y_batch, current_lr, epoch, int(i/batch_size))

            # --- Epoch-level evaluation (once per epoch, not per batch) ---
            y_pred_train = self.forward(X_train_orig)
            train_acc = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(y_train_orig, axis=1))

            if X_test is not None and y_test is not None:
                y_pred_test = self.forward(X_test)
                test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
            else:
                test_acc = np.nan

            if X_pois_3 is not None:
                pois_predictions_6 = self.predict(X_pois_6)
                pois_predictions_3 = self.predict(X_pois_3)
                predictions_6 = self.predict(X_non_pois_6)
                predictions_3 = self.predict(X_non_pois_3)
                acc_pois_6  = np.mean(pois_predictions_6 == 6)
                acc_pois_3  = np.mean(pois_predictions_3 == 3)
                acc_clean_6 = np.mean(predictions_6 == 6)
                acc_clean_3 = np.mean(predictions_3 == 3)

            pois_acc_label6  = acc_pois_6  if X_pois_6    is not None else np.nan
            clean_acc_label6 = acc_clean_6 if X_non_pois_6 is not None else np.nan
            pois_acc_label3  = acc_pois_3  if X_pois_3    is not None else np.nan
            clean_acc_label3 = acc_clean_3 if X_non_pois_3 is not None else np.nan

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
            self.send_in_chunks(obj)
            self._close_ptas_socket()

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
            batches_per_epoch = 1  # one point per epoch now
            for e in range(1, epochs):
                plt.axvline(x=e * batches_per_epoch, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Evolution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{fname}/accuracy_evolution.pdf", dpi=300, bbox_inches="tight")
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
                writedict(metrics_2, f"{fname}/ipta_metrics.txt")

            writedict(metrics, f"{fname}/metrics.txt")
            self.save_model(f"{fname}/nn_model.pkl")
        return history

    def save_model(self, path: str) -> None:
        weights = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        if self.hidden_size2 is not None:
            weights["W3"] = self.W3
            weights["b3"] = self.b3
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(weights, f)
        print(f"[NN] Model saved to {path}")

    def load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            weights = pickle.load(f)
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]
        if "W3" in weights:
            self.W3 = weights["W3"]
            self.b3 = weights["b3"]
        print(f"[NN] Model loaded from {path}")

    def end(self):
        if self.ptas:
            obj = MessageObject(Mode.END)
            try:
                self.send_in_chunks(obj)
            except Exception:
                pass
            self._close_ptas_socket()

    def predict_bin(self, X):
        """Make binary predictions"""
        y_pred = self.forward(X, getactivated=False)
        return (y_pred >= 0.5).astype(int).flatten()

    def send_message(self,obj):
        data = pickle.dumps(obj)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(('127.0.0.1', self.port))
            client_socket.sendall(data)
            if(DEBUG):
                print("Message sent:", obj)

    def _ensure_ptas_socket(self, host='127.0.0.1'):
        if getattr(self, "_ptas_socket", None) is not None:
            return self._ptas_socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, self.port))
        except Exception:
            s.close()
            raise
        self._ptas_socket = s
        return self._ptas_socket

    def _close_ptas_socket(self):
        sock = getattr(self, "_ptas_socket", None)
        if sock is None:
            return
        try:
            sock.close()
        finally:
            self._ptas_socket = None

    def send_in_chunks(self, data, host='127.0.0.1', chunk_size=1024):
        pickled_data = pickle.dumps(data)

        def _send(sock):
            total_data_length = len(pickled_data)
            sock.sendall(total_data_length.to_bytes(4, 'big'))
            for i in range(0, total_data_length, chunk_size):
                chunk = pickled_data[i:i+chunk_size]
                sock.sendall(chunk)
            if(DEBUG):
                print("Data sent successfully.")
            ack = sock.recv(1024)
            if pickle.loads(ack) == "ACK" and DEBUG:
                print("Acknowledgment received from server.")

        s = self._ensure_ptas_socket(host=host)
        try:
            _send(s)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._close_ptas_socket()
            s = self._ensure_ptas_socket(host=host)
            _send(s)
