import os
import sys

from NN.utils import writedict
folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)

import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder

import socket
import pickle
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
DEBUG = False


# ------------------------------------------------------------------ #
# Activation functions — accept numpy arrays (legacy callers such as
# eval_5g_noise.py / eval_chapter.py) or torch tensors (NeuralNetwork).
# ------------------------------------------------------------------ #

def binary_activation(x):
    """Binary step activation (+1 / -1)"""
    if isinstance(x, torch.Tensor):
        return torch.where(x >= 0, 1.0, -1.0)
    return np.where(x >= 0, 1, -1)

def binary_activation_derivative(x):
    """Approx derivative for backprop (straight-through estimator)"""
    if isinstance(x, torch.Tensor):
        return (x.abs() <= 1).float()  # 1 in range [-1,1], 0 outside
    return (np.abs(x) <= 1).astype(float)

def softmax(x):
    """Softmax function to output probabilities"""
    if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=-1)
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def relu(x):
    """ReLU activation function"""
    if isinstance(x, torch.Tensor):
        return torch.relu(x)
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    if isinstance(x, torch.Tensor):
        return (x > 0).float()
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation function"""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of Sigmoid"""
    sigmoidval = sigmoid(x)
    return sigmoidval * (1 - sigmoidval)


# Create neural network components from scratch (manual backprop on torch
# tensors so per-layer gradients can be streamed to PTAS).
class NeuralNetwork:

    def __init__(self, input_size, hidden_size=None, output_size=10, hidden_size2=None,
                 hidden_sizes=None, ptas=True, operation=False, port=5000,
                 binary_weights=False, eval=False, device=None):
        """
        Parameters
        ----------
        hidden_sizes : list[int] | None
            Sizes of ALL hidden layers, e.g. [128, 64, 32]. Preferred API.
        hidden_size / hidden_size2 :
            Legacy API (1 or 2 hidden layers); used when hidden_sizes is None.
        device :
            torch device ("cuda", "cpu", torch.device). Defaults to CUDA
            when available.
        """
        if hidden_sizes is None:
            if hidden_size is None:
                raise ValueError("Provide hidden_sizes=[...] or a legacy hidden_size")
            hidden_sizes = [hidden_size] if hidden_size2 is None else [hidden_size, hidden_size2]
        hidden_sizes = [int(h) for h in hidden_sizes]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_size = hidden_sizes[0]
        self.hidden_size2 = hidden_sizes[1] if len(hidden_sizes) > 1 else None
        self.output_size = output_size
        self.operation = operation
        self.port = port
        self.ptas = ptas
        self.binary_weights = binary_weights

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layer_sizes = layer_sizes

        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = torch.randn(fan_in, fan_out, dtype=torch.float32, device=self.device)
            if self.binary_weights:
                W = torch.sign(W)
            else:
                W = W * (2.0 / fan_in) ** 0.5   # He initialization
            self.weights.append(W)
            self.biases.append(torch.zeros(1, fan_out, dtype=torch.float32, device=self.device))

        self._ptas_socket = None
        self.eval = eval

    # ---------------- Legacy attribute access (W1/b1/W2/b2/W3/b3) -------- #
    # Kept for external scripts (external_bridge.py, notebooks) that read or
    # assign individual weight matrices as numpy arrays.

    def _get_np(self, lst, i):
        return lst[i].detach().cpu().numpy()

    def _set_from(self, lst, i, v):
        lst[i] = self._to_tensor(v)

    @property
    def W1(self): return self._get_np(self.weights, 0)
    @W1.setter
    def W1(self, v): self._set_from(self.weights, 0, v)
    @property
    def b1(self): return self._get_np(self.biases, 0)
    @b1.setter
    def b1(self, v): self._set_from(self.biases, 0, v)
    @property
    def W2(self): return self._get_np(self.weights, 1)
    @W2.setter
    def W2(self, v): self._set_from(self.weights, 1, v)
    @property
    def b2(self): return self._get_np(self.biases, 1)
    @b2.setter
    def b2(self, v): self._set_from(self.biases, 1, v)
    @property
    def W3(self): return self._get_np(self.weights, 2)
    @W3.setter
    def W3(self, v): self._set_from(self.weights, 2, v)
    @property
    def b3(self): return self._get_np(self.biases, 2)
    @b3.setter
    def b3(self, v): self._set_from(self.biases, 2, v)

    # --------------------------------------------------------------------- #

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=self.device)

    def structure(self):
        return [self.input_size] + self.hidden_sizes + [self.output_size]

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        return np.sum(log_likelihood) / m

    def forward(self, X, getactivated=False):
        """
        Forward pass over any number of hidden layers.
        Accepts numpy input, computes on self.device, returns numpy output.
        """
        a = self._to_tensor(X)
        L = len(self.weights)
        self._zs = []
        self._activations = [a]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self._zs.append(z)
            if i == L - 1:
                a = softmax(z)
            elif self.binary_weights:
                a = binary_activation(z)
            else:
                a = relu(z)
            self._activations.append(a)

        out = a.detach().cpu().numpy()
        if getactivated:
            hidden_acts = [
                (h > 0).int().cpu().tolist() for h in self._activations[1:-1]
            ]
            # 1 hidden layer keeps the legacy flat layout; ≥2 use one entry per layer.
            activated_neurons = hidden_acts[0] if len(hidden_acts) == 1 else hidden_acts
            if self.ptas:
                obj = MessageObject(Mode.INFERENCE, {"X": X, "inference_path": activated_neurons})
                try:
                    self.send_in_chunks(obj)
                except Exception:
                    pass
            return out, activated_neurons
        return out

    def backward(self, X, y_true, learning_rate=0.001, epoch=0, ind_batch=0):
        """
        Manual backward pass over any number of layers. Streams per-layer
        (delta_W, delta_b) to PTAS from the output layer down to layer 0,
        matching the legacy message order, then applies all updates.
        """
        m = X.shape[0]
        y_t = self._to_tensor(y_true)
        L = len(self.weights)

        dz = self._activations[-1] - y_t   # softmax + cross-entropy gradient
        grads = [None] * L
        for i in range(L - 1, -1, -1):
            a_prev = self._activations[i]
            dW = (a_prev.T @ dz) / m
            db = dz.sum(0, keepdim=True) / m
            grads[i] = (dW, db)
            if self.ptas:
                obj = MessageObject(
                    Mode.TRAINING_BACKPROPAGATION,
                    {"y_true": y_true,
                     "delta_W": dW.detach().cpu().numpy().astype(np.float32),
                     "delta_b": db.detach().cpu().numpy().astype(np.float32)},
                    epoch, ind_batch, _layer=i)
                self.send_in_chunks(obj)
            if i > 0:
                da = dz @ self.weights[i].T
                if self.binary_weights:
                    dz = da * binary_activation_derivative(self._zs[i - 1])
                else:
                    dz = da * relu_derivative(self._zs[i - 1])

        for i, (dW, db) in enumerate(grads):
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
        if self.binary_weights:
            self.weights = [torch.sign(W) for W in self.weights]

    def predict(self, X):
        y_pred = self.forward(X, getactivated=False)
        return np.argmax(y_pred, axis=1)

    def train_old(self, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001, shuffle=False, lr_scheduler=None):
        """Train the model using stochastic gradient descent"""
        if(self.ptas):
            obj = MessageObject(Mode.TRAINING, {"structure": self.structure()})
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
            obj = MessageObject(Mode.TRAINING, {"structure": self.structure(),
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
        """Save all layers as numpy arrays keyed W1/b1..Wn/bn (legacy-compatible)."""
        weights = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases), start=1):
            weights[f"W{i}"] = W.detach().cpu().numpy()
            weights[f"b{i}"] = b.detach().cpu().numpy()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(weights, f)
        print(f"[NN] Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a model saved by save_model (any depth, numpy or torch arrays)."""
        with open(path, "rb") as f:
            weights = pickle.load(f)
        ws, bs = [], []
        i = 1
        while f"W{i}" in weights:
            ws.append(self._to_tensor(weights[f"W{i}"]))
            bs.append(self._to_tensor(weights[f"b{i}"]))
            i += 1
        if not ws:
            raise ValueError(f"No weights found in {path}")
        self.weights, self.biases = ws, bs
        # Refresh architecture metadata from the loaded shapes
        self.input_size = int(ws[0].shape[0])
        self.hidden_sizes = [int(w.shape[1]) for w in ws[:-1]]
        self.output_size = int(ws[-1].shape[1])
        self.hidden_size = self.hidden_sizes[0] if self.hidden_sizes else None
        self.hidden_size2 = self.hidden_sizes[1] if len(self.hidden_sizes) > 1 else None
        self.layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
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
