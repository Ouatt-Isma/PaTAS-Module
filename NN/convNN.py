"""Convolutional ResNet-style clients with PaTAS gradient streaming.

PaTAS mirrors each weight layer of these networks with an opinion matrix:

    conv layer   →  omega of shape (c_in·k² + 1, c_out)
                    (the flattened kernel acts as a dense map from the
                    receptive-field inputs to the output channels)
    dense layer  →  omega of shape (in + 1, out)   — as in the MLP case

Under **spatially uniform input trust** (the trust/vacuous/distrust fills used
throughout the experiments) this reduction is exact: every output location of
a convolution sees identically distributed input opinions, so opinion
propagation through the conv layer equals the standard dense discount-fusion
rule applied to the flattened kernel matrix.  Pooling and ReLU are identity in
opinion space; a residual skip connection becomes an averaging-fusion of the
block-input opinion with the block-output opinion.

The network architecture is described by a list of layer *specs* shared with
PTASConv (NN/convPTAS.py):

    {"kind": "conv",    "cin": _, "cout": _, "k": _, "pool": bool}
    {"kind": "resconv", "cin": _, "cout": _, "k": _, "pool": bool}   (cin == cout)
    {"kind": "dense",   "cin": _, "spatial": _, "out": _}

Gradients are computed with torch autograd (manual backprop through conv +
skip connections would be error-prone) and streamed per weight layer to PTAS
in the same MessageObject format used by the MLP client.
"""

import os
import pickle
import sys

folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)

import numpy as np
import torch
import torch.nn.functional as F

from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
from NN.primaryNN import NeuralNetwork as _NNBase
from tqdm import tqdm

torch.manual_seed(42)


def default_resnet_lite_specs(img_size=28, in_channels=1, num_classes=10,
                              base_channels=8, k=3):
    """ResNet-lite for MNIST/grayscale: conv → residual block → fc (4 omegas)."""
    c = base_channels
    s = img_size // 4   # two 2× poolings
    return [
        {"kind": "conv",    "cin": in_channels, "cout": c, "k": k, "pool": True},
        {"kind": "resconv", "cin": c, "cout": c, "k": k, "pool": True},
        {"kind": "dense",   "cin": c, "spatial": s * s, "out": num_classes},
    ]


def cifar10_resnet_specs(img_size=32, in_channels=3, num_classes=10,
                         base_channels=16, k=3):
    """Three-stage CIFAR-10 ResNet (conv/resconv per stage, widening 1×/2×/4×).

    With base_channels=16: 7 conv layers + fc (10 omegas, ~155k parameters),
    spatial 32 → 16 → 8 → 4, classifier over 4·c·(img/8)² features.
    Same spirit as the classic CIFAR ResNet family (stages of residual blocks
    with doubling widths), sized to stay trainable on CPU.
    """
    c = base_channels
    s = img_size // 8   # three 2× poolings
    return [
        {"kind": "conv",    "cin": in_channels, "cout": c,     "k": k, "pool": True},
        {"kind": "resconv", "cin": c,     "cout": c,     "k": k, "pool": False},
        {"kind": "conv",    "cin": c,     "cout": 2 * c, "k": k, "pool": True},
        {"kind": "resconv", "cin": 2 * c, "cout": 2 * c, "k": k, "pool": False},
        {"kind": "conv",    "cin": 2 * c, "cout": 4 * c, "k": k, "pool": True},
        {"kind": "resconv", "cin": 4 * c, "cout": 4 * c, "k": k, "pool": False},
        {"kind": "dense",   "cin": 4 * c, "spatial": s * s, "out": num_classes},
    ]


def spec_omega_shapes(specs):
    """(rows, cols) of each PaTAS opinion matrix, one per weight layer."""
    shapes = []
    for sp in specs:
        if sp["kind"] == "conv":
            shapes.append((sp["cin"] * sp["k"] ** 2 + 1, sp["cout"]))
        elif sp["kind"] == "resconv":
            shapes.append((sp["cin"] * sp["k"] ** 2 + 1, sp["cout"]))
            shapes.append((sp["cout"] * sp["k"] ** 2 + 1, sp["cout"]))
        elif sp["kind"] == "dense":
            shapes.append((sp["cin"] * sp["spatial"] + 1, sp["out"]))
        else:
            raise ValueError(f"Unknown spec kind: {sp['kind']}")
    return shapes


class ConvNet:
    """Spec-driven ResNet-style CNN with PaTAS streaming."""

    # Reuse the MLP client's socket plumbing (methods only touch
    # self.port / self._ptas_socket).
    _ensure_ptas_socket = _NNBase._ensure_ptas_socket
    _close_ptas_socket = _NNBase._close_ptas_socket
    send_in_chunks = _NNBase.send_in_chunks
    send_message = _NNBase.send_message

    def __init__(self, img_size=28, in_channels=1, num_classes=10, base_channels=8,
                 k=3, ptas=True, operation=False, port=5000, device=None, specs=None):
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ptas = ptas
        self.operation = operation
        self.port = port
        self._ptas_socket = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.specs = specs if specs is not None else default_resnet_lite_specs(
            img_size, in_channels, num_classes, base_channels, k)
        dev = self.device

        def _conv_w(cout, cin, kk):
            w = torch.randn(cout, cin, kk, kk, device=dev) * (2.0 / (cin * kk * kk)) ** 0.5
            return w.requires_grad_(True)

        def _bias(cout):
            return torch.zeros(cout, device=dev, requires_grad=True)

        # one block per spec; a block holds 1 (conv/dense) or 2 (resconv) weight
        # layers, in the same order PaTAS numbers its omegas
        self._blocks = []
        for sp in self.specs:
            kind = sp["kind"]
            if kind == "conv":
                kk = sp["k"]
                weights = [(_conv_w(sp["cout"], sp["cin"], kk), _bias(sp["cout"]))]
            elif kind == "resconv":
                assert sp["cin"] == sp["cout"], "resconv requires cin == cout (identity skip)"
                kk = sp["k"]
                weights = [
                    (_conv_w(sp["cout"], sp["cin"], kk), _bias(sp["cout"])),
                    (_conv_w(sp["cout"], sp["cout"], kk), _bias(sp["cout"])),
                ]
            elif kind == "dense":
                fc_in = sp["cin"] * sp["spatial"]
                W = (torch.randn(fc_in, sp["out"], device=dev)
                     * (2.0 / fc_in) ** 0.5).requires_grad_(True)
                b = torch.zeros(1, sp["out"], device=dev, requires_grad=True)
                weights = [(W, b)]
            else:
                raise ValueError(f"Unknown spec kind: {kind}")
            self._blocks.append({"sp": sp, "weights": weights})

        self._params = [p for blk in self._blocks for wb in blk["weights"] for p in wb]

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32,
                               device=self.device)

    def _forward_logits(self, xt: torch.Tensor) -> torch.Tensor:
        h = xt.view(-1, self.in_channels, self.img_size, self.img_size)
        for blk in self._blocks:
            sp = blk["sp"]
            kind = sp["kind"]
            if kind == "conv":
                (w, b), = blk["weights"]
                h = F.relu(F.conv2d(h, w, b, padding=sp["k"] // 2))
                if sp.get("pool", True):
                    h = F.avg_pool2d(h, 2)
            elif kind == "resconv":
                (w1, b1), (w2, b2) = blk["weights"]
                pad = sp["k"] // 2
                r = F.relu(F.conv2d(h, w1, b1, padding=pad))
                r = F.conv2d(r, w2, b2, padding=pad)
                h = F.relu(h + r)                    # residual skip connection
                if sp.get("pool", True):
                    h = F.avg_pool2d(h, 2)
            else:  # dense
                (W, b), = blk["weights"]
                h = h.reshape(h.shape[0], -1) @ W + b
        return h

    def forward(self, X, chunk=4096):
        """Softmax probabilities; evaluates in chunks to bound activation memory."""
        outs = []
        with torch.no_grad():
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            for i in range(0, max(n, 1), chunk):
                logits = self._forward_logits(self._to_tensor(X[i:i + chunk]))
                outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def _omega_grads(self):
        """Per weight layer (in omega order): (delta_W, delta_b) in omega layout.

        Conv kernel grad (cout, cin, k, k) → (cin·k², cout) with row index
        c·k² + u·k + v — the same flattening PTASConv uses when tiling
        channel opinions.
        """
        grads = []
        for blk in self._blocks:
            if blk["sp"]["kind"] == "dense":
                W, b = blk["weights"][0]
                grads.append((W.grad, b.grad.reshape(1, -1)))
            else:
                for w, b in blk["weights"]:
                    g = w.grad.permute(1, 2, 3, 0).reshape(-1, w.shape[0])
                    grads.append((g, b.grad.reshape(1, -1)))
        return grads

    def train(self, X_train, y_train, X_test=None, y_test=None,
              epochs=10, batch_size=128, shuffle=True, lr_scheduler=None):
        """Mini-batch SGD with autograd; streams per-layer deltas to PTAS."""
        if lr_scheduler is None:
            lr_scheduler = lambda e: 0.05
        history = {"train_acc": [], "test_acc": []}
        n = X_train.shape[0]

        if self.ptas:
            obj = MessageObject(Mode.TRAINING, {
                "structure": self.specs,
                "batch_size": batch_size,
                "total_rounds": epochs * (n // batch_size),
            })
            try:
                self.send_in_chunks(obj)
            except Exception as e:
                print("init")
                print(e)
                return history

        for epoch in range(epochs):
            permutation = np.random.permutation(n) if shuffle else np.arange(n)
            X_epoch = X_train[permutation]
            y_epoch = y_train[permutation]
            lr = lr_scheduler(epoch)
            print(f"Epoch {epoch+1}/{epochs}")

            for i in tqdm(range(0, n, batch_size)):
                X_batch = self._to_tensor(X_epoch[i:i + batch_size])
                y_batch = y_epoch[i:i + batch_size]
                y_t = self._to_tensor(y_batch)

                if self.ptas:
                    obj = MessageObject(Mode.TRAINING_FEEDFORWARD,
                                        {"X": permutation[i:i + batch_size],
                                         "y": permutation[i:i + batch_size]},
                                        epoch, int(i / batch_size))
                    try:
                        self.send_in_chunks(obj)
                    except Exception as e:
                        print("during")
                        print(e)
                        return history

                for p in self._params:
                    if p.grad is not None:
                        p.grad = None
                logits = self._forward_logits(X_batch)
                loss = F.cross_entropy(logits, y_t.argmax(dim=1))
                loss.backward()

                if self.ptas:
                    grads = self._omega_grads()
                    for layer in range(len(grads) - 1, -1, -1):
                        dW, db = grads[layer]
                        obj = MessageObject(
                            Mode.TRAINING_BACKPROPAGATION,
                            {"y_true": y_batch,
                             "delta_W": dW.detach().cpu().numpy().astype(np.float32),
                             "delta_b": db.detach().cpu().numpy().astype(np.float32)},
                            epoch, int(i / batch_size), _layer=layer)
                        self.send_in_chunks(obj)

                with torch.no_grad():
                    for p in self._params:
                        p -= lr * p.grad

            train_acc = float(np.mean(self.predict(X_train) == np.argmax(y_train, axis=1)))
            if X_test is not None and y_test is not None:
                test_acc = float(np.mean(self.predict(X_test) == np.argmax(y_test, axis=1)))
            else:
                test_acc = float("nan")
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        if self.ptas and not self.operation:
            obj = MessageObject(Mode.END)
            self.send_in_chunks(obj)
            self._close_ptas_socket()
        return history

    def save_model(self, path: str) -> None:
        """Save all weight layers (omega order) as numpy arrays keyed W1/b1..Wn/bn."""
        weights = {}
        i = 1
        for blk in self._blocks:
            for w, b in blk["weights"]:
                weights[f"W{i}"] = w.detach().cpu().numpy()
                weights[f"b{i}"] = b.detach().cpu().numpy()
                i += 1
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(weights, f)
        print(f"[NN] Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load weights saved by save_model into the spec-built layers.

        Raises ValueError if the file's layer count doesn't match the current
        specs; torch raises on per-layer shape mismatch.
        """
        with open(path, "rb") as f:
            weights = pickle.load(f)
        i = 1
        with torch.no_grad():
            for blk in self._blocks:
                for w, b in blk["weights"]:
                    if f"W{i}" not in weights:
                        raise ValueError(
                            f"{path} has fewer weight layers than the current specs")
                    w.copy_(self._to_tensor(weights[f"W{i}"]))
                    b.copy_(self._to_tensor(weights[f"b{i}"]).reshape(b.shape))
                    i += 1
        if f"W{i}" in weights:
            raise ValueError(f"{path} has more weight layers than the current specs")
        print(f"[NN] Model loaded from {path}")

    def end(self):
        if self.ptas:
            try:
                self.send_in_chunks(MessageObject(Mode.END))
            except Exception:
                pass
            self._close_ptas_socket()
