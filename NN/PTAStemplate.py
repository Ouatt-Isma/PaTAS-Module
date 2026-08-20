"""PTAS runtime implementation used by the listener-side trust assessment process."""

import os
import sys

from concrete.TensorTO import (
    TensorArrayTO,
    normalize_tensor,
    as_tensor,
    to_numpy,
    default_device,
    fill as tfill,
    _t102,
    _is_torch,
    depth_normalize_vec,
)

sys.path.append(os.getcwd())
folder_path = os.getcwd()
import socket
import pickle
from PTASTemp.ptasInterface import PTASInterface
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from concrete.TrustOpinion import TrustOpinion
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # For custom legend handles
from .utils import writeto
from tqdm import tqdm

from concrete.ArrayTO import ArrayTO
import time
# DEBUG levels: 0=quiet, 1=flow/info, 2=verbose details.
DEBUG = 0
CHUNK_SIZE = 1024
fuse_func = TrustOpinion.avFuseGen

def opinion_to_tensor(op, dtype=np.float32):
    """
    Convert TrustOpinion or array-like into tensor (3,).
    Torch tensors stay torch; numpy stays numpy.
    """
    if isinstance(op, TrustOpinion):
        return np.array([op.t, op.d, op.u], dtype=dtype)

    if _is_torch(op):
        if op.ndim == 1:
            return op
        if op.ndim == 3:
            return op[0, 0]
        if op.ndim == 2:
            return op[0]

    if isinstance(op, np.ndarray):
        if op.shape == (3,):
            return op.astype(dtype, copy=False)
        if op.ndim == 3:
            return op[0, 0].astype(dtype, copy=False)
        if op.ndim == 2:
            return op[0].astype(dtype, copy=False)

    if isinstance(op, (list, tuple)):
        return np.array(op, dtype=dtype)

    raise TypeError(f"Unsupported opinion type: {type(op)}")

def get_first_opinion(tensor):
    v = tensor.value
    if v.ndim == 3:
        return v[0, 0]
    if v.ndim == 2:
        return v[0]
    raise ValueError("Unexpected tensor shape")
class PTAS:
    def __init__(
    self,
    omega_thetas: list,              # list[ArrayTO] originally
    operator_mapping: str,
    nn_interface: PTASInterface,
    trust_assessment_func,
    epsilon_low=0.5,
    epsilon_up=100,
    structure=None,
    learning_rate=TrustOpinion.ftrust(),
    nntype="linear",
    eval=False,
    patch=None,
    use_tensor=True,
    tensor_dtype=np.float32,
    no_round = None,
    device=None,
    fuse_method="average",   # trust-revision fusion: average | cumulative | weighted | compromise | constraint
):
        """
        Initialize the PTAS with essential components.

        This patched version stores omega_thetas as tensors (n,m,3) for fast
        ops — torch tensors on `device` (CUDA when available) so trust
        propagation runs on GPU.
        """
        # Scale-tied threshold: epsilon_low="auto" or "auto<c>" ties the
        # gradient-stability threshold to the task's own gradient scale,
        # eps_layer = c * median(|delta|), estimated per layer over the first
        # revision batches and frozen afterwards (c defaults to 0.5).
        self._eps_auto_coef = None
        self._eps_layer = {}
        self._eps_hist = {}
        self._eps_freeze_n = 50
        if isinstance(epsilon_low, str):
            if not epsilon_low.startswith("auto"):
                raise ValueError(f"epsilon_low string must be 'auto[<c>]', got {epsilon_low!r}")
            self._eps_auto_coef = float(epsilon_low[4:] or 0.5)
        elif epsilon_up is not None:
            assert epsilon_low <= epsilon_up

        self.Ops = operator_mapping
        self.nn_interface = nn_interface
        self.TrustAssessment = trust_assessment_func
        self.training_mode = False
        self.learning_rate = learning_rate
        self.epsilon_low = epsilon_low
        self.epsilon_up = epsilon_up
        self.structure = structure if structure is not None else []
        self.nntype = nntype
        self.eval = eval
        self.patch = patch
        self.batch_size = None
        self.no_round = no_round
        self.fuse_method = fuse_method

        # NEW: tensor mode
        self.use_tensor = use_tensor
        self.tensor_dtype = tensor_dtype
        # torch device for the tensor path (None when torch is unavailable)
        if device is not None and torch is not None:
            self.device = torch.device(device)
        else:
            self.device = default_device()

        # Keep original around if you want (optional)
        self.omega_thetas_obj = omega_thetas

        # Will hold either ArrayTO objects or TensorArrayTO depending on mode
        self.omega_thetas = omega_thetas

        # History container
        # In tensor mode we store TensorArrayTO objects; in object mode keep existing type.
        self.Typrime_layers_history = None

        if self.eval:
            if self.patch:
                self.EVAL = {"trust": [], "untrust": [], "distrust": [], "patch_tr": [], "patch_vac": []}
                self.EVAL_HIDDEN = {"trust": [], "untrust": [], "distrust": [], "patch_tr": [], "patch_vac": []}
            else:
                self.EVAL = {"trust": [], "untrust": [], "distrust": []}
                self.EVAL_HIDDEN = {"trust": [], "untrust": [], "distrust": []}

        if not self.use_tensor:
            # old behavior
            return

        # Convert omega_thetas: list[ArrayTO | TensorArrayTO | ndarray]
        # -> list[TensorArrayTO] on self.device
        self.omega_thetas = [self._to_tensor_opinions(layer_w) for layer_w in omega_thetas]

    def _op_matrix_to_tensor(self, op_mat: np.ndarray) -> np.ndarray:
        """
        Convert np.ndarray dtype=TrustOpinion with shape (n,m) into float tensor (n,m,3).
        One-time conversion so a Python loop here is OK.
        """
        n, m = op_mat.shape
        out = np.empty((n, m, 3), dtype=self.tensor_dtype)

        # flatten once to speed attribute extraction
        flat = op_mat.ravel()
        # Extract fields (still Python-level but done once)
        t = np.fromiter((o.t for o in flat), dtype=self.tensor_dtype, count=flat.size)
        d = np.fromiter((o.d for o in flat), dtype=self.tensor_dtype, count=flat.size)
        u = np.fromiter((o.u for o in flat), dtype=self.tensor_dtype, count=flat.size)

        out[..., 0] = t.reshape(n, m)
        out[..., 1] = d.reshape(n, m)
        out[..., 2] = u.reshape(n, m)
        return out

    def _to_tensor_opinions(self, T) -> TensorArrayTO:
        """
        Normalize any opinion container (ArrayTO of TrustOpinion objects,
        TensorArrayTO, raw ndarray/tensor of shape (...,3)) into a
        TensorArrayTO whose value lives on self.device.
        """
        v = T.value if hasattr(T, "value") else T
        fuse_method = getattr(T, "fuse_method", None)
        if _is_torch(v) or (isinstance(v, np.ndarray) and v.ndim >= 2
                            and v.shape[-1] == 3 and v.dtype != object):
            tens = v
        else:
            # 2D object matrix of TrustOpinion
            tens = self._op_matrix_to_tensor(np.asarray(v))
        out = TensorArrayTO(as_tensor(tens, device=self.device))
        if fuse_method is not None:
            out.fuse_method = fuse_method
        return out

    def run(self):
        """
        Listens for incoming connections and processes them.
        """
        port = self.nn_interface.port_number
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(('127.0.0.1', port))
            server_socket.listen()
            if(DEBUG>=0):
                print(f"Listening on port {port}...")
                print()

            while True:
                client_socket, address = server_socket.accept()
                with client_socket:
                    while True:
                        data = client_socket.recv(CHUNK_SIZE)
                        if not data:
                            break
                        message_obj = pickle.loads(data)
                        self.process_data(message_obj)

    def run_chunk(self, host='127.0.0.1', chunk_size=CHUNK_SIZE, ready_event=None):
        port = self.nn_interface.port_number
        # Establish a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            if ready_event is not None:
                ready_event.set()
            if(DEBUG>=0):
                print("[CHUNK] Waiting for a connection...")
                print()

            while True:
                conn, addr = s.accept()
                with conn:
                    if(DEBUG>=2):
                        print(f"Connection from {addr}")
                        print()

                    while True:
                        # Receive the total length of the next message.
                        header = self._recv_exact(conn, 4, allow_eof=True)
                        if header is None:
                            break
                        total_data_length = int.from_bytes(header, 'big')

                        # Receive and deserialize the message payload.
                        received_data = self._recv_exact(conn, total_data_length, chunk_size=chunk_size)
                        data = pickle.loads(received_data)
                        if(DEBUG>=2):
                            print("Data received and unpickled:", data)
                            print()
                        end = self.process_data(data)
                        ack_message = pickle.dumps("ACK")
                        conn.sendall(ack_message)
                        if(end):
                            return

    @staticmethod
    def _recv_exact(conn, expected_size: int, chunk_size=CHUNK_SIZE, allow_eof=False):
        """Read exactly expected_size bytes from a socket."""
        buffer = bytearray(expected_size)
        view = memoryview(buffer)
        received = 0
        while received < expected_size:
            to_read = min(chunk_size, expected_size - received)
            nbytes = conn.recv_into(view[received:received + to_read], to_read)
            if nbytes == 0:
                if allow_eof and received == 0:
                    return None
                raise ConnectionError("Socket closed before receiving expected number of bytes")
            received += nbytes
        return buffer




    def process_data(self, message_obj: MessageObject, client_socket=None):
        """
        Process the received message based on its mode.
        """
        if(DEBUG>=1):
            print("Message Being Processed")
            print(message_obj.easy_print())
            print()
        if message_obj.mode == Mode.END:
            return True
        if message_obj.mode == Mode.INFERENCE:
            # Stop training mode
            self.stop_training()
            # Load Computational Path
            inference_path = message_obj.content['inference_path']
            # Generate CPTA Corresponding to that Inference
            tempIPTA = self.GenIPTA(inference_path)
            # Compute Trust in X
            Tx = self.TrustAssessment(message_obj.content['X'], dim = self.omega_thetas[0].get_shape()[0] - 1)
            # Compute Trust in y
            Ty = tempIPTA(Tx)
            if(DEBUG>=1):
                print(Ty)
                print()

        elif message_obj.mode == Mode.TRAINING:
            # Verify that both structure of PTAS and NN matches
            assert self.structure == message_obj.content['structure'], f"PTAS structure {self.structure} does not match NN structure {message_obj.content['structure']} in training message."
            
            self.batch_size = message_obj.content['batch_size']
            # Start training mode
            self.start_training(message_obj.content['total_rounds'])
        elif self.training_mode:
            if message_obj.mode == Mode.TRAINING_FEEDFORWARD:
                # If in training mode and receive feedforward data, apply feedforward
                Tx = self.TrustAssessment(message_obj.content['X'], dim = self.omega_thetas[0].get_shape()[0] - 1)
                self.apply_feedforward(Tx)
                if(self.eval):
                    dim0 = self.omega_thetas[0].get_shape()[0] - 1
                    Txtrust = TensorArrayTO(tfill((1, dim0), method="trust", device=self.device))
                    Txuntrust = TensorArrayTO(tfill((1, dim0), method="vacuous", device=self.device))
                    Txdistrust = TensorArrayTO(tfill((1, dim0), method="distrust", device=self.device))
                    ytrust = self.apply_feedforward(Txtrust, tmp=False)
                    self.EVAL["trust"].append(ytrust)
                    self.EVAL_HIDDEN["trust"].append(self.Typrime_layers_history[1])

                    yuntrust= self.apply_feedforward(Txuntrust, tmp=False)
                    self.EVAL["untrust"].append(yuntrust)
                    self.EVAL_HIDDEN["untrust"].append(self.Typrime_layers_history[1])

                    ydistrust= self.apply_feedforward(Txdistrust, tmp=False)
                    self.EVAL["distrust"].append(ydistrust)
                    self.EVAL_HIDDEN["distrust"].append(self.Typrime_layers_history[1])

                    if(self.patch):
                        img_h_l = int(np.sqrt(dim0))
                        for base_method, eval_key in (("trust", "patch_tr"), ("vacuous", "patch_vac")):
                            Txpatch = TensorArrayTO(tfill((1, dim0), method=base_method, device=self.device))
                            v = Txpatch.value
                            for i in range(self.patch):
                                for j in range(self.patch):
                                    v[0, img_h_l*i+j, 0] = 0.0   # distrust the patch pixels
                                    v[0, img_h_l*i+j, 1] = 1.0
                                    v[0, img_h_l*i+j, 2] = 0.0
                            ypatch = self.apply_feedforward(Txpatch, tmp=False)
                            self.EVAL[eval_key].append(ypatch)


                    if(DEBUG>=2):
                        print("trust")
                        print(self.EVAL_HIDDEN["trust"][0])
                        print(self.EVAL["trust"][0])
                        print("Untrust")
                        print(self.EVAL["untrust"][0])
                        print("Distrust")
                        print(self.EVAL["distrust"][0])
                # self.Typrime_layers_history[-1] = self.aggregation(self.Typrime_layers_history[-1])
            elif message_obj.mode == Mode.TRAINING_BACKPROPAGATION:
                # Generalised: works for any number of hidden layers.
                # layer index == len(omega_thetas)-1  → output weights (last layer)
                # layer index  < len(omega_thetas)-1  → hidden weights
                deltaW = message_obj.content['delta_W']
                deltab = message_obj.content['delta_b']
                layer  = message_obj.layer
                n_layers = len(self.omega_thetas)

                Tybatch = self._to_tensor_opinions(
                    self.TrustAssessment(message_obj.content['y_true'], dim=self._output_dim())
                )
                Ty_fused = Tybatch.fuse_batch()
                initial_y0 = opinion_to_tensor(get_first_opinion(Ty_fused), self.tensor_dtype)

                y_prime = PTAS.aggregation(self.Typrime_layers_history[layer + 1])

                if layer == n_layers - 1:
                    # Last weight layer: pass the full per-class fused y-trust so that
                    # each output neuron gets its own class-specific trust revision.
                    # Using a scalar (class-0 only) here caused all output neurons to
                    # receive identical updates, losing the per-class distrust signal.
                    y_batch = Ty_fused
                    ty_revision = Ty_fused.value[:, 0, :]   # (output_dim, 3)
                else:
                    # Hidden weight layers: use next layer's tensor trust
                    y_batch = self.Typrime_layers_history[layer + 1]
                    ty_revision = initial_y0

                if DEBUG >= 2:
                    print(f"weights before (layer {layer})")
                    for i, ot in enumerate(self.omega_thetas):
                        print(f"  omega_thetas[{i}]:", ot)

                self.apply_trust_revision(
                    [deltaW, deltab],
                    layer,
                    y_prime,
                    y_batch,
                    self.learning_rate,
                    ty_revision
                )

                if DEBUG >= 2:
                    print(f"weights after (layer {layer})")
                    for i, ot in enumerate(self.omega_thetas):
                        print(f"  omega_thetas[{i}]:", ot)

                self.pbar.update(1)

                # Comment this part for full training
                if self.no_round is not None and message_obj.batch >= self.no_round:
                    return True

        return False

    def GenIPTA(self, inference_path: list):
        """
        Generate a subPTAS based on the computational path.

        Works in tensor mode (self.omega_thetas are TensorArrayTO).
        Supports any number of hidden layers.

        inference_path layout
        ---------------------
        1 hidden layer : inference_path = [[0,1,...]]          (batch-wrapped activations)
        2 hidden layers: inference_path = [[[0,1,...]], [[1,0,...]]]
        """
        print("Generating IPTA Function")
        print()

        n_weight_layers = len(self.omega_thetas)   # = n_hidden_layers + 1
        n_hidden = n_weight_layers - 1

        # --- Normalise: strip the batch dimension from each hidden layer's activations ---
        # forward() wraps each activation vector in a batch list, so:
        #   1 hidden layer:  inference_path = [[0,1]]          → take [0] → [0,1]
        #   ≥2 hidden layers: inference_path = [[[0,1]],[[1,0]]]→ take [0] per element
        if n_hidden == 1:
            activations = [inference_path[0]]          # strip batch dim of the single layer
        else:
            activations = [acts[0] for acts in inference_path[:n_hidden]]  # strip per layer

        # Validate sizes
        for i, acts in enumerate(activations):
            expected = self.omega_thetas[i + 1].get_shape()[0] - 1  # rows of next omega minus bias
            assert len(acts) == expected, (
                f"Hidden layer {i}: activation length {len(acts)} != expected {expected}"
            )

        # --- Build pruned weight matrices ---
        new_omegas = []
        for i in range(n_weight_layers):
            W = self.omega_thetas[i].value   # (rows, cols, 3)

            # Row selection: keep activated neurons from previous hidden layer + bias row
            if i == 0:
                row_idx = None   # keep all input rows
            else:
                acts_prev = list(activations[i - 1]) + [1]   # append bias flag
                row_idx = np.where(np.array(acts_prev, dtype=bool))[0]

            # Column selection: keep activated neurons in the next hidden layer
            if i < n_hidden:
                acts_next = list(activations[i]) + [1]        # append bias placeholder
                col_mask = np.array(acts_next, dtype=bool)
                col_idx = np.where(col_mask)[0][:-1]          # drop the bias placeholder
            else:
                col_idx = None   # keep all output columns

            if row_idx is not None:
                idx = torch.as_tensor(row_idx, device=W.device) if _is_torch(W) else row_idx
                W = W[idx, :, :]
            if col_idx is not None:
                idx = torch.as_tensor(col_idx, device=W.device) if _is_torch(W) else col_idx
                W = W[:, idx, :]

            new_omegas.append(TensorArrayTO(W))

        iptaPtas = PTAS(
            new_omegas,
            operator_mapping=self.Ops,
            nn_interface=None,
            trust_assessment_func=self.TrustAssessment,
            structure=self.structure,
            epsilon_low=self.epsilon_low,
            epsilon_up=self.epsilon_up,
            learning_rate=self.learning_rate,
            nntype=self.nntype,
            eval=False,
            patch=None,
            use_tensor=True,
            tensor_dtype=self.tensor_dtype,
            device=self.device,
            fuse_method=self.fuse_method,
        )

        def IPTA(Tx):
            print("Running IPTA Function")
            print()
            return iptaPtas.apply_feedforward(Tx)

        return IPTA
    
    def _output_dim(self):
        """Output dimension used to size the y trust assessment.

        For MLP structures ([in, h1, ..., out]) this is the last entry.
        Conv subclasses override this because their structure holds layer
        descriptors rather than plain integers.
        """
        return self.structure[-1]

    def _tx_for_layer(self, layer):
        """Input-trust container used by update_2 for a given weight layer.

        The default (MLP) case reads the feedforward history: the input of
        layer i is the output of layer i-1.  Conv subclasses override this
        because the layer input is a tiled version of the previous output.
        """
        return self.Typrime_layers_history[layer]

    def start_training(self, total_rounds=None):
        """
        Set the PTAS in training mode.
        """
        self.training_mode = True
        if self.no_round is not None:
            total_rounds = self.no_round

        self.pbar = tqdm(
                total= total_rounds,
                desc="Training PTAS", 
                unit="round",
                ncols=100
            )

        if(DEBUG>=1):
            print("PTAS is now in training mode.")
            print()

    def stop_training(self):
        """
        Stop the training mode and revert to inference.
        """
        self.pbar.close()
        self.training_mode = False
        if(DEBUG>=1):
            print("PTAS has exited training mode.")
            print()

    def apply_feedforward(self, Tx, tmp=True):
        """
        Tensor-accelerated feedforward.
        - Tx is expected to be ArrayTO (dtype=TrustOpinion) coming from TrustAssessment,
        OR already a TensorArrayTO if you migrated TrustAssessment too.
        """
        if DEBUG >= 1:
            print("Applying feedforward function (tensor mode)...")
            print()
            deb = time.time()

        if not self.use_tensor:
            # fallback – generalised loop over all weight layers
            one = TrustOpinion.fill(shape=(Tx.value.shape[0], 1), method="one")
            layer_outputs = [Tx]
            current = Tx
            for omega in self.omega_thetas:
                X_with_bias = ArrayTO(np.c_[current.value, one])
                current = ArrayTO.dot(X_with_bias, omega)
                layer_outputs.append(current)
            Ty_final = current
            if tmp:
                if self.batch_size == 1:
                    # transpose all layers except the final output
                    for lo in layer_outputs[:-1]:
                        lo.value = lo.value.T
                    self.Typrime_layers_history = layer_outputs
                else:
                    self.Typrime_layers_history = [lo.fuse_batch() for lo in layer_outputs]
            if DEBUG >= 1:
                print("End Applying feedforward function...")
                print(f"{time.time() - deb}s")
                print()
            return Ty_final

        # -------- tensor path --------
        # 1) Ensure Tx is a TensorArrayTO with shape (batch, dim, 3) on self.device
        Tx_t = self._to_tensor_opinions(Tx)

        # 2) Bias opinion column: preserve your current semantics.
        # NOTE: In your TrustOpinion.fill(), method="one" maps to vacuous (0,0,1). (Preserved!)
        one = tfill((Tx_t.value.shape[0], 1), method="one", dtype=self.tensor_dtype,
                    device=self.device)  # (batch,1,3)

        # 3) Loop through all weight layers (supports any number of hidden layers)
        layer_outputs = [Tx_t]   # history[0] = input trust
        current = Tx_t
        for omega in self.omega_thetas:
            if _is_torch(current.value):
                X_with_bias = TensorArrayTO(torch.cat([current.value, one], dim=1))
            else:
                X_with_bias = TensorArrayTO(np.concatenate([current.value, one], axis=1))
            current = TensorArrayTO.dot(X_with_bias, omega)
            layer_outputs.append(current)
        Ty_final = current  # output trust

        # 4) Store history (tensor mode)
        if tmp:
            if self.batch_size == 1:
                # Transpose all layers except the final output to column-vector form
                history = []
                for i, lo in enumerate(layer_outputs):
                    if i < len(layer_outputs) - 1:
                        history.append(TensorArrayTO(_t102(lo.value)))
                    else:
                        history.append(lo)
                self.Typrime_layers_history = history
            else:
                self.Typrime_layers_history = [lo.fuse_batch() for lo in layer_outputs]

        if DEBUG >= 1:
            print("End Applying feedforward function (tensor mode)...")
            print(f"{time.time() - deb}s")
            print()

        return Ty_final
    @staticmethod
    def aggregation(Tys):
        """
        Tensor aggregation:
        - input TensorArrayTO with value shape (n,1,3) or (batch,n,3) etc.
        - output tensor opinion shape (3,)
        Equivalent to TrustOpinion.avFuseGen over all elements.
        Works on numpy and torch values.
        """
        v = Tys.value
        # mean over all but last axis
        mean_op = v.reshape(-1, 3).mean(0)
        return normalize_tensor(mean_op)

    def depth_normalized_aggregation(self, Tys):
        """
        Aggregate like aggregation(), then undo the geometric depth decay
        so scores are comparable across architectures of different depth:
        the committed mass (b + d) is taken to the power 1/L with the b:d
        ratio preserved, where L = len(self.omega_thetas) is the number of
        trust-propagation steps ("trust retention per layer").
        """
        agg = self.aggregation(Tys)
        return depth_normalize_vec(agg, len(self.omega_thetas))

    def _layer_eps(self, layer, delta):
        """Effective epsilon_low for one layer's revision. Numeric epsilon
        is returned unchanged. In scale-tied mode ("auto<c>") the threshold
        is c * median(|delta|), with the median estimated over the first
        _eps_freeze_n revision batches of that layer and frozen afterwards,
        so the stability criterion adapts to each task's and layer's own
        gradient scale instead of an absolute magnitude."""
        if self._eps_auto_coef is None:
            return self.epsilon_low
        frozen = self._eps_layer.get(layer)
        if frozen is not None:
            return frozen
        med = float(np.median(np.abs(to_numpy(delta))))
        hist = self._eps_hist.setdefault(layer, [])
        hist.append(med)
        eps = self._eps_auto_coef * float(np.median(hist))
        if len(hist) >= self._eps_freeze_n:
            self._eps_layer[layer] = eps
            print(f"[PaTAS] layer {layer}: scale-tied eps frozen at "
                  f"{eps:.4g} (c={self._eps_auto_coef:g}, "
                  f"median|delta|={eps / self._eps_auto_coef:.4g}, "
                  f"n={len(hist)} batches)")
        return eps

    def apply_trust_revision(
        self,
        data: list,
        layer: int,
        y_prime,                      # Tensor opinion (3,) from aggregation
        y_batch_all_opinion,          # TensorArrayTO (k,1,3) or scalar (3,)
        learning_rate,                # TrustOpinion or tensor (3,)
        initial_y_batch_single_opinion
    ):
        """
        Tensor trust revision:
        Removes ArrayTO.theta_given_y/op_theta_y/op_theta/update_2 loops. 
        """
        if DEBUG >= 1:
            print(f"Applying trust revision (tensor) for layer {layer}")

        from concrete.TensorTO import (
            theta_given_y,
            theta_given_not_y,
            fast_deduction,
            op_theta,
            update,
            update_2,
        )

        # Convert learning_rate to tensor (3,) on self.device
        if isinstance(learning_rate, TrustOpinion):
            lr = np.array([learning_rate.t, learning_rate.d, learning_rate.u], dtype=self.tensor_dtype)
            lr = normalize_tensor(lr)
        else:
            lr = learning_rate
        lr = as_tensor(lr, device=self.device)

        # Convert initial_y_batch_single_opinion to tensor.
        # For the output layer it is (output_dim, 3) — per-class opinions.
        # For hidden layers it is (3,) — a scalar opinion.
        if hasattr(initial_y_batch_single_opinion, "ndim") and initial_y_batch_single_opinion.ndim == 2:
            Ty0 = normalize_tensor(as_tensor(initial_y_batch_single_opinion, device=self.device))
        else:
            Ty0 = opinion_to_tensor(initial_y_batch_single_opinion, self.tensor_dtype)
            Ty0 = normalize_tensor(as_tensor(Ty0, device=self.device))
        if len(data) != 2:
            raise NotImplementedError

        # deltas arrive as numpy float32 over the socket; move them to device
        deltaW = to_numpy(data[0])
        deltab = to_numpy(data[1])
        delta = as_tensor(np.concatenate((deltaW, deltab), axis=0), device=self.device)   # (in+1, out)

        # 1) vectorized theta evidence from delta
        op_tgy = theta_given_y(delta, self._layer_eps(layer, delta),
                               dtype=self.tensor_dtype)                                   # (1,out,3)
        op_tgny = theta_given_not_y(delta, None, dtype=self.tensor_dtype)                 # (1,out,3) vacuous

        # 2) Build y input for deduction
        # - output layer: y_batch_all_opinion is the fused per-class y trust
        # - hidden layers: y_batch_all_opinion is next layer's trust (k,1,3)
        if isinstance(y_batch_all_opinion, TensorArrayTO):
            y_in = as_tensor(y_batch_all_opinion.value, device=self.device)   # e.g., (k,1,3)
            # Deduction expects op_x shaped like (1,k,3) to match op_tgy (1,k,3)
            y_in_row = _t102(y_in)  # (1,k,3)
            op_theta_y = fast_deduction(y_in_row, op_tgy, op_tgny)  # (1,k,3)
        else:
            # scalar (3,) -> broadcast to (1,out,3)
            y_scalar = y_batch_all_opinion
            if hasattr(y_scalar, "t") and isinstance(y_scalar, TrustOpinion):
                y_scalar = np.array([y_scalar.t, y_scalar.d, y_scalar.u], dtype=self.tensor_dtype)
                y_scalar = normalize_tensor(y_scalar)
            y_scalar = as_tensor(y_scalar, device=self.device)
            y_row = y_scalar[None, None, :]          # (1,1,3)
            if _is_torch(y_row):
                y_row = torch.broadcast_to(y_row, op_tgy.shape)  # (1,out,3)
            else:
                y_row = np.broadcast_to(y_row, op_tgy.shape)
            op_theta_y = fast_deduction(y_row, op_tgy, op_tgny)

        # 3) op_theta: fuse weight columns with evidence
        W = as_tensor(self.omega_thetas[layer].value, device=self.device)           # (n,out,3)
        W_new = op_theta(W, op_theta_y, method=self.fuse_method)   # (n,out,3)

        # 4) update step (binMult + fuse via self.fuse_method)
        W_new = update(W_new, W_new, lr, method=self.fuse_method)  # (n,out,3)

        # 5) update_2: extra binMult constraints using Tx and initial y opinion
        Tx_layer = as_tensor(self._tx_for_layer(layer).value, device=self.device)  # (ni,1,3)
        W_new = update_2(W_new, Tx_layer, Ty0)               # (ni1,out,3)

        # store back
        self.omega_thetas[layer] = TensorArrayTO(W_new)

    #     axs[0,0].set_title('Evolution of trust mass for fully trusted input')
    #     axs[0,1].set_title('Evolution of uncertainty mass for fully trusted input')
    #     axs[0,2].set_title('Evolution of distrust mass for fully trusted input')


    #     axs[1,0].set_title('Evolution of trust mass for vacuous input')
    #     axs[1,1].set_title('Evolution of uncertainty mass for vacuous input')
    #     axs[1,2].set_title('Evolution of distrust mass for vacuous input')

    #     axs[2,0].set_title('Evolution of trust mass for fully distrusted input')
    #     axs[2,1].set_title('Evolution of uncertainty mass for fully distrusted input')
    #     axs[2,2].set_title('Evolution of distrust mass for fully distrusted input')

    #     handles, labels = axs[0,0].get_legend_handles_labels()

    @staticmethod
    def _eval_values(entries):
        """Convert a list of TensorArrayTO (numpy or torch) to numpy arrays for plotting."""
        return [to_numpy(e.value) if hasattr(e, "value") else to_numpy(e) for e in entries]

    def eval_plot(EVAL, output_size, title, fname, n_epoch, patch=None):
        eval_trust = PTAS._eval_values(EVAL["trust"])
        eval_untrust = PTAS._eval_values(EVAL["untrust"])
        eval_distrust = PTAS._eval_values(EVAL["distrust"])

        num_rounds = len(eval_trust)
        rounds_per_epoch = num_rounds // n_epoch
        epoch_boundaries = [(i + 1) * rounds_per_epoch for i in range(n_epoch - 1)]
        if(patch):
            n_row = 5
            eval_patch_tr = PTAS._eval_values(EVAL["patch_tr"])
            eval_patch_vac = PTAS._eval_values(EVAL["patch_vac"])
            fig, axs = plt.subplots(n_row, 3, figsize=(20, 20))
        else:
            n_row = 3
            fig, axs = plt.subplots(n_row, 3, figsize=(20, 11))

        def plot_epoch_lines(ax):
            for boundary in epoch_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)

        for digit_label in range(output_size):
            axs[0,0].plot(
                range(1, num_rounds + 1),
                [eval_trust[i][0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[0,1].plot(
                range(1, num_rounds + 1),
                [eval_trust[i][0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[0,2].plot(
                range(1, num_rounds + 1),
                [eval_trust[i][0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[0,0].set_title('Evolution of trust mass for fully trusted input')
        axs[0,1].set_title('Evolution of uncertainty mass for fully trusted input')
        axs[0,2].set_title('Evolution of distrust mass for fully trusted input')

        for digit_label in range(output_size):
            axs[1,0].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i][0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
            axs[1,1].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i][0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
            axs[1,2].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i][0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[1,0].set_title('Evolution of trust mass for vacuous input')
        axs[1,1].set_title('Evolution of uncertainty mass for vacuous input')
        axs[1,2].set_title('Evolution of distrust mass for vacuous input')

        for digit_label in range(output_size):
            axs[2,0].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i][0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[2,1].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i][0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[2,2].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i][0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[2,0].set_title('Evolution of trust mass for fully distrusted input')
        axs[2,1].set_title('Evolution of uncertainty mass for fully distrusted input')
        axs[2,2].set_title('Evolution of distrust mass for fully distrusted input')

        if(patch):
            for digit_label in range(output_size):
                axs[3,0].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0, digit_label, 0] for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,1].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0, digit_label, 2] for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,2].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0, digit_label, 1] for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[3,0].set_title('Evolution of trust mass for patch trusted input')
            axs[3,1].set_title('Evolution of uncertainty mass for patch trusted input')
            axs[3,2].set_title('Evolution of distrust mass for patch trusted input')

            for digit_label in range(output_size):
                axs[4,0].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0, digit_label, 0] for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,1].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0, digit_label, 2] for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,2].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0, digit_label, 1] for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[4,0].set_title('Evolution of trust mass for patch vacuous input')
            axs[4,1].set_title('Evolution of uncertainty mass for patch vacuous input')
            axs[4,2].set_title('Evolution of distrust mass for patch vacuous input')

        # Add vertical lines to mark epochs
        if(n_epoch>1):
            for i in range(n_row):
                for j in range(3):
                    plot_epoch_lines(axs[i,j])

        # Add legend and title

        epoch_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='Epoch boundary')
        # Add all label handles + custom epoch line to legend
        handles, labels = axs[0,0].get_legend_handles_labels()
        handles.append(epoch_line)
        labels.append("Epoch boundary")
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=output_size)
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(fname)

    def eval_plot_simpl(EVAL, output_size, title, fname):
        eval_trust = PTAS._eval_values(EVAL["trust"])
        eval_untrust = PTAS._eval_values(EVAL["untrust"])
        eval_distrust = PTAS._eval_values(EVAL["distrust"])
        timesteps = np.arange(1, len(eval_trust) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(20, 11))
        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_trust)):
            t_vals = [eval_trust[i][0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_trust[i][0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_trust[i][0, digit_label, 2] for digit_label in range(output_size)]

            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))

        axs[0].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[0].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[0].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[0].set_title('fully trusted input')
        axs[0].legend()

        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_untrust)):
            t_vals = [eval_untrust[i][0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_untrust[i][0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_untrust[i][0, digit_label, 2] for digit_label in range(output_size)]

            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))

        axs[1].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[1].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[1].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[1].set_title('vacuous input')
        axs[1].legend()

        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_untrust)):
            t_vals = [eval_distrust[i][0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_distrust[i][0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_distrust[i][0, digit_label, 2] for digit_label in range(output_size)]

            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))

        axs[2].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[2].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[2].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[2].set_title('fully distrusted input')
        axs[2].legend()

        plt.savefig(fname, dpi=300, bbox_inches='tight')


# input_dim = 30
# hidden_dim = 16
# output_dim = 2


def T_NOT_poisoned(x: np.array, dim):
    op = "trust"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T_poisoned(x: np.array, dim):
    op = "trust"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T1(x: np.array, dim):
    op = "vacuous"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
def Ttrust(x: np.array, dim):
    op = "trust"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tdistrust(x: np.array, dim):
    op = "distrust"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tvacuous(x: np.array, dim):
    op = "vacuous"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T2(x: np.array, dim):

    op = "vacuous"
    n = len(x)
    res = np.empty((n, dim))
    for i in range(n):
        for j in range(dim):
            res[i][j] = TrustOpinion.ftrust()

def Tgen(xx, yy):
    assert xx[0]=='x'
    assert yy[0]=='y'
    def inner_function(x: np.array, dim):
        if(dim==input_dim):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function


XX = ["xtrust", 'xvacuous', "xdistrust"]
YY = ["ytrust", 'yvacuous', "ydistrust"]


def run_uni_test(xx, yy, epsilon_low, epsilon_up):
    assert xx in XX
    assert yy in YY
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = Tgen(xx, yy)
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                structure = [input_dim, hidden_dim, output_dim],
                epsilon_low=epsilon_low, epsilon_up=epsilon_up)

    datapath = folder_path+'NN\\eval\\'+str(epsilon_low)+"-"+str(epsilon_up)+"\\"+xx+yy
    os.mkdir(datapath)
    ptas.run_chunk()


    writeto(ptas.omega_thetas, datapath+"\\omegas.pkl")
    at = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    writeto(at, datapath+"\\at.pkl")
    av = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    writeto(av, datapath+"\\av.pkl")
    ad = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    writeto(ad, datapath+"\\ad.pkl")


def test():
    epsilon_low = 0.0001
    epsilon_up = 0.001
    datapath = folder_path+'NN\\eval\\'+str(epsilon_low)+"-"+str(epsilon_up)
    os.mkdir(datapath)
    for xx in XX:
        for yy in YY:
            print(f"Running Test for {xx}, {yy}")
            print()
            run_uni_test(xx, yy, epsilon_low, epsilon_up)
def main_cancer():
    input_dim = 30
    hidden_dim = 16
    output_dim = 2
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tdistrust, structure = [input_dim, hidden_dim, output_dim], epsilon_low=0.03)
    datapath = folder_path+'NN\\eval_cancer'
    os.mkdir(datapath)

    ptas.run_chunk()
    print("--------------------------- 0 0 0 ----------------------")
    print(ptas.omega_thetas[0].get_shape())
    print(ptas.omega_thetas[0])
    print("-------------------------------------------------")
    print()

    print("--------------------------- 1 1 1 ----------------------")
    print(ptas.omega_thetas[1].get_shape())
    print(ptas.omega_thetas[1])
    print("-------------------------------------------------")
    print()

    print("Apply Feed Forward on fully Trusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\at.pkl")

    print("Apply Feed Forward on Vacuous Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\av.pkl")

    print("Apply Feed Forward on fully Untrusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\ad.pkl")


def tryy():
    xx = "xtrust"
    yy = "ytrust"
    datapath = folder_path+'NN\\eval\\'+xx+yy
    with open(datapath+"\\omegas.pkl", "rb") as file:  # Use 'rb' for reading binary
        loaded_data = pickle.load(file)
    print("Loaded data:", loaded_data[0])


if __name__ == "__main__":
    print("nothing to run")
