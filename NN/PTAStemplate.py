"""PTAS runtime implementation used by the listener-side trust assessment process."""

import os
import sys

from concrete.TensorTO import normalize_tensor

sys.path.append(os.getcwd())
folder_path = os.getcwd()
import socket
import pickle
from PTASTemp.ptasInterface import PTASInterface
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import numpy as np
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
    Convert TrustOpinion or array-like into tensor (3,)
    """
    if isinstance(op, TrustOpinion):
        return np.array([op.t, op.d, op.u], dtype=dtype)

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
):
        """
        Initialize the PTAS with essential components.

        This patched version can store omega_thetas as tensors (n,m,3) for fast ops.
        """
        if epsilon_up is not None:
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
        
        # NEW: tensor mode
        self.use_tensor = use_tensor
        self.tensor_dtype = tensor_dtype

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

        # -------- Tensor conversion helpers (one-time cost) --------
        from concrete.TensorTO import TensorArrayTO  # adjust import path if needed

        def _op_matrix_to_tensor(op_mat: np.ndarray) -> np.ndarray:
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

        # Convert omega_thetas: list[ArrayTO] -> list[TensorArrayTO]
        omega_tensors = []
        for layer_w in omega_thetas:
            # layer_w may be ArrayTO, TensorArrayTO, or raw ndarray
            op_mat = layer_w.value if hasattr(layer_w, "value") else layer_w

            # If already tensor (n,m,3), accept directly
            if isinstance(op_mat, np.ndarray) and op_mat.ndim == 3 and op_mat.shape[-1] == 3:
                omega_tensors.append(TensorArrayTO(op_mat.astype(self.tensor_dtype, copy=False)))
            else:
                # Otherwise assume 2D TrustOpinion matrix
                omega_tensors.append(TensorArrayTO(_op_matrix_to_tensor(op_mat)))

        self.omega_thetas = omega_tensors

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

    def run_chunk(self, host='127.0.0.1', chunk_size=CHUNK_SIZE):
        port = self.nn_interface.port_number
        # Establish a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
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
                    Txtrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="trust"))
                    Txuntrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
                    Txdistrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="distrust"))
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
                        img_h_l = int(np.sqrt(self.omega_thetas[0].get_shape()[0] - 1))
                        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="trust"))
                        for i in range(self.patch):
                            for j in range(self.patch):
                                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
                        ypatch = self.apply_feedforward(Txpatch, tmp=False)
                        self.EVAL["patch_tr"].append(ypatch)

                        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
                        for i in range(self.patch):
                            for j in range(self.patch):
                                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
                        ypatch = self.apply_feedforward(Txpatch, tmp=False)
                        self.EVAL["patch_vac"].append(ypatch)


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
                # If in training mode and receive backpropagation data,  apply trust revision on the layer specified
                #Load delta values
                deltaW = message_obj.content['delta_W'] #Weigths
                deltab = message_obj.content['delta_b'] #Bias
                Tybatch=  self.TrustAssessment(message_obj.content['y_true'], dim=self.structure[2])

                # after Tybatch computed:
                # tensor fused version for convenience
                Ty_fused = Tybatch.fuse_batch()                 # in tensor mode, this is (out,1,3)

                # scalar "first label" like your original: Tybatch.fuse_batch()[0][0]
                initial_y0 = opinion_to_tensor(get_first_opinion(Ty_fused), self.tensor_dtype)

                if message_obj.layer == 1:
                    # y_batch_all_opinion used as scalar in layer-1 deduction
                    # y_scalar = Ty_fused.value[0, 0, :]         # keep same semantics (first output)
                    # y_scalar = opinion_to_tensor(get_first_opinion(Ty_fused), self.tensor_dtype)         # more robust way to get the scalar opinion
                    if(DEBUG>=2):
                        print("weights before")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                    # self.apply_trust_revision(
                    #     [deltaW, deltab],
                    #     message_obj.layer,
                    #     PTAS.aggregation(self.Typrime_layers_history[message_obj.layer+1]),
                    #     y_scalar,
                    #     self.learning_rate,
                    #     initial_y0
                    # )

                    self.apply_trust_revision(
                        [deltaW, deltab],
                        message_obj.layer,
                        PTAS.aggregation(self.Typrime_layers_history[message_obj.layer+1]),
                        Ty_fused,
                        self.learning_rate,
                        initial_y0
                    )
                    
                    if(DEBUG>=2):
                        print("weights After")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])

                if message_obj.layer == 0:
                    # y_batch_all_opinion is the hidden layer trust (TensorArrayTO)
                    if(DEBUG>=2):
                        print("weights before")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])

                    self.apply_trust_revision(
                        [deltaW, deltab],
                        message_obj.layer,
                        PTAS.aggregation(self.Typrime_layers_history[message_obj.layer+1]),
                        self.Typrime_layers_history[message_obj.layer+1],
                        self.learning_rate,
                        initial_y0
                    )
                    if(DEBUG>=2):
                        print("weights After")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                self.pbar.update(1)
                    

                # Comment this part for full training
                if self.no_round is not None and message_obj.batch >= self.no_round:
                    return True

        return False

    def GenIPTA(self, inference_path: list):
        """
        Generate a subPTAS based on the computational path.

        Works in tensor mode (self.omega_thetas are TensorArrayTO).
        """
        print("Generating IPTA Function")
        print()

        from concrete.TensorTO import TensorArrayTO

        inference_path = inference_path[0]  # same as your original

        # last hidden bias neuron always included
        assert len(inference_path) == self.omega_thetas[1].get_shape()[0] - 1
        inference_path = list(inference_path) + [1]

        mask = np.array(inference_path, dtype=bool)      # length hidden+1
        idx = np.where(mask)[0]                          # selected hidden indices
        n = idx.size

        # omega0: (in+1, hidden, 3) -> keep selected hidden neurons excluding bias position at end of mask
        # Your original builds omega0 with shape (input+1, n-1) (excluding the final bias element).
        idx_no_bias = idx[:-1]                           # selected hidden neurons (no bias)
        W0 = self.omega_thetas[0].value[:, idx_no_bias, :]   # (in+1, n-1, 3)

        # omega1: (hidden+1, out, 3) -> keep selected hidden neurons + bias row
        W1 = self.omega_thetas[1].value[idx, :, :]          # (n, out, 3)

        new_omegas = [TensorArrayTO(W0), TensorArrayTO(W1)]

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
        )

        def IPTA(Tx):
            print("Running IPTA Function")
            print()
            return PTAS.aggregation(iptaPtas.apply_feedforward(Tx))

        return IPTA
    
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
            # fallback to your original implementation
            one = TrustOpinion.fill(shape=(Tx.value.shape[0], 1), method="one")
            X_with_bias = ArrayTO(np.c_[Tx.value, one])
            Ty1 = ArrayTO.dot(X_with_bias, self.omega_thetas[0])
            X_with_bias = ArrayTO(np.c_[Ty1.value, one])
            Ty2 = ArrayTO.dot(X_with_bias, self.omega_thetas[1])
            if tmp:
                if self.batch_size == 1:
                    Tx.value = Tx.value.T
                    Ty1.value = Ty1.value.T
                    self.Typrime_layers_history = [Tx, Ty1, Ty2]
                else:
                    self.Typrime_layers_history = [Tx.fuse_batch(), Ty1.fuse_batch(), Ty2.fuse_batch()]
            if DEBUG >= 1:
                print("End Applying feedforward function...")
                print(f"{time.time() - deb}s")
                print()
            return Ty2

        # -------- tensor path --------
        from concrete.TensorTO import TensorArrayTO, fill as tfill  # adjust import path if needed

        # 1) Ensure Tx is a TensorArrayTO with shape (batch, dim, 3)
        if hasattr(Tx, "value") and isinstance(Tx.value, np.ndarray) and Tx.value.ndim == 3 and Tx.value.shape[-1] == 3:
            # already tensor-like
            Tx_t = TensorArrayTO(Tx.value)
        elif Tx.__class__.__name__ == "TensorArrayTO":
            Tx_t = Tx
        else:
            # Tx is ArrayTO of TrustOpinion objects: shape (batch, dim)
            op_mat = Tx.value
            b, d = op_mat.shape
            tens = np.empty((b, d, 3), dtype=self.tensor_dtype)
            flat = op_mat.ravel()
            tens[..., 0] = np.fromiter((o.t for o in flat), dtype=self.tensor_dtype, count=flat.size).reshape(b, d)
            tens[..., 1] = np.fromiter((o.d for o in flat), dtype=self.tensor_dtype, count=flat.size).reshape(b, d)
            tens[..., 2] = np.fromiter((o.u for o in flat), dtype=self.tensor_dtype, count=flat.size).reshape(b, d)
            Tx_t = TensorArrayTO(tens)

        # 2) Bias opinion column: preserve your current semantics.
        # NOTE: In your TrustOpinion.fill(), method="one" maps to vacuous (0,0,1). (Preserved!)
        one = tfill((Tx_t.value.shape[0], 1), method="one", dtype=self.tensor_dtype)  # (batch,1,3)

        # 3) Hidden layer trust
        X_with_bias = TensorArrayTO(np.concatenate([Tx_t.value, one], axis=1))  # (batch, in+1, 3)
        Ty1 = TensorArrayTO.dot(X_with_bias, self.omega_thetas[0])              # (batch, hidden, 3)

        # 4) Output layer trust
        X_with_bias2 = TensorArrayTO(np.concatenate([Ty1.value, one], axis=1))  # (batch, hidden+1, 3)
        Ty2 = TensorArrayTO.dot(X_with_bias2, self.omega_thetas[1])             # (batch, out, 3)

        # 5) Store history (tensor mode)
        if tmp:
            if self.batch_size == 1:
                # Match your old behavior of storing column-vectors:
                # Tx: (1, in, 3) -> (in, 1, 3)
                # Ty1: (1, hidden,3)->(hidden,1,3)
                Tx_col = TensorArrayTO(Tx_t.value.transpose(1, 0, 2))
                Ty1_col = TensorArrayTO(Ty1.value.transpose(1, 0, 2))
                self.Typrime_layers_history = [Tx_col, Ty1_col, Ty2]
            else:
                self.Typrime_layers_history = [Tx_t.fuse_batch(), Ty1.fuse_batch(), Ty2.fuse_batch()]

        if DEBUG >= 1:
            print("End Applying feedforward function (tensor mode)...")
            print(f"{time.time() - deb}s")
            print()

        return Ty2
    @staticmethod
    def aggregation(Tys):
        """
        Tensor aggregation:
        - input TensorArrayTO with value shape (n,1,3) or (batch,n,3) etc.
        - output tensor opinion shape (3,)
        Equivalent to TrustOpinion.avFuseGen over all elements. :contentReference[oaicite:11]{index=11}
        """
        from concrete.TensorTO import normalize_tensor
        v = Tys.value
        # mean over all but last axis
        mean_op = np.mean(v, axis=tuple(range(v.ndim-1)))
        return normalize_tensor(mean_op)

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
            TensorArrayTO,
            theta_given_y,
            theta_given_not_y,
            fast_deduction,
            op_theta,
            update,
            update_2,
            fill as tfill,
            normalize_tensor
        )

        # Convert learning_rate to tensor (3,)
        if isinstance(learning_rate, TrustOpinion):
            lr = np.array([learning_rate.t, learning_rate.d, learning_rate.u], dtype=self.tensor_dtype)
            lr = normalize_tensor(lr)
        else:
            lr = learning_rate.astype(self.tensor_dtype, copy=False)

        # Convert initial_y_batch_single_opinion to tensor (3,)
        Ty0 = opinion_to_tensor(initial_y_batch_single_opinion, self.tensor_dtype)
        Ty0 = normalize_tensor(Ty0)
        if len(data) != 2:
            raise NotImplementedError

        deltaW = data[0]
        deltab = data[1]
        delta = np.concatenate((deltaW, deltab), axis=0)   # (in+1, out)

        # 1) vectorized theta evidence from delta
        op_tgy = theta_given_y(delta, self.epsilon_low, dtype=self.tensor_dtype)          # (1,out,3)
        op_tgny = theta_given_not_y(delta, None, dtype=self.tensor_dtype)                 # (1,out,3) vacuous :contentReference[oaicite:15]{index=15}

        # 2) Build y input for deduction
        # - for layer 1 in your code: y_batch_all_opinion is a fused scalar/array (output layer)
        # - for layer 0 in your code: y_batch_all_opinion is hidden layer trust (k,1,3)
        if isinstance(y_batch_all_opinion, TensorArrayTO):
            y_in = y_batch_all_opinion.value   # e.g., (k,1,3)
            # Deduction expects op_x shaped like (1,k,3) to match op_tgy (1,k,3)
            y_in_row = y_in.transpose(1, 0, 2)  # (1,k,3)
            op_theta_y = fast_deduction(y_in_row, op_tgy, op_tgny)  # (1,k,3)
        else:
            # scalar (3,) -> broadcast to (1,out,3)
            y_scalar = y_batch_all_opinion
            if hasattr(y_scalar, "t"):
                y_scalar = np.array([y_scalar.t, y_scalar.d, y_scalar.u], dtype=self.tensor_dtype)
                y_scalar = normalize_tensor(y_scalar)
            y_row = y_scalar[None, None, :]          # (1,1,3)
            y_row = np.broadcast_to(y_row, op_tgy.shape)  # (1,out,3)
            op_theta_y = fast_deduction(y_row, op_tgy, op_tgny)

        # 3) op_theta: fuse weight columns with evidence
        W = self.omega_thetas[layer].value.astype(self.tensor_dtype, copy=False)           # (n,out,3)
        W_new = op_theta(W, op_theta_y)              # (n,out,3)

        # 4) update step (binMult + avFuse)
        W_new = update(W_new, W_new, lr)             # (n,out,3)

        # 5) update_2: extra binMult constraints using Tx and initial y opinion
        Tx_layer = self.Typrime_layers_history[layer].value.astype(self.tensor_dtype, copy=False)  # (ni,1,3)
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

    def eval_plot(EVAL, output_size, title, fname, n_epoch, patch=None):
        eval_trust = EVAL["trust"]
        eval_untrust = EVAL["untrust"]
        eval_distrust = EVAL["distrust"]

        num_rounds = len(eval_trust)
        rounds_per_epoch = num_rounds // n_epoch
        epoch_boundaries = [(i + 1) * rounds_per_epoch for i in range(n_epoch - 1)]
        if(patch):
            n_row = 5
            eval_patch_tr = EVAL["patch_tr"]
            eval_patch_vac = EVAL["patch_vac"]
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
                [eval_trust[i].value[0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[0,1].plot(
                range(1, num_rounds + 1),
                [eval_trust[i].value[0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[0,2].plot(
                range(1, num_rounds + 1),
                [eval_trust[i].value[0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[0,0].set_title('Evolution of trust mass for fully trusted input')
        axs[0,1].set_title('Evolution of uncertainty mass for fully trusted input')
        axs[0,2].set_title('Evolution of distrust mass for fully trusted input')

        for digit_label in range(output_size):
            axs[1,0].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i].value[0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
            axs[1,1].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i].value[0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
            axs[1,2].plot(
                range(1, num_rounds + 1),
                [eval_untrust[i].value[0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[1,0].set_title('Evolution of trust mass for vacuous input')
        axs[1,1].set_title('Evolution of uncertainty mass for vacuous input')
        axs[1,2].set_title('Evolution of distrust mass for vacuous input')

        for digit_label in range(output_size):
            axs[2,0].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i].value[0, digit_label, 0] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[2,1].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i].value[0, digit_label, 2] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )

            axs[2,2].plot(
                range(1, num_rounds + 1),
                [eval_distrust[i].value[0, digit_label, 1] for i in range(num_rounds)],
                label=f"label = {digit_label}"
            )
        axs[2,0].set_title('Evolution of trust mass for fully distrusted input')
        axs[2,1].set_title('Evolution of uncertainty mass for fully distrusted input')
        axs[2,2].set_title('Evolution of distrust mass for fully distrusted input')

        if(patch):
            for digit_label in range(output_size):
                axs[3,0].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,1].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,2].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[3,0].set_title('Evolution of trust mass for patch trusted input')
            axs[3,1].set_title('Evolution of uncertainty mass for patch trusted input')
            axs[3,2].set_title('Evolution of distrust mass for patch trusted input')

            for digit_label in range(output_size):
                axs[4,0].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,1].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,2].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
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
        eval_trust = EVAL["trust"]
        eval_untrust = EVAL["untrust"]
        eval_distrust = EVAL["distrust"]
        timesteps = np.arange(1, len(eval_trust) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(20, 11))
        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_trust)):
            t_vals = [eval_trust[i].value[0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_trust[i].value[0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_trust[i].value[0, digit_label, 2] for digit_label in range(output_size)]

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
            t_vals = [eval_untrust[i].value[0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_untrust[i].value[0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_untrust[i].value[0, digit_label, 2] for digit_label in range(output_size)]

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
            t_vals = [eval_distrust[i].value[0, digit_label, 0] for digit_label in range(output_size)]
            u_vals = [eval_distrust[i].value[0, digit_label, 1] for digit_label in range(output_size)]
            d_vals = [eval_distrust[i].value[0, digit_label, 2] for digit_label in range(output_size)]

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
