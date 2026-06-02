# patas_module — Socket-based PTAS Implementation

This sub-package is the **full, socket-based implementation** of the Parallel Trust
Assessment System (PTAS) described in the dissertation (Chapters 5–7).

Unlike the standalone scripts at the repository root (which run everything in a single
process for transparency and reproducibility), `patas_module` implements PTAS as a
**server/client pair communicating over TCP sockets**: the Trust Node Network (PTAS
server) and the Neural Network (NN client) run as separate processes and exchange
gradients at each mini-batch, matching the distributed architecture of Definition 7.2.

The evaluation scripts at the root (`eval_chapter.py`, `eval_5g_noise.py`, etc.) connect
to this module via `external_bridge.py`.

---

## Quick start

```bash
# From the v2/ directory — run PTAS server and NN client in one command:
python -m patas_module --mode both --testcase cancer --xtrust trust --ytrust trust

# Or in two separate terminals:
python -m patas_module --mode server --testcase cancer --xtrust trust --ytrust trust
python -m patas_module --mode client --testcase cancer --xtrust trust --ytrust trust

# MNIST, vacuous trust (uncertain data):
python -m patas_module --mode both --testcase mnist --xtrust vacuous --ytrust vacuous

# MNIST with poisoning attack (patch size 4, classes 6 and 9 flipped):
python -m patas_module --mode both --testcase mnist --mnist-poisoned-soph --mnist-patch-size 4
```

Alternatively, if the package is installed (`pip install -e .` from `v2/`):

```bash
patas --mode both --testcase cancer --xtrust trust --ytrust trust
```

---

## CLI reference

| Argument | Values | Default | Description |
|---|---|---|---|
| `--mode` | `server` \| `client` \| `both` | **required** | Start PTAS only, NN only, or both (via `multiprocessing`) |
| `--testcase` | `cancer` \| `mnist` | `cancer` | Dataset and architecture preset |
| `--xtrust` | see below | `trust` | Trust opinion applied to input features |
| `--ytrust` | see below | `trust` | Trust opinion applied to labels |
| `--hidden-neurons` | int | 16 | Hidden layer size (overrides the testcase default) |
| `--port` | int | 5000 | TCP port for server–client communication |
| `--epsilon-low` | float | 0.01 | Lower gradient threshold ε (Alg. 6) |
| `--epsilon-up` | float | None | Upper gradient threshold (optional) |
| `--mnist-poisoned-soph` | flag | off | Enable poisoning-aware trust generator for MNIST |
| `--mnist-patch-size` | int | 4 | Patch size for the poisoning trigger |
| `--no-round` | int | None | Stop PTAS after N mini-batch rounds (testing only) |
| `--dry-run` | flag | off | Validate config and print it; do not start sockets |
| `--not-ptas` | flag | off | Disable PTAS mode — run NN standalone |

### Trust specifications (`--xtrust` / `--ytrust`)

| Spec | Meaning |
|---|---|
| `trust` | Fully trusted opinion (b=1, d=0, u=0) |
| `distrust` | Fully distrusted opinion (b=0, d=1, u=0) |
| `vacuous` | Vacuous / uncertain opinion (b=0, d=0, u=1) |
| `random` | Random valid opinion per sample |
| `t,d,u` | Custom fixed triplet, e.g. `0.5,0.3,0.2` |

---

## Testcase presets

| Testcase | Dataset | Architecture | Epochs | lr | ε |
|---|---|---|---|---|---|
| `cancer` | Breast Cancer (sklearn) | 30–16–2 | 15 | 0.2 | 0.1 |
| `mnist` | MNIST (keras/tensorflow) | 784–128–10 | 20 | 0.05 | 0.05 |

Results (trained weights, evaluation logs, plots) are saved under
`patas_module/results/` with the naming convention
`NN_Train_<dataset>_<arch>_<xtrust>_<ytrust>_PathSize_<N>` and
`PTAS_Eval_<dataset>_<arch>_<xtrust>_<ytrust>_eps_<ε>_PathSize_<N>`.

Cached runs are reused automatically on subsequent launches; delete the relevant
`results/` subfolder to force a full re-run.

---

## Module structure

```
patas_module/
├── __init__.py          Public API re-exports (see below)
├── __main__.py          Entry point for python -m patas_module
├── main.py              CLI argument parsing, server/client orchestration
├── subjective_logic.py  Subjective Logic operators (scalar + NumPy-vectorized)
│
├── NN/                  Neural network side (client)
│   ├── primaryNN.py     NeuralNetwork class — forward, backprop, PTAS socket call
│   ├── primaryNNGen.py  Generalised multi-hidden-layer NeuralNetwork
│   ├── PTAStemplate.py  PTAS class — Trust Feedforward (Alg. 5), Parameter-Trust Update (Alg. 6)
│   ├── datasets.py      Data loaders: Breast Cancer, MNIST, poisoned MNIST
│   ├── utils.py         Pickle helpers, logging utilities
│   ├── CancerPTAS.py    Cancer-specific training experiment
│   ├── MnistPTAS.py     MNIST-specific training experiment
│   ├── GtrsbPTAS.py     GTSRB experiment
│   ├── cancer_eval.py   Post-training cancer evaluation
│   ├── cancer_pred.py   Cancer prediction helper
│   ├── cifar10.py       CIFAR-10H experiment
│   ├── gtsrb.py         GTSRB (balancing bias) experiment
│   ├── Mnist.py         MNIST standalone runner
│   └── plot.py          Plotting utilities
│
├── PTASTemp/            Communication layer (server socket)
│   ├── ptasInterface.py PTASInterface — TCP listener / sender abstraction
│   ├── messageObject.py MessageObject — serialisation of gradient batches
│   ├── mode.py          Mode enum (SERVER / CLIENT / BOTH)
│   ├── listener.py      Low-level socket listener
│   ├── sender.py        Low-level socket sender
│   ├── nn2.py           Auxiliary NN utilities for the transport layer
│   └── cancer_org.py    Baseline cancer experiment (no PTAS, reference only)
│
└── concrete/            Trust-opinion data structures
    ├── TrustOpinion.py  TrustOpinion — scalar opinion (b, d, u, a)
    ├── ArrayTO.py       ArrayTO — 2-D NumPy array of opinions (shape × 3)
    ├── TensorTO.py      TensorArrayTO — batch-indexed opinion tensor (n × dim × 3)
    └── __init__.py
```

---

## Public API

After `pip install -e .` from `v2/`, or with `patas_module/` on `sys.path`:

```python
import patas_module as patas

# Core PTAS classes
from patas_module import PTAS, NeuralNetwork
from patas_module import TrustOpinion, ArrayTO, TensorArrayTO
from patas_module import PTASInterface, MessageObject, Mode

# Subjective Logic — scalar operators
from patas_module import Opinion, vacuous, trusted, distrusted
from patas_module import averaging_fusion, cumulative_fusion, fuse_many
from patas_module import discount, multiply, deduce, revise

# Subjective Logic — NumPy-vectorized (batch) operators
from patas_module import averaging_fusion_vec, cumulative_fusion_vec
from patas_module import multiply_vec, discount_vec, bpq_vec, deduce_vec

# Dataset quality operators
from patas_module import bpq, ewq, cuq
```

### Key classes

**`TrustOpinion`** — a single Subjective Logic opinion (b, d, u, a).

```python
op = TrustOpinion(b=0.8, d=0.1, u=0.1, a=0.5)
op = TrustOpinion.fill((30, 16), method="vacuous")   # 2-D array of vacuous opinions
```

**`ArrayTO`** — wraps a `(rows, cols)` NumPy array of opinions as a `(rows, cols, 3)` tensor.

**`TensorArrayTO`** — batch-indexed opinion tensor `(n, dim, 3)` used during mini-batch training.

**`PTAS`** — Trust Node Network.  Key methods:

```python
ptas = PTAS(omega_thetas, operator_mapping, nn_interface,
            trust_assessment_func, structure,
            epsilon_low=0.01, epsilon_up=None)
ptas.run_chunk()                    # run full training (server mode)
out = ptas.apply_feedforward(x_to)  # Trust Feedforward (Alg. 5)
PTAS.aggregation(out)               # collapse output opinion to scalar
```

**`NeuralNetwork`** — MLP with optional PTAS socket connection.

```python
nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True, port=5000)
nn.train(X_train, y_train, X_test, y_test, epochs=15, lr_scheduler=lambda e: 0.2)
nn.predict(X_test)
```

---

## How server and client interact

```
NN client (NeuralNetwork)              PTAS server (PTAS / PTASInterface)
──────────────────────────             ──────────────────────────────────
forward pass → gradients               listen on TCP port
                       ── gradients + activations ──→
                                        trust_assessment(x, dim)
                                        Trust Feedforward (Alg. 5)
                                        Parameter-Trust Update (Alg. 6)
                       ←── acknowledgement ─────────
next mini-batch ...
```

When `--mode both` is used, both processes are spawned via `multiprocessing.Process`
and a 1-second delay ensures the server socket is bound before the client connects.
