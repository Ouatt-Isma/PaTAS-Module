import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import multiprocessing
import time
from NN.datasets import mnist_get_inverse_scaling, mnist_get_scaling, gtsrb_get_scaling
from NN.utils import writeto
from concrete.TensorTO import TensorArrayTO, fill as tfill, as_tensor, to_numpy

# Ensure repository root is on import path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from NN.PTAStemplate import PTAS
from PTASTemp.ptasInterface import PTASInterface
from concrete.ArrayTO import ArrayTO
from concrete.TrustOpinion import TrustOpinion


ALIASES = {
        "fully_trust": "trust",
        "ftrust": "trust",
        "fully_distrust": "distrust",
        "fdistrust": "distrust",
        "fully_uncertain": "vacuous",
        "fully_uncertainty": "vacuous",
        "uncertain": "vacuous",
        "uncertainty": "vacuous",
        "vacuous": "vacuous",
        "trust": "trust",
        "distrust": "distrust",
        "random": "random",
    }

TRUST_TO_DATASET = {
    "trust": "clean",
    "distrust": "corrupt",
    "vacuous": "noise"
    } 




@dataclass(frozen=True)
class TestCaseConfig:
    dataset: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int
    learning_rate: Callable[[int], float]
    epsilon_low: float

    x_trust: str | None = None # can take specific values 
    y_trust: str | None = None # can take specific values 
    epsilon_up: float | None= None
    port: int | None = None
    mnist_patch_size: int | None = None
    mnist_poisoned_soph: bool | None= None
    no_round: int | None = None
    hidden_dims: tuple[int, ...] | None = None
    x_dataset: str | None = None
    y_dataset: str | None = None
    noise_level: float | None = None
    fuse_method: str = "average"   # trust-revision fusion: average | cumulative | weighted | compromise | constraint

lr_cancer = 0.2
def get_lr_cancer(epoch):
    return lr_cancer
    # if epoch < 10:
    #     return lr_cancer
    # else:
    #     return lr_cancer*0.5

lr_mnist = 0.05
def get_lr_mnist(epoch):
    return lr_mnist

lr_gtsrb = 0.05
def get_lr_gtsrb(epoch):
    return lr_gtsrb



TEST_CASES: dict[str, TestCaseConfig] = {
    "mnist": TestCaseConfig(dataset="mnist", input_dim=28 * 28, output_dim=10, hidden_dim=128,
                             epochs=20, batch_size=128, learning_rate=get_lr_mnist,
                             epsilon_low=0.05, epsilon_up=None),
    "cancer": TestCaseConfig(dataset="cancer", input_dim=30, output_dim=2, hidden_dim=16,
                             epochs=15, batch_size=64, learning_rate=get_lr_cancer, epsilon_low=1e-1, epsilon_up=None, no_round=None),
    "gtsrb": TestCaseConfig(dataset="gtsrb", input_dim=32 * 32, output_dim=43, hidden_dim=128,
                             epochs=20, batch_size=128, learning_rate=get_lr_gtsrb,
                             epsilon_low=0.05, epsilon_up=None),
    # cifar10 script in this repo currently uses binary classification (2 classes)
    # "cifar10grey": TestCaseConfig(dataset="cifar10", input_dim=32 * 32 * 3, output_dim=2, hidden_dim=64),
}


# Per-dataset facts needed by the poisoned-backdoor machinery.
#   img_size      : square image side (input_dim = img_size²)
#   scale_patch   : maps a raw patch value (1.0) into the dataset's pixel scale
#   pois_pair     : the two classes whose labels are swapped by the backdoor
#   control_class : a clean class used as a no-backdoor control at evaluation
DATASET_META: dict[str, dict] = {
    "mnist": {"img_size": 28, "scale_patch": mnist_get_scaling,
              "output_dim": 10, "pois_pair": (6, 9), "control_class": 3},
    # GTSRB features are standardized on load (NN.datasets.load_gtsrb), so a
    # raw patch value must be mapped through the same train-split statistics.
    "gtsrb": {"img_size": 32, "scale_patch": gtsrb_get_scaling,
              "output_dim": 43, "pois_pair": (6, 9), "control_class": 3},
}


TrustGen = Callable[[int, int], ArrayTO]


# ---------------------------------------------------------------------------
# Cache-directory naming — single source of truth.
#
# The noise level is part of the cache key: two scenarios that differ only in
# the label-flip / feature-noise rate are different datasets, and without the
# suffix a sweep over rates would silently reuse the first rate's caches.
# ``noise_level=None`` keeps the historical names, so every existing cache
# trained without an explicit rate stays valid.
# ---------------------------------------------------------------------------

def _nl_suffix(noise_level) -> str:
    return "" if noise_level is None else f"_nl{noise_level:g}"


def nn_cache_dir(dataset: str, arch_str: str, x_trust, y_trust,
                 patch=None, noise_level=None) -> str:
    return (f"results/NN_Train_{dataset}_{arch_str}_{x_trust}_{y_trust}"
            f"_PathSize_{patch if patch else 'None'}{_nl_suffix(noise_level)}")


def ptas_cache_dir(dataset: str, arch_str: str, x_trust, y_trust, eps,
                   patch=None, fuse_method: str = "average",
                   noise_level=None) -> str:
    fuse_method = fuse_method or "average"
    fuse_suffix = "" if fuse_method == "average" else f"_fuse_{fuse_method}"
    return (f"results/PTAS_Eval_{dataset}_{arch_str}_{x_trust}_{y_trust}"
            f"_eps_{eps}"
            f"_PathSize_{patch if patch else 'None'}"
            f"{_nl_suffix(noise_level)}{fuse_suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configurable PTAS test listener. "
            "Choose testcase + X/Y trust modes + optional hidden-layer size."
        )
    )
    #make params auto 
    parser.add_argument(
    "--mode", choices=["server", "client", "both"], required=True, help="Run either PTAS server or NN client",)
    parser.add_argument("--testcase", choices=sorted(TEST_CASES.keys()), default="cancer")
    parser.add_argument("--xtrust", help="X trust spec: trust|distrust|vacuous|random|t,d,u", default="trust")
    parser.add_argument("--ytrust",  help="Y trust spec: trust|distrust|vacuous|random|t,d,u", default="trust")
    parser.add_argument("--hidden-neurons", type=int, help="Number of neurons in the hidden layer", default=16)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-round", type=int, default=None, 
                        help="For testing: stop PTAS after processing this many batches (full training round = all batches processed)")
    parser.add_argument("--epsilon-low", type=float, default=1e-2)
    parser.add_argument("--epsilon-up", type=float, default=None)
    parser.add_argument(
        "--mnist-patch-size",
        type=int,
        default=4,
        help="MNIST patch size for poisoned trust generation (used with --mnist-poisoned-soph)",)
    parser.add_argument(
        "--mnist-poisoned-soph",
        action="store_true",
        default=False,
        help=(
            "For testcase=mnist, use poisoned-aware trust generator equivalent to Tgenpoisoned_soph "
            "with the provided --mnist-patch-size"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate/print config; do not start the PTAS socket listener",
    )
    parser.add_argument(
    "--not-ptas",
    action="store_true",
    help="Disable PTAS mode"
    )
    return parser.parse_args()


def _normalize_spec(spec: str) -> str:
    return spec.strip().lower().replace("-", "_")


def _parse_triplet(spec: str) -> tuple[float, float, float] | None:
    cleaned = spec.strip().replace("(", "").replace(")", "").replace(" ", "")
    if not re.fullmatch(r"[0-9]*\.?[0-9]+,[0-9]*\.?[0-9]+,[0-9]*\.?[0-9]+", cleaned):
        return None
    t, d, u = (float(x) for x in cleaned.split(","))
    return t, d, u


def build_trust_generator(spec: str, dtype=np.float32):

    normalized = _normalize_spec(spec)

    # --- predefined opinions ---
    if normalized in ALIASES:

        method = ALIASES[normalized]

        def _gen(n: int, dim: int) -> TensorArrayTO:

            if method == "trust":
                t = np.ones((n, dim), dtype=dtype)
                d = np.zeros((n, dim), dtype=dtype)
                u = np.zeros((n, dim), dtype=dtype)

            elif method == "distrust":
                t = np.zeros((n, dim), dtype=dtype)
                d = np.ones((n, dim), dtype=dtype)
                u = np.zeros((n, dim), dtype=dtype)

            elif method == "vacuous":
                t = np.zeros((n, dim), dtype=dtype)
                d = np.zeros((n, dim), dtype=dtype)
                u = np.ones((n, dim), dtype=dtype)

            elif method == "random":

                t = np.random.rand(n, dim).astype(dtype)
                d = np.random.rand(n, dim).astype(dtype)

                s = t + d
                mask = s > 1

                t[mask] /= s[mask]
                d[mask] /= s[mask]

                u = 1 - (t + d)

            else:
                raise ValueError(f"Unsupported alias method: {method}")

            tensor = np.stack([t, d, u], axis=-1)

            return TensorArrayTO(tensor)

        return _gen

    # --- custom triplet ---
    triplet = _parse_triplet(spec)

    if triplet is not None:

        t, d, u = triplet

        def _gen(n: int, dim: int) -> TensorArrayTO:

            tensor = np.zeros((n, dim, 3), dtype=dtype)

            tensor[..., 0] = t
            tensor[..., 1] = d
            tensor[..., 2] = u

            return TensorArrayTO(tensor)

        return _gen

    raise ValueError(
        f"Unsupported trust spec '{spec}'. Use trust/distrust/vacuous/random or a triplet t,d,u"
    )

def _check_patch(sample: np.ndarray, patch_size: int, img_size: int = 28,
                 patch_value: float = 1.0, scale_patch=None) -> bool:
    from NN.datasets import mnist_get_scaling
    if scale_patch is None:
        scale_patch = mnist_get_scaling
    x = sample.reshape(img_size, img_size)
    scaled = scale_patch(patch_value)
    return np.allclose(x[:patch_size, :patch_size], scaled, atol=1e-2)


def build_poisoned_soph_generator(patch_size: int, dataset: str = "mnist") -> TrustGen:
    """
    Poisoned-aware trust generator for backdoored datasets (MNIST or GTSRB).

    Input trust : all pixels trusted, except the trigger-patch pixels of
                  poisoned training samples, which are distrusted.
    Output trust: high-but-uncertain trust for all classes; the two swapped
                  classes are distrusted for samples whose true label is one
                  of them.
    """
    if patch_size <= 0:
        raise ValueError("patch size must be > 0")
    meta = DATASET_META[dataset]
    img_size = meta["img_size"]
    scale_patch = meta["scale_patch"]
    pois_a, pois_b = meta["pois_pair"]

    if dataset == "mnist":
        from NN.datasets import load_mnist, load_poisoned_mnist
        x_train, _, y_train, _ = load_mnist()
        x_train, y_train, _ = load_poisoned_mnist(x_train, y_train, patch_size=patch_size)
    else:
        from NN.datasets import load_gtsrb, load_poisoned_gtsrb
        x_train, _, y_train, _ = load_gtsrb()
        x_train, y_train, _ = load_poisoned_gtsrb(x_train, y_train, patch_size=patch_size)

    input_dim = img_size * img_size
    output_dim = meta["output_dim"]

    def _gen(x: np.ndarray, dim: int) -> TensorArrayTO:
        n = len(x)

        if dim == input_dim:
            tensor = np.zeros((n, dim, 3), dtype=np.float32)
            tensor[..., 0] = 1.0  # default: trust all pixels
            for t in range(n):
                if _check_patch(x_train[int(x[t])], patch_size,
                                img_size=img_size, scale_patch=scale_patch):
                    for i in range(patch_size):
                        for j in range(patch_size):
                            tensor[t, img_size * i + j, 0] = 0.0
                            tensor[t, img_size * i + j, 1] = 1.0
            return TensorArrayTO(tensor)

        if dim == output_dim:
            # Default: high-but-uncertain trust for all classes (u>0 damps noise)
            tensor = np.zeros((n, dim, 3), dtype=np.float32)
            tensor[..., 0] = 0.9
            tensor[..., 2] = 0.1
            # Distrust the swapped-class outputs ONLY for samples whose true
            # label is one of the swapped classes.
            indices = np.argwhere(x == 1)
            filtered = indices[np.isin(indices[:, 1], [pois_a, pois_b])]
            for i in filtered[:, 0]:
                for c in (pois_a, pois_b):
                    tensor[i, c, 0] = 0.0
                    tensor[i, c, 1] = 0.9
                    tensor[i, c, 2] = 0.1
            return TensorArrayTO(tensor)

        # fallback: vacuous [0, 0, 1]
        tensor = np.zeros((n, dim, 3), dtype=np.float32)
        tensor[..., 2] = 1.0
        return TensorArrayTO(tensor)

    return _gen


def build_mnist_poisoned_soph_generator(patch_size: int) -> TrustGen:
    """Backward-compatible alias for the MNIST poisoned trust generator."""
    return build_poisoned_soph_generator(patch_size, dataset="mnist")


class TeeLogger:
    """Writes to both terminal and a file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def format_value(obj):
    """Resolve TensorArrayTO objects to their .value array (numpy or torch) for display."""
    if hasattr(obj, "value"):
        return repr(to_numpy(obj.value))
    return repr(obj)


def ptas_evaluation(ptas: PTAS, input_dim: int, datapath: str):
    log_path = os.path.join(datapath, "evaluation_log.txt")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        for i, ot in enumerate(ptas.omega_thetas):
            print(f"--------------------------- omega[{i}] ----------------------")
            print(ot.get_shape())
            print(format_value(ot))
            print("-------------------------------------------------")
            print()

        for label, method, out_name in (
            ("fully Trusted", "trust", "at.pkl"),
            ("Vacuous", "vacuous", "av.pkl"),
            ("fully Untrusted", "distrust", "ad.pkl"),
        ):
            print(f"Apply Feed Forward on {label} Input")
            a = ptas.apply_feedforward(TensorArrayTO(tfill((1, input_dim), method=method)))
            print(format_value(a))
            print("Aggregated Value: ", to_numpy(PTAS.aggregation(a)))
            print(f"Depth-Normalized Value (L={len(ptas.omega_thetas)}): ",
                  to_numpy(ptas.depth_normalized_aggregation(a)))
            print()
            # store as numpy so the pickle loads on machines without a GPU
            writeto(TensorArrayTO(a.to_numpy()), os.path.join(datapath, out_name))

        # Save omega_arrays so they can be reloaded without retraining
        writeto([ot.to_numpy().copy() for ot in ptas.omega_thetas],
                os.path.join(datapath, "omega_arrays.pkl"))

    finally:
        sys.stdout = logger.terminal  # always restore stdout
        logger.close()
    


def start_ptas(cfg, ready_event=None, post_training_callback=None, force_retrain=False):
    import pickle as _pickle
    mnist_patch_size = cfg.mnist_patch_size

    hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
    all_dims = [cfg.input_dim] + hidden_list + [cfg.output_dim]
    structure = all_dims
    arch_str = "_".join(str(h) for h in hidden_list)

    fuse_method = getattr(cfg, "fuse_method", "average") or "average"
    datapath = ptas_cache_dir(
        cfg.dataset, arch_str, cfg.x_trust, cfg.y_trust, cfg.epsilon_low,
        patch=cfg.mnist_patch_size if cfg.mnist_poisoned_soph else None,
        fuse_method=fuse_method, noise_level=cfg.noise_level,
    )
    omega_path = os.path.join(datapath, "omega_arrays.pkl")

    print("PTAS Configuration:")
    print("epsilon_low:", cfg.epsilon_low)
    print("epsilon_up:", cfg.epsilon_up)
    print("Structure:", structure)
    print("epochs:", cfg.epochs)
    print("X Trust Spec:", cfg.x_trust)
    print("Y Trust Spec:", cfg.y_trust)
    print("Using poisoned-aware trust generator for MNIST:", cfg.dataset == "mnist" and cfg.mnist_poisoned_soph)
    print("patch size (MNIST poison):", cfg.mnist_patch_size if cfg.mnist_poisoned_soph else "N/A")
    print("PTAS Port:", cfg.port)

    if cfg.dataset in DATASET_META and cfg.mnist_poisoned_soph:
        print(f"Using poisoned-aware trust generator for {cfg.dataset} with patch size", cfg.mnist_patch_size)
        trust_assessment = build_poisoned_soph_generator(mnist_patch_size, dataset=cfg.dataset)
    else:
        x_gen = build_trust_generator(cfg.x_trust)
        y_gen = build_trust_generator(cfg.y_trust)

        def trust_assessment(x: np.ndarray, dim: int) -> ArrayTO:
            n = len(x)
            if dim == cfg.input_dim:
                return x_gen(n, dim)
            if dim == cfg.output_dim:
                return y_gen(n, dim)

    omega_thetas = [
        TensorArrayTO(tfill((all_dims[i] + 1, all_dims[i + 1]), method="vacuous"))
        for i in range(len(all_dims) - 1)
    ]

    ptas = PTAS(
        omega_thetas=omega_thetas,
        operator_mapping=None,
        nn_interface=PTASInterface(cfg.port),
        trust_assessment_func=trust_assessment,
        structure=structure,
        epsilon_low=cfg.epsilon_low,
        epsilon_up=cfg.epsilon_up,
        eval=True,
        no_round=cfg.no_round,
        fuse_method=fuse_method,
    )

    if not force_retrain and os.path.exists(omega_path):
        print(f"[PTAS] Loading saved weights from {omega_path} — skipping training.")
        with open(omega_path, "rb") as _f:
            omega_arrays = _pickle.load(_f)
        # Validate saved shapes match current architecture before loading.
        expected_shapes = [
            (all_dims[i] + 1, all_dims[i + 1], 3)
            for i in range(len(all_dims) - 1)
        ]
        shapes_ok = (
            len(omega_arrays) == len(expected_shapes)
            and all(arr.shape == exp for arr, exp in zip(omega_arrays, expected_shapes))
        )
        if not shapes_ok:
            print(
                f"[PTAS] omega_arrays.pkl shape mismatch — expected {expected_shapes}, "
                f"got {[arr.shape for arr in omega_arrays]}. Deleting stale file and retraining."
            )
            os.remove(omega_path)
        else:
            for i, arr in enumerate(omega_arrays):
                ptas.omega_thetas[i] = TensorArrayTO(as_tensor(arr, device=ptas.device))
            if ready_event is not None:
                ready_event.set()
            if post_training_callback is not None:
                post_training_callback(ptas)
            print("Evaluating PTAS outputs...")
            ptas_evaluation(ptas, cfg.input_dim, datapath=datapath)
            return ptas

    print("PTAS server started.")
    ptas.run_chunk(ready_event=ready_event)
    print("PTAS server finished processing.")

    if post_training_callback is not None:
        post_training_callback(ptas)

    print("Evaluating PTAS outputs...")
    ptas_evaluation(ptas, cfg.input_dim, datapath=datapath)
    PTAS.eval_plot(ptas.EVAL, cfg.output_dim, None, os.path.join(datapath, "plot_ptas.pdf"), n_epoch=cfg.epochs)
    return ptas

def start_client(cfg: TestCaseConfig, not_ptas: bool, force_retrain: bool = False):
    import pickle as _pkl
    import json as _json
    from NN.primaryNN import NeuralNetwork
    from NN.datasets import load_data

    print("Starting NN client...")

    hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
    arch_str = "_".join(str(h) for h in hidden_list)

    print("PTAS Configuration:")
    structure = [cfg.input_dim] + hidden_list + [cfg.output_dim]
    print("Structure:", structure)
    print("epochs:", cfg.epochs)
    print("X Trust Spec:", cfg.x_trust)
    print("Y Trust Spec:", cfg.y_trust)
    print("Using poisoned-aware trust generator for MNIST:", cfg.dataset == "mnist" and cfg.mnist_poisoned_soph)
    print("patch size (MNIST poison):", cfg.mnist_patch_size if cfg.mnist_poisoned_soph else "N/A")
    print("PTAS Port:", cfg.port)

    # Resolve dataset variant: explicit x_dataset/y_dataset override the
    # trust-spec→dataset mapping (needed for custom-triplet and extra scenarios).
    x_how = cfg.x_dataset if cfg.x_dataset is not None else TRUST_TO_DATASET.get(cfg.x_trust, "clean")
    y_how = cfg.y_dataset if cfg.y_dataset is not None else TRUST_TO_DATASET.get(cfg.y_trust, "clean")
    print(f"Dataset variant  →  features: '{x_how}'  labels: '{y_how}'")

    _load_kwargs = {}
    if cfg.noise_level is not None:
        _load_kwargs["noise_level"] = cfg.noise_level
    X_train, X_test, y_train, y_test, encoder = load_data(
        cfg.dataset,
        x_how,
        y_how,
        poisoned_patch=cfg.mnist_patch_size if cfg.mnist_poisoned_soph else None,
        **_load_kwargs,
    )

    input_size = cfg.input_dim
    output_size = cfg.output_dim

    datapath = nn_cache_dir(
        cfg.dataset, arch_str, cfg.x_trust, cfg.y_trust,
        patch=cfg.mnist_patch_size if cfg.mnist_poisoned_soph else None,
        noise_level=cfg.noise_level,
    )
    os.makedirs(datapath, exist_ok=True)
    nn_model_path = os.path.join(datapath, "nn_model.pkl")
    metrics_path  = os.path.join(datapath, "metrics.txt")
    # Require metrics.txt alongside nn_model.pkl: a model saved without metrics
    # (e.g. when a previous run was interrupted before plot=True wrote the file)
    # should trigger a full retrain rather than returning NaN from a missing file.
    model_cached = (
        not force_retrain
        and os.path.exists(nn_model_path)
        and os.path.exists(metrics_path)
    )
    if model_cached:
        print(f"[NN] Saved model found — loading from {nn_model_path}, skipping training.")

    # When PTAS omega is already cached, start_ptas never binds a socket.
    # Disable PTAS communication in the NN so training always completes and
    # metrics.txt is written, and to skip the useless gradient-stream replay.
    _fuse_method = getattr(cfg, "fuse_method", "average") or "average"
    ptas_omega_path = os.path.join(
        ptas_cache_dir(
            cfg.dataset, arch_str, cfg.x_trust, cfg.y_trust, cfg.epsilon_low,
            patch=cfg.mnist_patch_size if cfg.mnist_poisoned_soph else None,
            fuse_method=_fuse_method, noise_level=cfg.noise_level,
        ),
        "omega_arrays.pkl",
    )
    ptas_omega_cached = not force_retrain and os.path.exists(ptas_omega_path)

    nn = NeuralNetwork(
        input_size,
        hidden_sizes=hidden_list,
        output_size=output_size,
        ptas=not not_ptas and not ptas_omega_cached,
        operation=True,
        port=cfg.port,
    )

    if cfg.mnist_poisoned_soph:
        from NN.datasets import add_trigger_patch

        meta = DATASET_META[cfg.dataset]
        img_size = meta["img_size"]
        pois_class = meta["pois_pair"][0]      # class whose patched samples should flip
        ctrl_class = meta["control_class"]     # clean control class (no backdoor trained)
        scaled_patch = meta["scale_patch"](1.0)
        y_test_one_hot = y_test
        ids_6 = np.where(np.argmax(y_test_one_hot, axis=1) == pois_class)[0]
        ids_3 = np.where(np.argmax(y_test_one_hot, axis=1) == ctrl_class)[0]

        print(f"Number of {pois_class}: {len(ids_6)}")
        print(f"Number of {ctrl_class}: {len(ids_3)}")

        pois_X_test_6 = np.empty((len(ids_6), *X_test.shape[1:]), dtype=X_test.dtype)
        pois_X_test_3 = np.empty((len(ids_3), *X_test.shape[1:]), dtype=X_test.dtype)

        for out_i, in_i in enumerate(ids_6):
            pois_X_test_6[out_i] = add_trigger_patch(X_test[in_i], patch_value=scaled_patch,
                                                     patch_size=cfg.mnist_patch_size, img_size=img_size)

        for out_i, in_i in enumerate(ids_3):
            pois_X_test_3[out_i] = add_trigger_patch(X_test[in_i], patch_value=scaled_patch,
                                                     patch_size=cfg.mnist_patch_size, img_size=img_size)

        if model_cached:
            nn.load_model(nn_model_path)
            if not not_ptas and not ptas_omega_cached:
                # Replay training to feed gradient stream to PTAS
                # (skipped when PTAS omega is already cached — no socket is bound)
                nn.train(
                    X_train, y_train, X_test, y_test,
                    epochs=cfg.epochs, batch_size=cfg.batch_size,
                    lr_scheduler=cfg.learning_rate, plot=False, shuffle=True,
                    fname=datapath,
                    X_non_pois_3=X_test[ids_3],
                    X_non_pois_6=X_test[ids_6],
                    X_pois_3=pois_X_test_3,
                    X_pois_6=pois_X_test_6,
                )
        else:
            nn.train(
                X_train, y_train, X_test, y_test,
                epochs=cfg.epochs, batch_size=cfg.batch_size,
                lr_scheduler=cfg.learning_rate, plot=True, shuffle=True,
                fname=datapath,
                X_non_pois_3=X_test[ids_3],
                X_non_pois_6=X_test[ids_6],
                X_pois_3=pois_X_test_3,
                X_pois_6=pois_X_test_6,
            )
            nn.save_model(nn_model_path)
    else:
        if model_cached:
            nn.load_model(nn_model_path)
            if not not_ptas and not ptas_omega_cached:
                # Replay training to feed gradient stream to PTAS
                # (skipped when PTAS omega is already cached — no socket is bound)
                nn.train(
                    X_train, y_train, X_test, y_test,
                    epochs=cfg.epochs, batch_size=cfg.batch_size,
                    lr_scheduler=cfg.learning_rate, plot=False, shuffle=True,
                    fname=datapath,
                )
        else:
            nn.train(
                X_train, y_train, X_test, y_test,
                epochs=cfg.epochs, batch_size=cfg.batch_size,
                lr_scheduler=cfg.learning_rate, plot=True, shuffle=True,
                fname=datapath,
            )
            nn.save_model(nn_model_path)

    print("X_train shape:", X_train.shape)
    predictions = nn.predict(X_train)
    print("Predictions shape:", predictions.shape)
    print("y_train shape:", y_train.shape)
    print("y_train argmax shape:", np.argmax(y_train, axis=1).shape)

    accuracy = np.mean(predictions == np.argmax(y_train, axis=1))
    print(f"Train Accuracy: {accuracy * 100:.2f}%")

    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    if cfg.mnist_poisoned_soph:
        predictions_6 = nn.predict(X_test[ids_6])
        predictions_3 = nn.predict(X_test[ids_3])

        probs_clean_6, activations_clean_6 = nn.forward(X_test[ids_6[0]], getactivated=True)
        print(f"Clean activations (sample 6[0]): {activations_clean_6}")
        probs_clean_3, activations_clean_3 = nn.forward(X_test[ids_3[0]], getactivated=True)

        pois_predictions_6 = nn.predict(pois_X_test_6)

        probs_pois_6, activations_pois_6 = nn.forward(pois_X_test_6[0], getactivated=True)
        print(f"Poisoned activations (sample 6[0]): {activations_pois_6}")
        probs_pois_3, activations_pois_3 = nn.forward(pois_X_test_3[0], getactivated=True)

        ipta_paths_data = {
            "clean_3":       activations_clean_3,
            "clean_6":       activations_clean_6,
            "pois_6":        activations_pois_6,
            "pois_3":        activations_pois_3,
            "probs_clean_3": np.array(probs_clean_3).flatten().tolist(),
            "probs_clean_6": np.array(probs_clean_6).flatten().tolist(),
            "probs_pois_6":  np.array(probs_pois_6).flatten().tolist(),
            "probs_pois_3":  np.array(probs_pois_3).flatten().tolist(),
        }
        with open(os.path.join(datapath, "ipta_paths.json"), "w", encoding="utf-8") as _jf:
            _json.dump(ipta_paths_data, _jf)

        labels_6 = np.argmax(y_test_one_hot[ids_6], axis=1)
        labels_3 = np.argmax(y_test_one_hot[ids_3], axis=1)

        acc_pois_6  = np.mean(pois_predictions_6 == labels_6)
        acc_clean_6 = np.mean(predictions_6 == labels_6)
        acc_clean_3 = np.mean(predictions_3 == labels_3)

        print(f"Accuracy on poisoned   class 6: {acc_pois_6  * 100:.2f}%")
        print(f"Accuracy on clean      class 6: {acc_clean_6 * 100:.2f}%")
        print(f"Accuracy on clean      class 3: {acc_clean_3 * 100:.2f}%")

    nn.forward(X_test[0], getactivated=True)
    nn.end()

def main():
    args = parse_args()
    cfg = TEST_CASES[args.testcase]

    cfg = TestCaseConfig(
        dataset=cfg.dataset,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dim=cfg.hidden_dim,
        x_trust=args.xtrust,
        y_trust=args.ytrust,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        epsilon_low=cfg.epsilon_low,
        epsilon_up=cfg.epsilon_up,
        port=args.port, 
        mnist_patch_size=args.mnist_patch_size,
        mnist_poisoned_soph=args.mnist_poisoned_soph,
        no_round=args.no_round, 
    )
    
    if args.mode == "server":
        print("Starting PTAS server...")
        start_ptas(cfg)

    elif args.mode == "client":
        print("Starting NN client...")
        start_client(cfg, args.not_ptas)

    elif args.mode == "both":
        print("Launching PTAS + Client...")

        ptas_process = multiprocessing.Process(
            target=start_ptas,
            args=(cfg,),
        )

        ptas_process.start()

        # Give PTAS time to bind socket
        time.sleep(1)

        client_process = multiprocessing.Process(
            target=start_client,
            args=(cfg,args.not_ptas),
        )

        client_process.start()

        client_process.join()
        ptas_process.join()

if __name__ == "__main__":
    main()
