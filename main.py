import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import multiprocessing
import time
from NN.utils import writeto

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
    mnist_poisoned_soph: int | None= None

lr_cancer = 0.2
def get_lr_cancer(epoch):
    return lr_cancer
    # if epoch < 10:
    #     return lr_cancer
    # else:
    #     return lr_cancer*0.5

lr_mnist = 0.001
def get_lr_mnist(epoch):
    return lr_mnist
    # if epoch < 10:
    #     return lr_mnist
    # else:
    #     return lr_mnist*0.5


TEST_CASES: dict[str, TestCaseConfig] = {
    "mnist": TestCaseConfig(dataset="mnist", input_dim=28 * 28, output_dim=10, hidden_dim=10,
                             epochs=10, batch_size=18, learning_rate=get_lr_mnist, epsilon_low=10e-2, epsilon_up=None),
    "cancer": TestCaseConfig(dataset="cancer", input_dim=30, output_dim=2, hidden_dim=16, 
                             epochs=15, batch_size=64, learning_rate=get_lr_cancer, epsilon_low=10e-2, epsilon_up=None),
    # cifar10 script in this repo currently uses binary classification (2 classes)
    # "cifar10": TestCaseConfig(dataset="cifar10", input_dim=32 * 32 * 3, output_dim=2, hidden_dim=64),
    # "gtsrb": TestCaseConfig(dataset="gtsrb", input_dim=32 * 32, output_dim=43, hidden_dim=64),
}


TrustGen = Callable[[int, int], ArrayTO]


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
    parser.add_argument("--epsilon-low", type=float, default=10e-2)
    parser.add_argument("--epsilon-up", type=float, default=None)
    parser.add_argument(
        "--mnist-patch-size",
        type=int,
        default=5,
        help="MNIST patch size for poisoned trust generation (used with --mnist-poisoned-soph)",
    )
    parser.add_argument(
        "--mnist-poisoned-soph",
        action="store_true",
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


def build_trust_generator(spec: str) -> TrustGen:
    normalized = _normalize_spec(spec)

    if normalized in ALIASES:
        method = ALIASES[normalized]

        def _gen(n: int, dim: int) -> ArrayTO:
            return ArrayTO(TrustOpinion.fill(shape=(n, dim), method=method))

        return _gen

    triplet = _parse_triplet(spec)
    if triplet is not None:
        t, d, u = triplet
        opinion = TrustOpinion(t, d, u)

        def _gen(n: int, dim: int) -> ArrayTO:
            return ArrayTO(TrustOpinion.fill(shape=(n, dim), value=opinion))

        return _gen

    raise ValueError(
        f"Unsupported trust spec '{spec}'. Use trust/distrust/vacuous/random or a triplet t,d,u"
    )


def _check_patch(sample: np.ndarray, patch_size: int, img_size: int = 28, patch_value: float = 1.0) -> bool:
    x = sample.reshape(img_size, img_size).copy()
    for i in range(patch_size):
        for j in range(patch_size):
            if x[i][j] != patch_value:
                return False
    return True


def build_mnist_poisoned_soph_generator(patch_size: int) -> TrustGen:
    if patch_size <= 0:
        raise ValueError("--mnist-patch-size must be > 0")

    from NN.datasets import load_mnist, load_poisoned_mnist

    x_train, _, y_train, _ = load_mnist()
    x_train, y_train, _ = load_poisoned_mnist(x_train, y_train, patch_size=patch_size)

    input_dim_mnist = 28 * 28
    output_dim_mnist = 10

    patched_ind: list[int] = []

    def _gen(x: np.ndarray, dim: int) -> ArrayTO:
        n = len(x)

        if dim == input_dim_mnist:
            res = ArrayTO(TrustOpinion.fill(shape=(n, dim), method="trust"))
            patched_ind.clear()
            for t in range(n):
                if _check_patch(x_train[int(x[t])], patch_size=patch_size):
                    patched_ind.append(t)
                    for i in range(patch_size):
                        for j in range(patch_size):
                            res.value[t][28 * i + j] = TrustOpinion.dtrust()
            return res

        if dim == output_dim_mnist:
            res = ArrayTO(TrustOpinion.fill(shape=(n, dim), method="trust"))
            indices = np.argwhere(x == 1)
            filtered_indices = indices[np.isin(indices[:, 1], [9, 6])]
            for i in filtered_indices[:, 0]:
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
            return res

        return ArrayTO(TrustOpinion.fill(shape=(n, dim), method="vacuous"))

    return _gen

def ptas_evaluation(ptas: PTAS, input_dim: int, datapath: str):
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
    print("Aggregated Value: ", PTAS.aggregation(a))
    print()
    writeto(a, datapath+"\\at.pkl")

    print("Apply Feed Forward on Vacuous Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    print(a)
    print("Aggregated Value: ", PTAS.aggregation(a))
    print()
    writeto(a, datapath+"\\av.pkl")

    print("Apply Feed Forward on fully Untrusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    print(a)
    print("Aggregated Value: ", PTAS.aggregation(a))
    print()
    writeto(a, datapath+"\\ad.pkl")
    


def start_ptas(args):
    cfg = TEST_CASES[args.testcase]

    hidden_dim = args.hidden_neurons if args.hidden_neurons is not None else cfg.hidden_dim
    epsilon_low = cfg.epsilon_low if args.epsilon_low is None else args.epsilon_low
    epsilon_up = cfg.epsilon_up if args.epsilon_up is None else args.epsilon_up

    x_gen = build_trust_generator(args.xtrust)
    y_gen = build_trust_generator(args.ytrust)

    def trust_assessment(x: np.ndarray, dim: int) -> ArrayTO:
        n = len(x)
        if dim == cfg.input_dim:
            return x_gen(n, dim)
        if dim == cfg.output_dim:
            return y_gen(n, dim)
        return ArrayTO(TrustOpinion.fill(shape=(n, dim), method="vacuous"))

    structure = [cfg.input_dim, hidden_dim, cfg.output_dim]

    omega_thetas = [
        ArrayTO(TrustOpinion.fill(shape=(cfg.input_dim + 1, hidden_dim), method="vacuous")),
        ArrayTO(TrustOpinion.fill(shape=(hidden_dim + 1, cfg.output_dim), method="vacuous")),
    ]

    ptas = PTAS(
        omega_thetas=omega_thetas,
        operator_mapping=None,
        nn_interface=PTASInterface(args.port),
        trust_assessment_func=trust_assessment,
        structure=structure,
        epsilon_low=epsilon_low,
        epsilon_up=epsilon_up,
        eval=True,
    )

    print("PTAS server started.")
    ptas.run_chunk()
    print("PTAS server finished processing.")
    print("Evaluating PTAS outputs...")
    datapath=f"PTAS_Eval_{args.testcase}_{args.xtrust}_{args.ytrust}"

    ptas_evaluation(ptas, cfg.input_dim, datapath=datapath)
    PTAS.eval_plot(ptas.EVAL, cfg.output_dim, None,f"{datapath}\\plot_ptas.pdf",  n_epoch=cfg.epochs)
def start_client(cfg: TestCaseConfig, not_ptas: bool):
    from NN.primaryNN import NeuralNetwork
    from NN.datasets import load_data  # adapt per testcase

    print("Starting NN client...")

    X_train, X_test, y_train, y_test, encoder = load_data(cfg.dataset, 
            TRUST_TO_DATASET[cfg.x_trust], 
            TRUST_TO_DATASET[cfg.y_trust])


    input_size = cfg.input_dim
    output_size = cfg.output_dim
    
    nn = NeuralNetwork(
        input_size,
        cfg.hidden_dim,
        output_size,
        ptas=False if not_ptas else True,
        operation=True,
        port=cfg.port  
    )

    # nn.train(X_train, y_train, X_test, y_test, epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate)
    datapath = f"NN_Train_{cfg.dataset}_{cfg.x_trust}_{cfg.y_trust}"
    os.makedirs(datapath, exist_ok=True)

    nn.train(X_train, y_train, X_test, y_test, 
             epochs=cfg.epochs, batch_size=cfg.batch_size, 
             lr_scheduler=cfg.learning_rate, plot=True,
             fname=datapath)

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

    nn.forward(X_test[0], getactivated=True)
    nn.end()

def main():
    args = parse_args()
    cfg = TEST_CASES[args.testcase]
    
    hidden_dim = args.hidden_neurons if args.hidden_neurons is not None else cfg.hidden_dim
    epsilon_low = args.epsilon_low if args.epsilon_low is None else args.epsilon_low
    epsilon_up = args.epsilon_up if args.epsilon_up is None else args.epsilon_up
    
    cfg = TestCaseConfig(
        dataset=cfg.dataset,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dim=hidden_dim,
        x_trust=args.xtrust,
        y_trust=args.ytrust,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        epsilon_low=epsilon_low,
        epsilon_up=epsilon_up,
        port=args.port, 
        mnist_patch_size=args.mnist_patch_size,
        mnist_poisoned_soph=args.mnist_poisoned_soph,
    )
    
    if args.mode == "server":
        print("Starting PTAS server...")
        start_ptas(args)

    elif args.mode == "client":
        print("Starting NN client...")
        start_client(cfg, args.not_ptas)

    elif args.mode == "both":
        print("Launching PTAS + Client...")

        ptas_process = multiprocessing.Process(
            target=start_ptas,
            args=(args,),
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
