import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Ensure repository root is on import path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from NN.PTAStemplate import PTAS
from PTASTemp.ptasInterface import PTASInterface
from concrete.ArrayTO import ArrayTO
from concrete.TrustOpinion import TrustOpinion


@dataclass(frozen=True)
class TestCaseConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    epsilon_low: float
    epsilon_up: float | None


TEST_CASES: dict[str, TestCaseConfig] = {
    "mnist": TestCaseConfig(input_dim=28 * 28, output_dim=10, hidden_dim=10, epsilon_low=0.1, epsilon_up=None),
    "cancer": TestCaseConfig(input_dim=30, output_dim=2, hidden_dim=16, epsilon_low=0.03, epsilon_up=None),
    # cifar10 script in this repo currently uses binary classification (2 classes)
    "cifar10": TestCaseConfig(input_dim=32 * 32 * 3, output_dim=2, hidden_dim=64, epsilon_low=0.1, epsilon_up=None),
    "gtsrb": TestCaseConfig(input_dim=32 * 32, output_dim=43, hidden_dim=64, epsilon_low=0.1, epsilon_up=None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configurable PTAS test listener. "
            "Choose testcase + X/Y trust modes + optional hidden-layer size."
        )
    )
    parser.add_argument("--testcase", choices=sorted(TEST_CASES.keys()), required=True)
    parser.add_argument("--xtrust", required=True, help="X trust spec: trust|distrust|vacuous|random|t,d,u")
    parser.add_argument("--ytrust", required=True, help="Y trust spec: trust|distrust|vacuous|random|t,d,u")
    parser.add_argument("--hidden-neurons", type=int, default=None, help="Number of neurons in the hidden layer")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--epsilon-low", type=float, default=None)
    parser.add_argument("--epsilon-up", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate/print config; do not start the PTAS socket listener",
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


def build_trust_generator(spec: str) -> Callable[[int, int], ArrayTO]:
    normalized = _normalize_spec(spec)

    aliases = {
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

    if normalized in aliases:
        method = aliases[normalized]

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


def main() -> None:
    args = parse_args()
    cfg = TEST_CASES[args.testcase]

    hidden_dim = args.hidden_neurons if args.hidden_neurons is not None else cfg.hidden_dim
    if hidden_dim <= 0:
        raise ValueError("--hidden-neurons must be > 0")

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

    print("PTAS test configuration:")
    print(f"  testcase={args.testcase}")
    print(f"  structure={structure}")
    print(f"  xtrust={args.xtrust}")
    print(f"  ytrust={args.ytrust}")
    print(f"  epsilon_low={epsilon_low}")
    print(f"  epsilon_up={epsilon_up}")
    print(f"  port={args.port}")

    if args.dry_run:
        print("Dry-run complete. No listener started.")
        return

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

    print("Starting PTAS listener (run corresponding NN training script in another process)...")
    ptas.run_chunk()


if __name__ == "__main__":
    main()
