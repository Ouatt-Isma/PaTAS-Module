"""
PATAS — Probabilistic Trust Assessment System for Neural Networks.

Usage (after ``pip install -e .`` from the v2/ directory):

    import patas_module as patas

    # Core classes
    from patas_module import PTAS, NeuralNetwork
    from patas_module import TrustOpinion, ArrayTO, TensorArrayTO
    from patas_module import PTASInterface, MessageObject, Mode

    # Subjective Logic primitives (Opinion + operators)
    from patas_module import Opinion, vacuous, trusted, distrusted
    from patas_module import averaging_fusion, fuse_many, discount

Or import sub-modules directly:

    from patas_module.NN.PTAStemplate import PTAS
    from patas_module.concrete.TrustOpinion import TrustOpinion
    from patas_module.subjective_logic import Opinion, fuse_many
"""

import sys as _sys
import os as _os

# ── Path bootstrap ────────────────────────────────────────────────────────────
# All internal code uses "flat" absolute imports (e.g. ``from NN.PTAStemplate
# import PTAS``).  Adding this directory to sys.path lets those imports resolve
# whether the package is used as a proper install or run in-place.
_module_dir = _os.path.dirname(_os.path.abspath(__file__))
if _module_dir not in _sys.path:
    _sys.path.insert(0, _module_dir)

# ── Subjective Logic layer (canonical source inside this package) ─────────────
from patas_module.subjective_logic import (                      # noqa: E402
    Opinion,
    vacuous, trusted, distrusted,
    bpq, ewq, cuq,
    discount,
    averaging_fusion, cumulative_fusion,
    weighted_belief_fusion, consensus_compromise_fusion,
    fuse_many,
    revise, multiply, conservative_div, deduce,
    opinions_to_array,
    # vectorized (NumPy) variants
    averaging_fusion_vec, cumulative_fusion_vec,
    weighted_belief_fusion_vec,
    ccf_fusion_vec, constraint_fusion_vec,
    multiply_vec, discount_vec, bpq_vec, deduce_vec,
    _normalize_vec,
)

# ── PATAS classes ─────────────────────────────────────────────────────────────
from NN.PTAStemplate import PTAS                    # noqa: E402
from NN.primaryNN import NeuralNetwork              # noqa: E402
from concrete.TrustOpinion import TrustOpinion      # noqa: E402
from concrete.ArrayTO import ArrayTO                # noqa: E402
from concrete.TensorTO import TensorArrayTO         # noqa: E402
from PTASTemp.ptasInterface import PTASInterface    # noqa: E402
from PTASTemp.messageObject import MessageObject    # noqa: E402
from PTASTemp.mode import Mode                      # noqa: E402

__version__ = "0.1.0"
__all__ = [
    # PATAS classes
    "PTAS",
    "NeuralNetwork",
    "TrustOpinion",
    "ArrayTO",
    "TensorArrayTO",
    "PTASInterface",
    "MessageObject",
    "Mode",
    # Subjective Logic — scalar
    "Opinion",
    "vacuous", "trusted", "distrusted",
    "bpq", "ewq", "cuq",
    "discount",
    "averaging_fusion", "cumulative_fusion",
    "weighted_belief_fusion", "consensus_compromise_fusion",
    "fuse_many",
    "revise", "multiply", "conservative_div", "deduce",
    "opinions_to_array",
    # Subjective Logic — vectorized (NumPy)
    "averaging_fusion_vec", "cumulative_fusion_vec",
    "weighted_belief_fusion_vec",
    "ccf_fusion_vec", "constraint_fusion_vec",
    "multiply_vec", "discount_vec", "bpq_vec", "deduce_vec",
    "_normalize_vec",
]
