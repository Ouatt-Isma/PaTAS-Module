"""NN sub-package: neural network and PTAS runtime."""
# Note: imports here use the flat absolute style that works once
# patas_module/__init__.py has set sys.path.
#
# datasets.py has a top-level `import kagglehub` (only needed for GTSRB),
# so we expose it lazily via __getattr__ rather than importing eagerly.

from NN.PTAStemplate import PTAS          # noqa: E402
from NN.primaryNN import NeuralNetwork    # noqa: E402
from NN import utils                      # noqa: E402

__all__ = ["PTAS", "NeuralNetwork", "datasets", "utils"]


def __getattr__(name: str):
    if name == "datasets":
        import NN.datasets as _ds  # lazy — avoids top-level kagglehub import
        return _ds
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
