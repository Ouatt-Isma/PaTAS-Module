"""
Entry point for ``python -m patas_module``.

Equivalent to running ``python main.py`` from inside the patas_module/ directory.

Examples
--------
# Both server and client, cancer dataset, trust-trust:
python -m patas_module --mode both --testcase cancer --xtrust trust --ytrust trust

# MNIST, server only:
python -m patas_module --mode server --testcase mnist --xtrust trust --ytrust trust

# MNIST with poisoning:
python -m patas_module --mode both --testcase mnist --mnist-poisoned-soph --mnist-patch-size 4
"""

import sys
import os

# Ensure patas_module/ is on sys.path so internal absolute imports resolve.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def main_entry():
    """Console-script entry point (used by pyproject.toml [project.scripts])."""
    from main import main  # main.py is inside patas_module/
    main()


if __name__ == "__main__":
    main_entry()
