"""Rayleigh-Sommerfeld binary-DMD optimization package."""

from __future__ import annotations

import os
from typing import Any

# LEAF_FINCH does not use torch.compile. Disable compiler probing before any
# submodule imports PyTorch; the numerical core uses ordinary eager tensor operations.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from .config import AppConfig

__all__ = ["AppConfig", "run_simulation"]
__version__ = "1.0.0"


def run_simulation(*args: Any, **kwargs: Any):
    """Lazily import and execute the simulation runner."""

    from .runner import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)
