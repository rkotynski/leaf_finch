from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KIND = "leaf_finch_training_state"


def save_training_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    """Atomically save a portable optimizer checkpoint.

    All tensors supplied by the optimizer are already moved to CPU.  The
    temporary-file + replace sequence prevents a partially written checkpoint
    from replacing the last valid model if saving is interrupted.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload.setdefault("format_version", CHECKPOINT_FORMAT_VERSION)
    payload.setdefault("kind", CHECKPOINT_KIND)
    payload.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a LEAF_FINCH optimizer checkpoint on CPU."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        # Current PyTorch releases default to weights_only=True.  Passing it
        # explicitly documents that checkpoints contain tensors and plain data,
        # not arbitrary Python objects.
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("The selected file is not a LEAF_FINCH training checkpoint")
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Unsupported checkpoint kind")
    version = int(payload.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    logits = payload.get("logits")
    mask = payload.get("mask")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("Checkpoint does not contain a valid logits tensor")
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError("Checkpoint does not contain a valid aperture mask")
    payload["source_path"] = str(source.resolve())
    return payload


def checkpoint_summary(state: dict[str, Any]) -> str:
    completed = int(state.get("completed_epochs", 0))
    target = int(state.get("target_epochs", completed))
    shape = tuple(int(value) for value in state["logits"].shape)
    return f"epoch {completed}/{target}, logits {shape[0]}×{shape[1]}"
