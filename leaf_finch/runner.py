from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .backend import resolve_accelerator
from .config import AppConfig
from .geometry import make_reconstruction_plane, plane_from_angles
from .io import (
    create_output_dir,
    save_convergence,
    save_patterns,
    save_reconstruction,
    write_run_metadata,
)
from .optimization import generate_fzp_patterns, optimize_patterns
from .propagation import check_cancel, make_incident_field_fn
from .reconstruction import reconstruct_field_on_grid
from .targets import build_target_context
from .training_state import load_training_checkpoint, save_training_checkpoint


def run_simulation(
    config: AppConfig | dict[str, Any],
    *,
    cancel_event: threading.Event | None = None,
    stop_after_epoch_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    model_checkpoint: str | Path | None = None,
    resume_optimizer: bool = True,
) -> dict[str, Any]:
    if isinstance(config, dict):
        config = AppConfig.from_dict(config)
    config.validate()
    accelerator = resolve_accelerator(config.backend.device)
    if log_callback:
        log_callback(f"Using {accelerator.label}")
    device = torch.device(accelerator.device)
    plane = plane_from_angles(config.plane_angles)
    target_context = build_target_context(config, plane)
    incident_field_fn = make_incident_field_fn(
        config.dmd.wavelength, config.incident.wavefront_radius_zi
    )

    resume_state = None
    if model_checkpoint is not None:
        resume_state = load_training_checkpoint(model_checkpoint)
        if log_callback:
            mode = "optimizer continuation" if resume_optimizer else "weights-only initialization"
            log_callback(f"Loaded model {resume_state['source_path']} ({mode})")

    check_cancel(cancel_event)
    if target_context.target_type == "fzp":
        if resume_state is not None:
            raise ValueError("Deterministic FZP generation does not use a trainable model checkpoint")
        result = generate_fzp_patterns(
            config,
            plane,
            target_context,
            accelerator,
            cancel_event=cancel_event,
            log_callback=log_callback,
        )
        result["completed_epochs"] = 0
        result["stopped_early"] = False
        result["training_state"] = None
        result["resumed_from"] = None
        if progress_callback:
            progress_callback({"phase": "optimization", "fraction": 1.0})
    else:
        result = optimize_patterns(
            config,
            plane,
            target_context,
            accelerator,
            incident_field_fn,
            cancel_event=cancel_event,
            stop_after_epoch_event=stop_after_epoch_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
            resume_state=resume_state,
            resume_optimizer=resume_optimizer,
        )

    check_cancel(cancel_event)
    out_dir, stem = create_output_dir(config)
    config.save_json(out_dir / "config.json")
    if log_callback:
        log_callback(f"Writing results to {out_dir}")
    files: dict[str, Any] = {}
    # Save the resumable state before rendering previews or reports. This makes
    # graceful stop robust even if a later optional output operation fails.
    model_path: Path | None = None
    training_state = result.get("training_state")
    if isinstance(training_state, dict):
        model_path = save_training_checkpoint(out_dir / f"model_{stem}.pt", training_state)
        files["model"] = str(model_path)
        if log_callback:
            log_callback(
                f"Saved model checkpoint after epoch {result.get('completed_epochs', 0):,}: {model_path}"
            )

    files["patterns"] = save_patterns(result["patterns"].cpu(), out_dir, stem)
    if config.output.save_convergence:
        files["convergence"] = save_convergence(result["history"], out_dir, stem)

    reconstruction = None
    stopped_early = bool(result.get("stopped_early", False))
    if stopped_early and log_callback:
        log_callback("Skipping reconstruction because a stop-after-current-epoch request was received.")
    if (
        not stopped_early
        and config.reconstruction.enabled
        and config.output.save_reconstruction
    ):
        check_cancel(cancel_event)
        reconstruction_plane = make_reconstruction_plane(plane, config.reconstruction)
        reconstruction = reconstruct_field_on_grid(
            result["patterns"],
            result["mask"],
            result["src_pos"],
            config,
            reconstruction_plane,
            accelerator,
            incident_field_fn,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        files["reconstruction"] = save_reconstruction(
            reconstruction,
            out_dir,
            stem,
            reconstruction_plane,
            config,
            device,
        )

    write_run_metadata(out_dir, config, accelerator, result, files, reconstruction)
    if progress_callback:
        progress_callback(
            {
                "phase": "stopped" if stopped_early else "finished",
                "fraction": 1.0,
                "completed_epochs": int(result.get("completed_epochs", 0)),
                "target_epochs": int(config.optimization.n_steps),
            }
        )
    return {
        "out_dir": str(out_dir),
        "stem": stem,
        "accelerator": accelerator,
        "files": files,
        "patterns": result["patterns"].detach().cpu(),
        "history": result["history"],
        "reconstruction": reconstruction,
        "model_path": None if model_path is None else str(model_path),
        "completed_epochs": int(result.get("completed_epochs", 0)),
        "stopped_early": stopped_early,
        "resumed_from": result.get("resumed_from"),
    }
