from __future__ import annotations

import math
import time
import threading
from collections.abc import Callable
from typing import Any

import torch

from .adam import TensorAdam
from .backend import (
    AcceleratorInfo,
    choose_pixel_chunk,
    empty_accelerator_cache,
    halve_chunk,
    is_out_of_memory,
)
from .config import AppConfig
from .geometry import (
    ObservationPlane,
    largest_regular_polygon_mask,
    orthonormal_basis_from_direction,
    sample_disk_points,
)
from .propagation import CancelledError, check_cancel, complex_abs2, complex_expi, rs_chunk_field
from .targets import TargetContext, lens_focal_for_incident_wavefront, target_amplitude

ProgressCallback = Callable[[dict[str, Any]], None]
LogCallback = Callable[[str], None]


def _emit_log(callback: LogCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def phase_aligned_mode_loss(
    field: torch.Tensor,
    target: torch.Tensor,
    *,
    shape_weight: float,
    mode_power_weight: float,
    eps: float,
):
    count_points, n_patterns = field.shape
    if target.ndim == 1:
        target = target[:, None].expand(count_points, n_patterns)
    elif target.shape != (count_points, n_patterns):
        raise ValueError(
            f"target must have shape [{count_points}] or [{count_points},{n_patterns}], got {tuple(target.shape)}"
        )
    target_norm2 = torch.sum(target.square(), dim=0).clamp_min(eps)
    field_norm2 = torch.sum(complex_abs2(field), dim=0).clamp_min(eps)
    c_re = torch.sum(field.real * target, dim=0)
    c_im = torch.sum(field.imag * target, dim=0)
    c_abs2 = c_re.square() + c_im.square()
    shape_score = torch.sqrt(c_abs2.clamp_min(eps)) / torch.sqrt(
        (field_norm2 * target_norm2).clamp_min(eps)
    )
    mode_power = c_abs2 / target_norm2
    shape_loss = -torch.mean(shape_score)
    mode_power_loss = -torch.mean(torch.log1p(mode_power))
    loss = shape_weight * shape_loss + mode_power_weight * mode_power_loss
    return loss, shape_score, mode_power, shape_loss, mode_power_loss


def _make_jittered_source(
    src_pos: torch.Tensor,
    pitch: float,
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    # fraction=0.5 means independent offsets uniformly distributed in [-pitch/2, +pitch/2].
    jitter = (
        torch.rand((src_pos.shape[0], 2), device=src_pos.device, dtype=src_pos.dtype, generator=generator)
        * 2.0
        - 1.0
    ) * (pitch * fraction)
    result = src_pos.clone()
    result[:, :2] += jitter
    return result


def optimize_patterns(
    config: AppConfig,
    plane: ObservationPlane,
    target_context: TargetContext,
    accelerator: AcceleratorInfo,
    incident_field_fn,
    *,
    cancel_event: threading.Event | None = None,
    stop_after_epoch_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    resume_state: dict[str, Any] | None = None,
    resume_optimizer: bool = True,
) -> dict[str, Any]:
    """Optimize DMD logits and optionally resume a portable checkpoint.

    ``cancel_event`` interrupts inside the current RS calculation and does not
    promise a checkpoint. ``stop_after_epoch_event`` is checked only after a
    complete optimizer update; it therefore returns a consistent model that the
    runner can save and resume from.
    """

    opt = config.optimization
    dmd = config.dmd
    device = torch.device(accelerator.device)
    dtype = torch.float32
    torch.manual_seed(opt.seed)
    generator = torch.Generator(device=device).manual_seed(opt.seed)

    mask, src_pos, aperture_radius, aperture_sides = largest_regular_polygon_mask(dmd, device, dtype)
    n_pixels = int(src_pos.shape[0])
    incident = incident_field_fn(src_pos).to(torch.complex64)
    _emit_log(
        log_callback,
        f"Numerics: real=torch.float32, complex=torch.complex64, device={device}",
    )

    start_epoch = 0
    history: list[dict[str, float | int]] = []
    elapsed_offset = 0.0
    resumed_from: str | None = None
    previous_target_epochs = opt.n_steps
    previous_temperature = opt.temperature_start

    if resume_state is None:
        logits = torch.empty((opt.n_patterns, n_pixels), device=device, dtype=dtype).normal_(0.0, 0.05)
    else:
        saved_logits = resume_state["logits"]
        saved_mask = resume_state["mask"].to(dtype=torch.bool, device="cpu")
        current_mask = mask.detach().to("cpu", dtype=torch.bool)
        if tuple(saved_logits.shape) != (opt.n_patterns, n_pixels):
            raise ValueError(
                "Loaded model is incompatible with the current number of patterns or active DMD pixels: "
                f"checkpoint {tuple(saved_logits.shape)}, current {(opt.n_patterns, n_pixels)}"
            )
        if tuple(saved_mask.shape) != tuple(current_mask.shape) or not torch.equal(saved_mask, current_mask):
            raise ValueError(
                "Loaded model uses a different DMD size or aperture mask. "
                "Restore the checkpoint configuration or start without this model."
            )
        logits = saved_logits.to(device=device, dtype=dtype).clone()
        resumed_from = str(resume_state.get("source_path", "checkpoint"))
        previous_target_epochs = int(resume_state.get("target_epochs", opt.n_steps))
        previous_temperature = float(resume_state.get("last_temperature", opt.temperature_start))
        if resume_optimizer:
            start_epoch = int(resume_state.get("completed_epochs", 0))
            history = [dict(row) for row in resume_state.get("history", [])]
            elapsed_offset = float(resume_state.get("elapsed_s", 0.0))
            saved_generator_state = resume_state.get("generator_state")
            saved_generator_backend = resume_state.get("generator_backend")
            if (
                isinstance(saved_generator_state, torch.Tensor)
                and saved_generator_backend == accelerator.backend
            ):
                generator.set_state(saved_generator_state.to("cpu"))
            elif isinstance(saved_generator_state, torch.Tensor):
                _emit_log(
                    log_callback,
                    "Checkpoint was created on a different backend; model and Adam state were restored, "
                    "but stochastic disk sampling restarts from the configured seed.",
                )
            _emit_log(
                log_callback,
                f"Continuing model from epoch {start_epoch:,}; target total epochs: {opt.n_steps:,}",
            )
        else:
            _emit_log(log_callback, "Starting a new optimizer run from loaded model weights")

    logits.requires_grad_(True)
    optimizer = TensorAdam(logits, lr=opt.lr)
    if resume_state is not None and resume_optimizer:
        optimizer_state = resume_state.get("optimizer")
        if isinstance(optimizer_state, dict):
            # The learning rate selected in the current GUI/config takes
            # precedence; moments, betas and step counter are restored.
            optimizer.load_state_dict(optimizer_state, keep_current_lr=True)

    if config.backend.auto_batch or opt.pixel_chunk is None:
        pixel_chunk = choose_pixel_chunk(
            n_points=opt.points_per_step,
            n_pixels=n_pixels,
            n_patterns=opt.n_patterns,
            training=True,
            accelerator=accelerator,
            policy=config.backend,
        )
    else:
        pixel_chunk = min(n_pixels, int(opt.pixel_chunk))
    if resume_state is not None and resume_optimizer:
        saved_chunk = resume_state.get("pixel_chunk")
        if isinstance(saved_chunk, int) and saved_chunk > 0 and not config.backend.auto_batch:
            pixel_chunk = min(n_pixels, saved_chunk)
    total_dmd_pixels = int(dmd.nx * dmd.ny)
    active_fraction = 100.0 * float(n_pixels) / max(1, total_dmd_pixels)
    chunks_per_epoch = math.ceil(n_pixels / max(1, pixel_chunk))
    _emit_log(
        log_callback,
        f"Active DMD pixels: {n_pixels:,} / {total_dmd_pixels:,} ({active_fraction:.1f}%)",
    )
    _emit_log(
        log_callback,
        f"Propagation pixel chunk: {pixel_chunk:,} pixels; chunks per epoch: {chunks_per_epoch:,}",
    )

    run_start = time.perf_counter()
    pattern_ids = torch.arange(opt.n_patterns, device=device)
    sampling_basis = orthonormal_basis_from_direction(plane.direction, device, dtype)
    stopped_early = False
    completed_epochs = start_epoch
    last_temperature = previous_temperature if start_epoch else opt.temperature_start

    if start_epoch >= opt.n_steps:
        _emit_log(
            log_callback,
            f"Checkpoint already contains {start_epoch:,} completed epochs; "
            f"increase target total epochs above {start_epoch:,} to continue.",
        )

    def temperature_for_epoch(epoch: int) -> float:
        if start_epoch == 0 or previous_target_epochs == opt.n_steps:
            schedule_t = epoch / max(1, opt.n_steps - 1)
            return opt.temperature_start * (opt.temperature_end / opt.temperature_start) ** schedule_t
        # If the target epoch count is extended, continue cooling from the
        # checkpoint temperature instead of jumping back to a hotter schedule.
        local_t = (epoch - start_epoch + 1) / max(1, opt.n_steps - start_epoch)
        start_temperature = min(opt.temperature_start, previous_temperature)
        return start_temperature * (opt.temperature_end / start_temperature) ** local_t

    for epoch in range(start_epoch, opt.n_steps):
        check_cancel(cancel_event)
        epoch_start = time.perf_counter()
        temperature = temperature_for_epoch(epoch)
        points, rho, theta = sample_disk_points(
            plane,
            opt.points_per_step,
            device=device,
            dtype=dtype,
            generator=generator,
            radial_sampling_f=opt.radial_sampling_f,
            basis=sampling_basis,
        )
        target = target_amplitude(rho, theta, pattern_ids, target_context).to(device=device, dtype=dtype)
        if opt.dmd_pixel_jitter:
            source_epoch = _make_jittered_source(src_pos, dmd.pitch, opt.jitter_fraction, generator)
            incident_epoch = incident_field_fn(source_epoch).to(torch.complex64)
        else:
            source_epoch = src_pos
            incident_epoch = incident

        while True:
            check_cancel(cancel_event)
            optimizer.zero_grad(set_to_none=True)
            try:
                soft = torch.sigmoid(logits / temperature)
                hard = (soft >= 0.5).to(soft.dtype)
                transmission = hard.detach() - soft.detach() + soft
                field = rs_chunk_field(
                    points,
                    source_epoch,
                    transmission,
                    incident_epoch,
                    dmd,
                    pixel_chunk,
                    cancel_event=cancel_event,
                    use_checkpoint=True,
                )
                data_loss, shape_score, mode_power, shape_loss, mode_power_loss = phase_aligned_mode_loss(
                    field,
                    target,
                    shape_weight=opt.shape_weight,
                    mode_power_weight=opt.mode_power_weight,
                    eps=opt.eps,
                )
                bin_loss = torch.mean(soft * (1.0 - soft))
                shape_component = opt.shape_weight * shape_loss
                mode_power_component = opt.mode_power_weight * mode_power_loss
                binarization_component = opt.binarization_weight * bin_loss
                loss = data_loss + binarization_component
                loss.backward()
                optimizer.step()
                break
            except RuntimeError as exc:
                if isinstance(exc, CancelledError):
                    raise
                if not is_out_of_memory(exc) or pixel_chunk <= config.backend.min_pixel_chunk:
                    raise
                old_chunk = pixel_chunk
                pixel_chunk = halve_chunk(pixel_chunk, config.backend.min_pixel_chunk)
                optimizer.zero_grad(set_to_none=True)
                soft = hard = transmission = field = data_loss = loss = None
                shape_score = mode_power = shape_loss = mode_power_loss = bin_loss = None
                shape_component = mode_power_component = binarization_component = None
                empty_accelerator_cache(accelerator.device)
                _emit_log(log_callback, f"Out of memory: reducing pixel chunk {old_chunk:,} -> {pixel_chunk:,}")

        completed_epochs = epoch + 1
        last_temperature = float(temperature)
        elapsed = elapsed_offset + time.perf_counter() - run_start
        # Move all scalar diagnostics to the host in one transfer. Individual
        # ``.cpu()`` calls would synchronize the accelerator repeatedly after
        # every epoch and can become noticeable for short epochs.
        metric_values = torch.stack(
            (
                loss,
                data_loss,
                shape_loss,
                mode_power_loss,
                bin_loss,
                shape_component,
                mode_power_component,
                binarization_component,
                torch.mean(shape_score),
                torch.mean(mode_power),
            )
        ).detach().to(device="cpu", dtype=torch.float32).tolist()
        (
            total_loss_value,
            data_loss_value,
            shape_loss_value,
            mode_power_loss_value,
            bin_loss_value,
            shape_component_value,
            mode_power_component_value,
            binarization_component_value,
            mean_shape_score_value,
            mean_mode_power_value,
        ) = metric_values
        metrics = {
            "step": epoch,
            "epoch": completed_epochs,
            "total_loss": float(total_loss_value),
            "data_loss": float(data_loss_value),
            "shape_loss_unweighted": float(shape_loss_value),
            "mode_power_loss_unweighted": float(mode_power_loss_value),
            "binarization_loss_unweighted": float(bin_loss_value),
            "shape_loss_weighted": float(shape_component_value),
            "mode_power_loss_weighted": float(mode_power_component_value),
            "binarization_loss_weighted": float(binarization_component_value),
            "mean_shape_score": float(mean_shape_score_value),
            "mean_mode_power": float(mean_mode_power_value),
            "temperature": last_temperature,
            "elapsed_s": float(elapsed),
            "epoch_time_s": float(time.perf_counter() - epoch_start),
            "step_time_s": float(time.perf_counter() - epoch_start),
            "pixel_chunk": int(pixel_chunk),
        }
        # Store every epoch. log_every now controls text verbosity only; the GUI
        # and saved convergence files retain the complete loss history.
        history.append(metrics)
        if epoch % max(1, opt.log_every) == 0 or completed_epochs == opt.n_steps:
            _emit_log(
                log_callback,
                f"epoch={completed_epochs:5d}/{opt.n_steps} loss={metrics['total_loss']:.4e} "
                f"shape={metrics['shape_loss_weighted']:.4e} "
                f"mode={metrics['mode_power_loss_weighted']:.4e} "
                f"bin={metrics['binarization_loss_weighted']:.4e} chunk={pixel_chunk:,}",
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "optimization",
                    "fraction": completed_epochs / opt.n_steps,
                    "target_epochs": opt.n_steps,
                    **metrics,
                }
            )

        # Graceful stop is deliberately checked only here, after the optimizer
        # update and metric collection, so the returned checkpoint is coherent.
        if stop_after_epoch_event is not None and stop_after_epoch_event.is_set():
            stopped_early = completed_epochs < opt.n_steps
            _emit_log(
                log_callback,
                f"Stop requested: optimization ended cleanly after epoch {completed_epochs:,}.",
            )
            break

    with torch.no_grad():
        binary_inside = (logits >= 0).to(torch.uint8)
        patterns = torch.zeros((opt.n_patterns, dmd.ny, dmd.nx), device=device, dtype=torch.uint8)
        patterns[:, mask] = binary_inside

    elapsed_total = elapsed_offset + time.perf_counter() - run_start
    training_state = {
        "format_version": 1,
        "kind": "leaf_finch_training_state",
        "config": config.to_dict(),
        "logits": logits.detach().to("cpu"),
        "mask": mask.detach().to("cpu", dtype=torch.bool),
        "optimizer": optimizer.state_dict(cpu=True),
        "completed_epochs": int(completed_epochs),
        "target_epochs": int(opt.n_steps),
        "history": history,
        "generator_state": generator.get_state().to("cpu"),
        "generator_backend": accelerator.backend,
        "last_temperature": float(last_temperature),
        "elapsed_s": float(elapsed_total),
        "pixel_chunk": int(pixel_chunk),
        "resumed_from": resumed_from,
    }
    return {
        "patterns": patterns,
        "mask": mask,
        "src_pos": src_pos,
        "history": history,
        "aperture_radius_m": aperture_radius,
        "aperture_sides": aperture_sides,
        "target_context": target_context,
        "pixel_chunk": pixel_chunk,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "training_state": training_state,
        "resumed_from": resumed_from,
    }

def generate_fzp_patterns(
    config: AppConfig,
    plane: ObservationPlane,
    target_context: TargetContext,
    accelerator: AcceleratorInfo,
    *,
    cancel_event: threading.Event | None = None,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    check_cancel(cancel_event)
    dmd = config.dmd
    opt = config.optimization
    device = torch.device(accelerator.device)
    mask, src_pos, aperture_radius, aperture_sides = largest_regular_polygon_mask(
        dmd, device, torch.float32
    )
    x, y = src_pos[:, 0], src_pos[:, 1]
    r2 = x.square() + y.square()
    k0 = 2.0 * torch.pi / target_context.wavelength
    f_plus_eff = target_context.plane_distance + target_context.distance_to_focus
    f_minus_eff = target_context.plane_distance - target_context.distance_to_focus
    if abs(f_minus_eff) < 1e-30:
        raise ValueError("Invalid FZP geometry: L - distance_to_focus is zero")
    zi = target_context.incident_wavefront_radius_zi
    f_plus = lens_focal_for_incident_wavefront(f_plus_eff, zi)
    f_minus = lens_focal_for_incident_wavefront(f_minus_eff, zi)
    ids = torch.arange(opt.n_patterns, device=device, dtype=torch.float32)[:, None]
    phi0 = 2.0 * torch.pi * ids / opt.n_patterns
    lens_plus = complex_expi(-k0 * r2[None, :] / (2.0 * f_plus))
    field = complex_expi(phi0) * lens_plus
    if abs(target_context.distance_to_focus) > 1e-5:
        field = field + complex_expi(-k0 * r2[None, :] / (2.0 * f_minus))
    carrier_phase = k0 * (x[None, :] * plane.direction[0] + y[None, :] * plane.direction[1])
    field = field * complex_expi(carrier_phase)
    binary_inside = (field.real >= 0.0).to(torch.uint8)
    patterns = torch.zeros((opt.n_patterns, dmd.ny, dmd.nx), device=device, dtype=torch.uint8)
    patterns[:, mask] = binary_inside
    _emit_log(log_callback, f"Generated {opt.n_patterns} deterministic FZP patterns")
    return {
        "patterns": patterns,
        "mask": mask,
        "src_pos": src_pos,
        "history": [],
        "aperture_radius_m": aperture_radius,
        "aperture_sides": aperture_sides,
        "target_context": target_context,
        "pixel_chunk": None,
    }
