from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import torch

from .backend import (
    AcceleratorInfo,
    choose_reconstruction_chunks,
    empty_accelerator_cache,
    halve_chunk,
    is_out_of_memory,
)
from .config import AppConfig
from .geometry import ObservationPlane, orthonormal_basis_from_direction
from .propagation import check_cancel, complex_abs2, complex_expi, rs_chunk_field


def rotate_complex_2d_to_real(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if z.ndim != 2 or not torch.is_complex(z):
        raise ValueError("Expected a complex 2D tensor")
    center = (z.shape[0] // 2, z.shape[1] // 2)
    moment = torch.sum(z * z)
    value = z[center]
    moment_phi = -0.5 * torch.atan2(moment.imag, moment.real)
    center_phi = -torch.atan2(value.imag, value.real)
    # Keep the decision on the tensor device. Calling ``item()`` here would
    # force a CPU/GPU synchronization for every reconstructed pattern.
    phi = torch.where(
        complex_abs2(moment) > float(eps) ** 2,
        moment_phi,
        center_phi,
    )
    out = complex_expi(phi) * z
    sign = torch.where(
        out[center].real < 0,
        out[center].real.new_tensor(-1.0),
        out[center].real.new_tensor(1.0),
    )
    return out * sign


def reconstruct_field_on_grid(
    patterns: torch.Tensor,
    mask: torch.Tensor,
    src_pos: torch.Tensor,
    config: AppConfig,
    plane: ObservationPlane,
    accelerator: AcceleratorInfo,
    incident_field_fn,
    *,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    recon = config.reconstruction
    dmd = config.dmd
    device = torch.device(accelerator.device)
    n_patterns = int(patterns.shape[0])
    n, e1, e2 = orthonormal_basis_from_direction(plane.direction, device, torch.float32)
    center = plane.distance * n
    extent = recon.extent_factor * plane.ring_radius
    grid_size = int(recon.grid_size)
    u = torch.linspace(-extent, extent, grid_size, device=device, dtype=torch.float32)
    v = torch.linspace(-extent, extent, grid_size, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    points_flat = (center[None, None, :] + uu[:, :, None] * e1 + vv[:, :, None] * e2).reshape(-1, 3)
    total_points = int(points_flat.shape[0])
    n_pixels = int(src_pos.shape[0])

    auto_point, auto_pixel = choose_reconstruction_chunks(
        n_pixels=n_pixels,
        total_points=total_points,
        accelerator=accelerator,
        policy=config.backend,
    )
    point_chunk = auto_point if config.backend.auto_batch or recon.point_chunk is None else int(recon.point_chunk)
    pixel_chunk = auto_pixel if config.backend.auto_batch or recon.pixel_chunk is None else int(recon.pixel_chunk)
    point_chunk = max(1, min(point_chunk, total_points))
    pixel_chunk = min(n_pixels, max(1, pixel_chunk))
    if log_callback:
        log_callback(f"Reconstruction chunks: points={point_chunk:,}, pixels={pixel_chunk:,}")

    incident = incident_field_fn(src_pos).to(torch.complex64)
    mask_flat = mask.reshape(-1)
    intensity_all: list[torch.Tensor] = []
    phase_all: list[torch.Tensor] = []

    # For ordinary reconstruction grids the complete complex field is small
    # compared with the K x C Rayleigh-Sommerfeld work arrays. Keeping it on
    # the accelerator avoids one device-to-host copy and synchronization per
    # point chunk. If allocation itself fails, retain the low-memory CPU path.
    field_buffer_device = device
    try:
        field_buffer = torch.empty(total_points, dtype=torch.complex64, device=device)
    except RuntimeError as exc:
        if device.type != "cuda" or not is_out_of_memory(exc):
            raise
        empty_accelerator_cache(accelerator.device)
        field_buffer_device = torch.device("cpu")
        field_buffer = torch.empty(total_points, dtype=torch.complex64, device="cpu")
        if log_callback:
            log_callback(
                "Complete reconstruction field does not fit on the accelerator; "
                "using a CPU accumulation buffer."
            )
    if log_callback:
        log_callback(
            f"Reconstruction accumulation buffer: {field_buffer_device.type}, "
            f"dtype={field_buffer.dtype}"
        )

    with torch.no_grad():
        for pattern_index in range(n_patterns):
            check_cancel(cancel_event)
            transmission = patterns[pattern_index].to(device=device, dtype=torch.float32).reshape(-1)[mask_flat]
            transmission = transmission[None, :].contiguous()
            cursor = 0
            while cursor < total_points:
                check_cancel(cancel_event)
                current_points = min(point_chunk, total_points - cursor)
                try:
                    value = rs_chunk_field(
                        points_flat[cursor : cursor + current_points],
                        src_pos,
                        transmission,
                        incident,
                        dmd,
                        pixel_chunk,
                        cancel_event=cancel_event,
                    )[:, 0]
                    destination = field_buffer[cursor : cursor + current_points]
                    if field_buffer_device == device:
                        destination.copy_(value)
                    else:
                        destination.copy_(value, non_blocking=False)
                    cursor += current_points
                except RuntimeError as exc:
                    if not is_out_of_memory(exc):
                        raise
                    old_pair = (point_chunk, pixel_chunk)
                    if pixel_chunk > config.backend.min_pixel_chunk:
                        pixel_chunk = halve_chunk(pixel_chunk, config.backend.min_pixel_chunk)
                    elif point_chunk > 16:
                        point_chunk = halve_chunk(point_chunk, 16)
                    else:
                        raise
                    empty_accelerator_cache(accelerator.device)
                    if log_callback:
                        log_callback(
                            f"Reconstruction OOM: chunks {old_pair[0]:,}x{old_pair[1]:,} -> "
                            f"{point_chunk:,}x{pixel_chunk:,}"
                        )
                if progress_callback:
                    fraction = (pattern_index + cursor / total_points) / n_patterns
                    progress_callback({"phase": "reconstruction", "fraction": fraction})

            field = rotate_complex_2d_to_real(field_buffer.reshape(grid_size, grid_size))
            # Compute post-processing with torch on the same device as the
            # accumulation buffer, then transfer each final result only once.
            intensity_all.append(complex_abs2(field).to(device="cpu", dtype=torch.float32))
            phase_all.append(
                torch.atan2(field.imag, field.real).to(device="cpu", dtype=torch.float32)
            )

    return {
        "intensity": torch.stack(intensity_all),
        "phase": torch.stack(phase_all),
        "u": u.cpu(),
        "v": v.cpu(),
        "extent_m": float(extent),
        "grid_size": grid_size,
        "distance_m": float(plane.distance),
        "ring_radius_m": float(plane.ring_radius),
        "point_chunk": point_chunk,
        "pixel_chunk": pixel_chunk,
    }
