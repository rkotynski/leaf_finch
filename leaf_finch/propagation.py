from __future__ import annotations

import math
import threading
from collections.abc import Callable

import numpy as np
import torch

from .config import DMDConfig


class CancelledError(RuntimeError):
    pass


def check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Simulation cancelled")


def complex_abs2(z: torch.Tensor) -> torch.Tensor:
    return z.real.square() + z.imag.square()


def complex_expi(phase: torch.Tensor) -> torch.Tensor:
    """Backend-portable exp(i*phase) without a complex exponential kernel."""

    return torch.complex(torch.cos(phase), torch.sin(phase))


def _rs_block(
    points: torch.Tensor,
    src: torch.Tensor,
    pattern_block: torch.Tensor,
    incident_block: torch.Tensor,
    *,
    k0: float,
    area: float,
    wavelength: float,
) -> torch.Tensor:
    displacement = points[:, None, :] - src[None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=2).clamp_min(1e-12)
    obliquity = displacement[:, :, 2] / distance
    kr = k0 * distance
    cos_kr = torch.cos(kr)
    sin_kr = torch.sin(kr)
    inv_kr = 1.0 / kr
    scale = obliquity * area / (wavelength * distance)
    kernel = torch.complex(
        scale * (cos_kr * inv_kr + sin_kr),
        scale * (sin_kr * inv_kr - cos_kr),
    )
    return (kernel * incident_block[None, :]) @ pattern_block.T


class _RecomputeRSBlock(torch.autograd.Function):
    """Checkpoint one RS block without importing ``torch.utils.checkpoint``.

    ``torch.utils.checkpoint.checkpoint`` is wrapped with PyTorch's lazy
    TorchDynamo decorator.  Calling it can import Triton even though this
    application never requests ``torch.compile``.  This specialized autograd
    function stores only the block inputs and recomputes the large K x C
    intermediates during backward, preserving the intended VRAM saving without
    touching TorchDynamo or Triton.
    """

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        src: torch.Tensor,
        pattern_block: torch.Tensor,
        incident_block: torch.Tensor,
        k0: float,
        area: float,
        wavelength: float,
        cancel_event: threading.Event | None,
    ) -> torch.Tensor:
        check_cancel(cancel_event)
        ctx.save_for_backward(points, src, pattern_block, incident_block)
        ctx.k0 = float(k0)
        ctx.area = float(area)
        ctx.wavelength = float(wavelength)
        ctx.cancel_event = cancel_event
        with torch.no_grad():
            return _rs_block(
                points,
                src,
                pattern_block,
                incident_block,
                k0=ctx.k0,
                area=ctx.area,
                wavelength=ctx.wavelength,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        check_cancel(ctx.cancel_event)
        points, src, pattern_block, incident_block = ctx.saved_tensors

        # Only the binary-pattern relaxation requires a gradient. Geometry and
        # illumination are constants in the current physical model.
        with torch.enable_grad():
            patterns = pattern_block.detach().requires_grad_(True)
            output = _rs_block(
                points.detach(),
                src.detach(),
                patterns,
                incident_block.detach(),
                k0=ctx.k0,
                area=ctx.area,
                wavelength=ctx.wavelength,
            )
            (grad_patterns,) = torch.autograd.grad(
                output,
                patterns,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )

        return None, None, grad_patterns, None, None, None, None, None


def rs_chunk_field(
    points: torch.Tensor,
    src_pos: torch.Tensor,
    trans_patterns: torch.Tensor,
    incident: torch.Tensor,
    dmd: DMDConfig,
    pixel_chunk: int,
    *,
    cancel_event: threading.Event | None = None,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Evaluate Rayleigh-Sommerfeld propagation without a full K x M matrix.

    During optimization, a project-local recomputation checkpoint avoids
    retaining every K x C intermediate tensor.  Unlike
    ``torch.utils.checkpoint``, it does not lazily import TorchDynamo/Triton.
    """

    k0 = 2 * math.pi / dmd.wavelength
    area = dmd.pitch * dmd.pitch
    if points.dtype != torch.float32 or src_pos.dtype != torch.float32:
        raise TypeError("Rayleigh-Sommerfeld geometry must use torch.float32")
    if trans_patterns.dtype != torch.float32:
        raise TypeError("Rayleigh-Sommerfeld transmissions must use torch.float32")
    if points.device != src_pos.device or points.device != trans_patterns.device:
        raise ValueError("RS points, source positions, and transmissions must be on one device")
    count_points = points.shape[0]
    n_patterns, n_pixels = trans_patterns.shape
    device = points.device
    complex_dtype = torch.complex64
    out = torch.zeros((count_points, n_patterns), device=device, dtype=complex_dtype)
    patt = trans_patterns.to(complex_dtype)
    inc = incident.to(complex_dtype)

    for start in range(0, n_pixels, pixel_chunk):
        check_cancel(cancel_event)
        stop = min(n_pixels, start + pixel_chunk)
        src = src_pos[start:stop]
        patterns = patt[:, start:stop]
        illumination = inc[start:stop]
        if use_checkpoint and trans_patterns.requires_grad:
            contribution = _RecomputeRSBlock.apply(
                points,
                src,
                patterns,
                illumination,
                k0,
                area,
                dmd.wavelength,
                cancel_event,
            )
        else:
            contribution = _rs_block(
                points,
                src,
                patterns,
                illumination,
                k0=k0,
                area=area,
                wavelength=dmd.wavelength,
            )
        out = out + contribution
    return out


def spherical_incident_field(src_pos: torch.Tensor, wavelength: float, zi: float) -> torch.Tensor:
    zi = float(zi)
    if src_pos.dtype != torch.float32:
        raise TypeError("Incident-field coordinates must use torch.float32")
    dtype = torch.complex64
    if not math.isfinite(zi):
        return torch.ones(src_pos.shape[0], device=src_pos.device, dtype=dtype)
    if abs(zi) < 1e-30:
        raise ValueError("incident.wavefront_radius_zi must be nonzero")
    k0 = 2.0 * math.pi / wavelength
    r2 = src_pos[:, 0].square() + src_pos[:, 1].square()
    return complex_expi(k0 * r2 / (2.0 * zi))


def make_incident_field_fn(wavelength: float, zi: float) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda src_pos: spherical_incident_field(src_pos, wavelength, zi)


def fresnel_propagate_fft_convolution(
    field: np.ndarray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    *,
    device: torch.device,
) -> np.ndarray:
    if abs(z) < 1e-30:
        raise ValueError("Fresnel propagation distance must be nonzero")
    field_t = torch.as_tensor(np.asarray(field, dtype=np.complex64), device=device)
    if field_t.ndim != 2:
        raise ValueError("field must be a 2D array")
    h, w = field_t.shape
    pad_h, pad_w = 2 * h, 2 * w
    yy = (torch.arange(h, device=device, dtype=torch.float32) - h // 2) * float(dy)
    xx = (torch.arange(w, device=device, dtype=torch.float32) - w // 2) * float(dx)
    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")
    k0 = 2.0 * math.pi / float(wavelength)
    phase = k0 * (x_grid.square() + y_grid.square()) / (2.0 * float(z))
    prefactor = complex(math.cos(k0 * z), math.sin(k0 * z)) / (1j * wavelength * z)
    impulse = complex_expi(phase) * prefactor
    field_pad = torch.zeros((pad_h, pad_w), device=device, dtype=torch.complex64)
    impulse_pad = torch.zeros_like(field_pad)
    field_pad[:h, :w] = field_t
    impulse_pad[:h, :w] = torch.fft.ifftshift(impulse)
    result = torch.fft.ifft2(torch.fft.fft2(field_pad) * torch.fft.fft2(impulse_pad))
    return (result[:h, :w] * (dx * dy)).detach().cpu().numpy()
