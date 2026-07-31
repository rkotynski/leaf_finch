from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import AppConfig
from .geometry import ObservationPlane


@dataclass(frozen=True)
class TargetContext:
    target_type: str
    n_patterns: int
    wavelength: float
    distance_to_focus: float
    plane_distance: float
    ring_radius: float
    edge_apodization_width: float
    incident_wavefront_radius_zi: float
    siemens_spokes: int
    siemens_rotation_step_deg: float
    siemens_rotation_offset_deg: float
    siemens_low: float
    siemens_high: float


def build_target_context(config: AppConfig, plane: ObservationPlane) -> TargetContext:
    target = config.target
    width = target.edge_apodization_width
    if width is None:
        width = target.edge_apodization_fraction * plane.ring_radius
    width = max(0.0, min(float(width), plane.ring_radius))
    rotation_step = target.siemens_rotation_step_deg
    if rotation_step is None:
        rotation_step = 180.0 / (target.siemens_spokes * config.optimization.n_patterns)
    return TargetContext(
        target_type=target.target_type.lower(),
        n_patterns=config.optimization.n_patterns,
        wavelength=config.dmd.wavelength,
        distance_to_focus=target.distance_to_focus,
        plane_distance=plane.distance,
        ring_radius=plane.ring_radius,
        edge_apodization_width=width,
        incident_wavefront_radius_zi=config.incident.wavefront_radius_zi,
        siemens_spokes=target.siemens_spokes,
        siemens_rotation_step_deg=float(rotation_step),
        siemens_rotation_offset_deg=target.siemens_rotation_offset_deg,
        siemens_low=target.siemens_low,
        siemens_high=target.siemens_high,
    )


def target_amplitude(
    rho: torch.Tensor,
    theta: torch.Tensor,
    pattern_ids: torch.Tensor,
    ctx: TargetContext,
) -> torch.Tensor:
    k_wave = 2.0 * math.pi / ctx.wavelength
    ids = pattern_ids[None, :].to(device=rho.device, dtype=rho.dtype)
    phi0 = 2.0 * math.pi * ids / ctx.n_patterns

    if ctx.target_type == "siemens":
        rotations = math.radians(ctx.siemens_rotation_offset_deg) + ids * math.radians(
            ctx.siemens_rotation_step_deg
        )
        star_phase = ctx.siemens_spokes * (theta[:, None] - rotations)
        binary = (torch.cos(star_phase) >= 0.0).to(rho.dtype)
        target = ctx.siemens_low + (ctx.siemens_high - ctx.siemens_low) * binary
    else:
        if ctx.target_type == "spherical":
            radius = torch.sqrt(rho[:, None].square() + ctx.distance_to_focus**2)
            target = torch.cos(k_wave * radius - phi0 / 2.0) / radius.clamp_min(1e-30)
        else:
            if abs(ctx.distance_to_focus) < 1e-30:
                raise ValueError("target.distance_to_focus must be nonzero for cosine target")
            target = torch.cos(
                k_wave * rho[:, None].square() / (2.0 * ctx.distance_to_focus) - phi0 / 2.0
            )

    if ctx.edge_apodization_width > 0.0:
        edge_start = max(0.0, ctx.ring_radius - ctx.edge_apodization_width)
        t = ((rho - edge_start) / max(ctx.edge_apodization_width, 1e-30)).clamp(0.0, 1.0)
        window = 0.5 * (1.0 + torch.cos(math.pi * t))
        target = target * window[:, None]
    return target


def lens_focal_for_incident_wavefront(effective_focal: float, zi: float) -> float:
    if abs(effective_focal) < 1e-30:
        raise ValueError("effective focal length must be nonzero")
    if not math.isfinite(zi):
        return effective_focal
    if abs(zi) < 1e-30:
        raise ValueError("incident wavefront radius must be nonzero")
    denominator = 1.0 / effective_focal + 1.0 / zi
    if abs(denominator) < 1e-30:
        raise ValueError("incident wavefront makes the lens focal length infinite")
    return 1.0 / denominator
