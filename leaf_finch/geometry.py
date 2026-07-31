from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from .config import DMDConfig, PlaneAnglesConfig


@dataclass(frozen=True)
class ObservationPlane:
    direction: tuple[float, float, float]
    distance: float
    ring_radius: float


def plane_from_angles(cfg: PlaneAnglesConfig) -> ObservationPlane:
    tx = math.radians(cfg.theta_x_deg)
    ty = math.radians(cfg.theta_y_deg)
    dx = math.sin(tx)
    dy = math.sin(ty)
    z2 = 1.0 - dx * dx - dy * dy
    if z2 <= 0.0:
        raise ValueError("Observation direction has no real positive z component")
    return ObservationPlane(
        direction=(dx, dy, math.sqrt(z2)),
        distance=float(cfg.L),
        ring_radius=float(cfg.L) * math.tan(math.radians(cfg.theta_ring_deg)),
    )


def make_reconstruction_plane(base: ObservationPlane, cfg) -> ObservationPlane:
    distance = float(cfg.distance) if cfg.distance is not None else base.distance
    if cfg.ring_radius is not None:
        radius = float(cfg.ring_radius)
    elif cfg.theta_ring_deg is not None:
        radius = distance * math.tan(math.radians(float(cfg.theta_ring_deg)))
    else:
        radius = base.ring_radius
    if distance <= 0 or radius <= 0:
        raise ValueError("Reconstruction distance and radius must be positive")
    return ObservationPlane(base.direction, distance, radius)


def largest_regular_polygon_mask(
    cfg: DMDConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    x = (torch.arange(cfg.nx, device=device, dtype=dtype) - (cfg.nx - 1) / 2) * cfg.pitch
    y = (torch.arange(cfg.ny, device=device, dtype=dtype) - (cfg.ny - 1) / 2) * cfg.pitch
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    half_w = cfg.nx * cfg.pitch / 2
    half_h = cfg.ny * cfg.pitch / 2

    if not cfg.use_aperture:
        mask = torch.ones((cfg.ny, cfg.nx), device=device, dtype=torch.bool)
        pos = torch.stack(
            (xx.reshape(-1), yy.reshape(-1), torch.zeros(cfg.nx * cfg.ny, device=device, dtype=dtype)),
            dim=1,
        )
        return mask, pos.contiguous(), math.hypot(half_w, half_h), 0

    n_sides = max(4, int(math.ceil(cfg.aperture_sides)))
    if n_sides % 2:
        n_sides += 1
    normal_rotation = math.pi / 2.0
    normal_angles = [normal_rotation + 2.0 * math.pi * j / n_sides for j in range(n_sides)]
    vertex_angles = [normal_rotation - math.pi / n_sides + 2.0 * math.pi * i / n_sides for i in range(n_sides)]
    max_abs_cos = max(abs(math.cos(a)) for a in vertex_angles)
    max_abs_sin = max(abs(math.sin(a)) for a in vertex_angles)
    radius = min(half_w / max_abs_cos, half_h / max_abs_sin)
    apothem = radius * math.cos(math.pi / n_sides)

    mask = torch.ones((cfg.ny, cfg.nx), device=device, dtype=torch.bool)
    for alpha in normal_angles:
        mask &= xx * math.cos(alpha) + yy * math.sin(alpha) <= apothem
    zeros = torch.zeros(int(mask.sum().item()), device=device, dtype=dtype)
    pos = torch.stack((xx[mask], yy[mask], zeros), dim=1)
    return mask, pos.contiguous(), radius, n_sides


def orthonormal_basis_from_direction(
    direction: Iterable[float], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = tuple(float(value) for value in direction)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("Observation direction must be nonzero")
    # Select the stable reference vector on the host from the original
    # direction. This avoids a device ``item()`` synchronization, which was
    # especially wasteful when this helper was called once per epoch.
    normalized_z = values[2] / norm
    reference = (0.0, 1.0, 0.0) if abs(normalized_z) > 0.95 else (0.0, 0.0, 1.0)
    n = torch.tensor(values, device=device, dtype=dtype)
    n = n / torch.linalg.vector_norm(n)
    ref = torch.tensor(reference, device=device, dtype=dtype)
    e1 = torch.linalg.cross(n, ref, dim=0)
    e1 = e1 / torch.linalg.vector_norm(e1)
    e2 = torch.linalg.cross(n, e1, dim=0)
    return n, e1, e2


def sample_disk_points(
    plane: ObservationPlane,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    radial_sampling_f: float,
    basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if basis is None:
        n, e1, e2 = orthonormal_basis_from_direction(plane.direction, device, dtype)
    else:
        n, e1, e2 = basis
    theta = 2 * math.pi * torch.rand(count, device=device, dtype=dtype, generator=generator)
    u = torch.rand(count, device=device, dtype=dtype, generator=generator)
    f = float(radial_sampling_f)
    rho = plane.ring_radius * (u * f + torch.sqrt(u) * (1.0 - f))
    center = plane.distance * n
    points = center[None, :] + rho[:, None] * (
        torch.cos(theta)[:, None] * e1 + torch.sin(theta)[:, None] * e2
    )
    return points.contiguous(), rho, theta
