from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_nonfinite(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        if normalized in {"-inf", "-infinity"}:
            return -math.inf
    return value


@dataclass
class BackendConfig:
    """Accelerator and memory-management options.

    ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API and
    therefore use device names such as ``cuda:0`` as well.
    """

    device: str = "auto"
    auto_batch: bool = True
    memory_fraction: float = 0.65
    reserve_memory_mb: int = 512
    min_pixel_chunk: int = 256
    max_pixel_chunk: int = 262_144
    cpu_max_pixel_chunk: int = 16_384


@dataclass
class DMDConfig:
    nx: int = 1024
    ny: int = 768
    pitch: float = 13.68e-6
    wavelength: float = 532e-9
    aperture_sides: int = 12
    use_aperture: bool = True


@dataclass
class PlaneAnglesConfig:
    L: float = 0.275
    theta_x_deg: float = 15.0
    theta_y_deg: float = 16.4
    theta_ring_deg: float = 0.15


@dataclass
class IncidentConfig:
    wavefront_radius_zi: float = 1000.0


@dataclass
class OptimizationConfig:
    n_patterns: int = 3
    n_steps: int = 1000
    points_per_step: int = 2048
    pixel_chunk: int | None = None
    lr: float = 0.05
    temperature_start: float = 1.0
    temperature_end: float = 0.08
    binarization_weight: float = 2e-4
    shape_weight: float = 1.0
    mode_power_weight: float = 0.02
    eps: float = 1e-30
    seed: int = 1
    dmd_pixel_jitter: bool = False
    jitter_fraction: float = 0.5
    radial_sampling_f: float = 0.0
    log_every: int = 25


@dataclass
class TargetConfig:
    target_type: str = "siemens"
    distance_to_focus: float = 0.025
    edge_apodization_fraction: float = 0.15
    edge_apodization_width: float | None = None
    siemens_spokes: int = 36
    siemens_rotation_step_deg: float | None = None
    siemens_rotation_offset_deg: float = 0.0
    siemens_low: float = 0.0
    siemens_high: float = 1.0


@dataclass
class ReconstructionConfig:
    enabled: bool = True
    distance: float | None = None
    ring_radius: float | None = None
    theta_ring_deg: float | None = 0.15
    grid_size: int = 300
    extent_factor: float = 1.25
    point_chunk: int | None = None
    pixel_chunk: int | None = None
    max_cols: int = 4
    fresnel_zr: float | None = None


@dataclass
class OutputConfig:
    base_dir: str = "results"
    save_convergence: bool = True
    save_reconstruction: bool = True


@dataclass
class AppConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    dmd: DMDConfig = field(default_factory=DMDConfig)
    plane_angles: PlaneAnglesConfig = field(default_factory=PlaneAnglesConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.dmd.nx <= 0 or self.dmd.ny <= 0:
            raise ValueError("DMD dimensions must be positive")
        if self.dmd.nx % 8 != 0:
            raise ValueError("DMD width must be divisible by 8 for bit-packed MAT output")
        if self.dmd.pitch <= 0 or self.dmd.wavelength <= 0:
            raise ValueError("DMD pitch and wavelength must be positive")
        sx = math.sin(math.radians(self.plane_angles.theta_x_deg))
        sy = math.sin(math.radians(self.plane_angles.theta_y_deg))
        if sx * sx + sy * sy >= 1.0:
            raise ValueError("Observation angles produce no real positive z component")
        if self.plane_angles.L <= 0 or self.plane_angles.theta_ring_deg <= 0:
            raise ValueError("Observation distance and disk angle must be positive")
        if self.optimization.n_patterns < 3:
            raise ValueError("At least three phase-shifted patterns are required")
        if self.optimization.n_steps < 1 or self.optimization.points_per_step < 1:
            raise ValueError("Optimization steps and sampled points must be positive")
        if not 0.0 <= self.optimization.jitter_fraction <= 0.5:
            raise ValueError("jitter_fraction must be in [0, 0.5]")
        if not 0.0 <= self.optimization.radial_sampling_f <= 1.0:
            raise ValueError("radial_sampling_f must be in [0, 1]")
        if self.target.target_type.lower() not in {"cosine", "spherical", "fzp", "siemens"}:
            raise ValueError("target_type must be cosine, spherical, fzp, or siemens")
        if self.target.siemens_spokes < 2:
            raise ValueError("siemens_spokes must be >= 2")
        if self.reconstruction.grid_size < 16:
            raise ValueError("reconstruction.grid_size must be >= 16")
        if not 0.05 <= self.backend.memory_fraction <= 0.95:
            raise ValueError("backend.memory_fraction must be in [0.05, 0.95]")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        data = dict(data)
        incident_data = dict(data.get("incident", {}))
        if "wavefront_radius_zi" in incident_data:
            incident_data["wavefront_radius_zi"] = _parse_nonfinite(
                incident_data["wavefront_radius_zi"]
            )

        cfg = cls(
            backend=BackendConfig(**dict(data.get("backend", {}))),
            dmd=DMDConfig(**dict(data.get("dmd", {}))),
            plane_angles=PlaneAnglesConfig(**dict(data.get("plane_angles", {}))),
            incident=IncidentConfig(**incident_data),
            optimization=OptimizationConfig(**dict(data.get("optimization", {}))),
            target=TargetConfig(**dict(data.get("target", {}))),
            reconstruction=ReconstructionConfig(**dict(data.get("reconstruction", {}))),
            output=OutputConfig(**dict(data.get("output", {}))),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load_json(cls, path: str | Path) -> "AppConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
