from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass

import torch

from .config import BackendConfig


@dataclass(frozen=True)
class AcceleratorInfo:
    device: str
    backend: str
    name: str
    free_bytes: int | None = None
    total_bytes: int | None = None

    @property
    def label(self) -> str:
        memory = ""
        if self.total_bytes:
            memory = f" ({self.total_bytes / 2**30:.1f} GiB)"
        return f"{self.backend.upper()}: {self.name}{memory} [{self.device}]"


def _cpu_memory() -> tuple[int | None, int | None]:
    try:
        import psutil  # optional

        mem = psutil.virtual_memory()
        return int(mem.available), int(mem.total)
    except Exception:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            available = os.sysconf("SC_AVPHYS_PAGES") * page
            total = os.sysconf("SC_PHYS_PAGES") * page
            return int(available), int(total)
        except Exception:
            return None, None


def list_accelerators() -> list[AcceleratorInfo]:
    free, total = _cpu_memory()
    devices = [AcceleratorInfo("cpu", "cpu", "Host processor", free, total)]
    if torch.cuda.is_available():
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            try:
                free_b, total_b = torch.cuda.mem_get_info(index)
            except Exception:
                free_b = total_b = None
            try:
                name = torch.cuda.get_device_name(index)
            except Exception:
                name = f"GPU {index}"
            devices.append(
                AcceleratorInfo(
                    device=device,
                    backend=backend,
                    name=name,
                    free_bytes=free_b,
                    total_bytes=total_b,
                )
            )
    return devices


def resolve_accelerator(requested: str = "auto") -> AcceleratorInfo:
    devices = list_accelerators()
    if requested in {"", "auto", None}:
        return devices[1] if len(devices) > 1 else devices[0]
    if requested in {"cuda", "rocm", "hip"}:
        wanted = "rocm" if requested in {"rocm", "hip"} else "cuda"
        matches = [device for device in devices if device.backend == wanted]
        if matches:
            return matches[0]
        raise ValueError(f"Requested {wanted.upper()} backend is not available")
    for device in devices:
        if device.device == requested:
            return device
    raise ValueError(f"Requested device {requested!r} is not available")


def refresh_memory(info: AcceleratorInfo) -> AcceleratorInfo:
    if info.device.startswith("cuda"):
        index = torch.device(info.device).index or 0
        free_b, total_b = torch.cuda.mem_get_info(index)
        return AcceleratorInfo(info.device, info.backend, info.name, free_b, total_b)
    free_b, total_b = _cpu_memory()
    return AcceleratorInfo(info.device, info.backend, info.name, free_b, total_b)


def empty_accelerator_cache(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    oom_type = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in text or "hip out of memory" in text


def _round_chunk(value: int, minimum: int, maximum: int) -> int:
    if maximum <= minimum:
        return max(1, int(maximum))
    value = max(minimum, min(maximum, int(value)))
    if value < 1024:
        quantum = 64
    elif value < 8192:
        quantum = 256
    else:
        quantum = 1024
    return max(minimum, (value // quantum) * quantum)


def choose_pixel_chunk(
    *,
    n_points: int,
    n_pixels: int,
    n_patterns: int,
    training: bool,
    accelerator: AcceleratorInfo,
    policy: BackendConfig,
) -> int:
    """Estimate a conservative source-pixel chunk from currently free memory.

    The Rayleigh-Sommerfeld kernel creates several K x C real/complex tensors.
    Autograd retains additional intermediates, so training uses a larger
    per-interaction estimate than reconstruction.
    """

    if n_points < 1 or n_pixels < 1:
        raise ValueError("n_points and n_pixels must be positive")
    refreshed = refresh_memory(accelerator)
    free = refreshed.free_bytes
    if free is None:
        return min(n_pixels, policy.cpu_max_pixel_chunk if accelerator.backend == "cpu" else 16_384)

    reserve = policy.reserve_memory_mb * 2**20
    usable = max(64 * 2**20, int(free * policy.memory_fraction) - reserve)
    # Empirical upper bounds for all live tensors per point-pixel pair.
    bytes_per_pair = 176 if training else 88
    fixed = n_patterns * n_pixels * 16 + n_points * n_patterns * 32
    available_for_kernel = max(32 * 2**20, usable - fixed)
    estimated = available_for_kernel // max(1, n_points * bytes_per_pair)

    maximum = min(n_pixels, policy.max_pixel_chunk)
    if accelerator.backend == "cpu":
        maximum = min(maximum, policy.cpu_max_pixel_chunk)
    return _round_chunk(estimated, policy.min_pixel_chunk, maximum)


def choose_reconstruction_chunks(
    *,
    n_pixels: int,
    total_points: int,
    accelerator: AcceleratorInfo,
    policy: BackendConfig,
) -> tuple[int, int]:
    """Choose point and pixel chunks for no-grad reconstruction."""

    refreshed = refresh_memory(accelerator)
    free = refreshed.free_bytes or 2**30
    reserve = policy.reserve_memory_mb * 2**20
    usable = max(64 * 2**20, int(free * policy.memory_fraction) - reserve)
    # The optimized reconstruction path retains the complete complex64 field
    # and the 3-D float32 observation coordinates on the accelerator. Account
    # for these persistent arrays before sizing the much larger K x C blocks.
    persistent_bytes = total_points * (8 + 3 * 4)
    usable_for_interactions = max(32 * 2**20, usable - persistent_bytes)
    interactions = max(4096, usable_for_interactions // 96)

    preferred_pixels = min(n_pixels, 32_768 if accelerator.backend != "cpu" else 8_192)
    point_chunk = max(32, min(4096, total_points, interactions // max(1, preferred_pixels)))
    pixel_chunk = choose_pixel_chunk(
        n_points=point_chunk,
        n_pixels=n_pixels,
        n_patterns=1,
        training=False,
        accelerator=accelerator,
        policy=policy,
    )
    return int(point_chunk), int(pixel_chunk)


def halve_chunk(value: int, minimum: int = 64) -> int:
    if value <= minimum:
        return value
    return max(minimum, int(math.floor(value / 2)))
