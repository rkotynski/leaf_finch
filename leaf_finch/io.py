from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

import matplotlib
if "PyQt5" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import savemat

from .backend import empty_accelerator_cache, is_out_of_memory
from .config import AppConfig
from .geometry import ObservationPlane
from .propagation import fresnel_propagate_fft_convolution


def patterns_to_numpy(patterns: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(patterns, torch.Tensor):
        array = patterns.detach().to("cpu", dtype=torch.uint8).numpy()
    else:
        array = np.asarray(patterns, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError(f"patterns must have shape [N, ny, nx], got {array.shape}")
    return (array != 0).astype(np.uint8, copy=False)


def create_output_dir(config: AppConfig) -> tuple[Path, str]:
    def fmt(value: float, decimals: int = 3) -> str:
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")

    p = config.plane_angles
    t = config.target
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    stem = (
        f"{t.target_type}_{stamp}_L{fmt(100*p.L, 2)}cm_"
        f"d{fmt(100*t.distance_to_focus, 2)}cm_"
        f"ang{fmt(p.theta_x_deg)}_{fmt(p.theta_y_deg)}deg"
    )
    base = Path(config.output.base_dir).expanduser()
    out_dir = base / stem
    suffix = 1
    while out_dir.exists():
        out_dir = base / f"{stem}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True)
    return out_dir, out_dir.name


def save_patterns(patterns: torch.Tensor, out_dir: Path, stem: str) -> dict[str, str | tuple[int, ...]]:
    array = patterns_to_numpy(patterns)
    if array.shape[2] % 8:
        raise ValueError("Pattern width must be divisible by 8")
    packed = np.packbits(array, axis=2, bitorder="big")
    mat_path = out_dir / f"patterns_{stem}.mat"
    pdf_path = out_dir / f"patterns_{stem}.pdf"
    png_path = out_dir / f"patterns_{stem}.png"
    npz_path = out_dir / f"patterns_{stem}.npz"
    savemat(mat_path, {"Patterns": packed}, do_compression=True)
    np.savez_compressed(npz_path, patterns=array)

    cols = min(3, array.shape[0])
    rows = math.ceil(array.shape[0] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < array.shape[0]:
            ax.imshow(array[index], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Pattern {index}")
    fig.tight_layout(pad=0.4)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"mat": str(mat_path), "pdf": str(pdf_path), "png": str(png_path), "npz": str(npz_path), "packed_shape": packed.shape}


def save_convergence(history: list[dict[str, Any]], out_dir: Path, stem: str) -> dict[str, str] | None:
    if not history:
        return None
    csv_path = out_dir / f"convergence_{stem}.csv"
    json_path = out_dir / f"convergence_{stem}.json"
    pdf_path = out_dir / f"convergence_{stem}.pdf"
    png_path = out_dir / f"convergence_{stem}.png"
    columns: list[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    epoch = np.asarray([row.get("epoch", int(row.get("step", 0)) + 1) for row in history])
    total = np.asarray([row.get("total_loss", np.nan) for row in history], dtype=float)
    data = np.asarray([row.get("data_loss", np.nan) for row in history], dtype=float)
    shape = np.asarray([
        row.get("shape_loss_weighted", row.get("shape_loss", np.nan))
        for row in history
    ], dtype=float)
    mode = np.asarray([
        row.get("mode_power_loss_weighted", row.get("mode_power_loss", np.nan))
        for row in history
    ], dtype=float)
    binary_weighted = np.asarray([
        row.get(
            "binarization_loss_weighted",
            row.get("binarization_loss_unweighted", np.nan),
        )
        for row in history
    ], dtype=float)
    binary_unweighted = np.asarray([
        row.get("binarization_loss_unweighted", np.nan) for row in history
    ], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(9, 16), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(epoch, total, label="total loss", linewidth=1.6, marker="o", markersize=2.5, markevery=5)
    ax.plot(epoch, data, label="data loss", linestyle="--", linewidth=1.8)
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epoch, shape, label="weighted shape component")
    ax.plot(epoch, mode, label="weighted mode-power component")
    ax.set_ylabel("weighted component")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(epoch, binary_weighted, label="weighted binary component")
    ax.plot(epoch, binary_unweighted, label="unweighted binary penalty", linestyle="--")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_ylabel("binary penalty")
    ax.set_title("Symmetric-log scale; exact zeros remain visible", fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    ax = axes[3, 0]
    ax.plot(epoch, [row.get("mean_shape_score", np.nan) for row in history], label="shape score")
    ax.plot(epoch, [row.get("mean_mode_power", np.nan) for row in history], label="mode power")
    ax.set_ylabel("metric")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[4, 0]
    ax.plot(epoch, [row.get("elapsed_s", np.nan) for row in history], label="elapsed time")
    ax.set_ylabel("time (s)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"csv": str(csv_path), "json": str(json_path), "pdf": str(pdf_path), "png": str(png_path)}

def save_reconstruction(
    reconstruction: dict[str, Any],
    out_dir: Path,
    stem: str,
    plane: ObservationPlane,
    config: AppConfig,
    device: torch.device,
) -> dict[str, str]:
    intensity = reconstruction["intensity"].numpy()
    phase = reconstruction["phase"].numpy()
    n_patterns = intensity.shape[0]
    hologram = sum(intensity[index] * np.exp(2j * index * np.pi / n_patterns) for index in range(n_patterns))
    u, v = reconstruction["u"], reconstruction["v"]
    dx = float((u[1] - u[0]).item())
    dy = float((v[1] - v[0]).item())
    zr = config.reconstruction.fresnel_zr
    if zr is None:
        zr = config.target.distance_to_focus / 2.0
    propagated = {}
    if abs(float(zr)) > 0:
        try:
            propagated["plus"] = fresnel_propagate_fft_convolution(
                hologram, config.dmd.wavelength, dx, dy, float(zr), device=device
            )
            propagated["minus"] = fresnel_propagate_fft_convolution(
                hologram, config.dmd.wavelength, dx, dy, -float(zr), device=device
            )
        except RuntimeError as exc:
            if device.type != "cuda" or not is_out_of_memory(exc):
                raise
            empty_accelerator_cache(str(device))
            cpu = torch.device("cpu")
            propagated["plus"] = fresnel_propagate_fft_convolution(
                hologram, config.dmd.wavelength, dx, dy, float(zr), device=cpu
            )
            propagated["minus"] = fresnel_propagate_fft_convolution(
                hologram, config.dmd.wavelength, dx, dy, -float(zr), device=cpu
            )

    mat_path = out_dir / f"reconstruction_{stem}.mat"
    pdf_path = out_dir / f"reconstruction_{stem}.pdf"
    png_path = out_dir / f"reconstruction_{stem}.png"
    holo_pdf_path = out_dir / f"hologram_{stem}.pdf"
    payload = {"intensity": intensity, "phase": phase, "hologram": hologram, "u": u.numpy(), "v": v.numpy()}
    for key, value in propagated.items():
        payload[f"fresnel_{key}"] = value
    savemat(mat_path, payload, do_compression=True)

    extent_mm = reconstruction["extent_m"] * 1e3
    extent = (-extent_mm, extent_mm, -extent_mm, extent_mm)
    cols = min(max(1, config.reconstruction.max_cols), 2 * n_patterns)
    rows = math.ceil(2 * n_patterns / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.7 * cols, 3.4 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= 2 * n_patterns:
            ax.axis("off"); continue
        pattern = index // 2
        if index % 2:
            image = ax.imshow(phase[pattern], origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"Pattern {pattern}: phase")
        else:
            image = ax.imshow(intensity[pattern], origin="lower", extent=extent, cmap="inferno")
            ax.set_title(f"Pattern {pattern}: intensity")
        ax.add_patch(plt.Circle((0, 0), plane.ring_radius * 1e3, fill=False, linewidth=0.8))
        ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    with PdfPages(holo_pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(np.abs(hologram), origin="lower", extent=extent, cmap="gray")
        axes[0].set_title("Complex hologram magnitude")
        axes[1].imshow(np.angle(hologram), origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1].set_title("Complex hologram phase")
        for ax in axes:
            ax.set_xlabel("u (mm)")
            ax.set_ylabel("v (mm)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fresnel_titles = {
            "plus": "Fresnel forward propagation",
            "minus": "Fresnel backward propagation",
        }
        for key, value in propagated.items():
            title = fresnel_titles.get(key, f"Fresnel {key} propagation")
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(np.abs(value) ** 2, origin="lower", extent=extent, cmap="inferno")
            axes[0].set_title(f"{title} (Intensity)")
            axes[1].imshow(np.angle(value), origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            axes[1].set_title(f"{title} (Phase)")
            for ax in axes:
                ax.set_xlabel("u (mm)")
                ax.set_ylabel("v (mm)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return {"mat": str(mat_path), "pdf": str(pdf_path), "png": str(png_path), "hologram_pdf": str(holo_pdf_path)}


def write_run_metadata(
    out_dir: Path,
    config: AppConfig,
    accelerator,
    result: dict[str, Any],
    files: dict[str, Any],
    reconstruction: dict[str, Any] | None,
) -> None:
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "accelerator": accelerator.__dict__,
        "numerics": {
            "real_dtype": "torch.float32",
            "complex_dtype": "torch.complex64",
            "mixed_precision": False,
        },
        "config": config.to_dict(),
        "active_aperture_pixels": int(result["src_pos"].shape[0]),
        "aperture_sides": int(result["aperture_sides"]),
        "aperture_radius_m": float(result["aperture_radius_m"]),
        "optimization_pixel_chunk": result.get("pixel_chunk"),
        "completed_epochs": int(result.get("completed_epochs", 0)),
        "stopped_early": bool(result.get("stopped_early", False)),
        "resumed_from": result.get("resumed_from"),
        "reconstruction_chunks": None if reconstruction is None else {
            "point_chunk": reconstruction["point_chunk"],
            "pixel_chunk": reconstruction["pixel_chunk"],
        },
        "files": files,
    }
    (out_dir / "run.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "LEAF_FINCH optimization run",
        "=" * 48,
        f"Created: {payload['created']}",
        f"Backend: {accelerator.backend}",
        f"Device: {accelerator.device}",
        f"Device name: {accelerator.name}",
        "Real compute dtype: torch.float32",
        "Complex compute dtype: torch.complex64 (float32 real and imaginary parts)",
        "Mixed precision / autocast: disabled",
        f"Active aperture pixels: {payload['active_aperture_pixels']}",
        f"Optimization pixel chunk: {payload['optimization_pixel_chunk']}",
        f"Completed epochs: {payload['completed_epochs']}",
        f"Stopped early: {payload['stopped_early']}",
        f"Resumed from: {payload['resumed_from']}",
    ]
    history = result.get("history", [])
    if history:
        final = history[-1]
        lines.extend(
            [
                "",
                "Final optimization metrics",
                f"  epoch: {int(final.get('epoch', int(final.get('step', 0)) + 1))}",
                f"  total_loss: {float(final.get('total_loss', float('nan'))):.12g}",
                f"  data_loss: {float(final.get('data_loss', float('nan'))):.12g}",
                f"  shape_loss_unweighted: {float(final.get('shape_loss_unweighted', float('nan'))):.12g}",
                f"  shape_loss_weighted: {float(final.get('shape_loss_weighted', float('nan'))):.12g}",
                f"  mode_power_loss_unweighted: {float(final.get('mode_power_loss_unweighted', float('nan'))):.12g}",
                f"  mode_power_loss_weighted: {float(final.get('mode_power_loss_weighted', float('nan'))):.12g}",
                f"  binarization_loss_unweighted: {float(final.get('binarization_loss_unweighted', float('nan'))):.12g}",
                f"  binarization_loss_weighted: {float(final.get('binarization_loss_weighted', float('nan'))):.12g}",
                f"  mean_shape_score: {float(final.get('mean_shape_score', float('nan'))):.12g}",
                f"  mean_mode_power: {float(final.get('mean_mode_power', float('nan'))):.12g}",
                f"  temperature: {float(final.get('temperature', float('nan'))):.12g}",
                "  Complete per-epoch history: convergence CSV/JSON files",
            ]
        )
    lines.extend(
        [
            "",
            "Full configuration",
            json.dumps(config.to_dict(), indent=2),
        ]
    )
    (out_dir / "info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
