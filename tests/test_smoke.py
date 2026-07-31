from pathlib import Path

import torch

from leaf_finch.backend import resolve_accelerator
from leaf_finch.config import AppConfig
from leaf_finch.geometry import plane_from_angles
from leaf_finch.runner import run_simulation


def test_default_geometry():
    cfg = AppConfig()
    plane = plane_from_angles(cfg.plane_angles)
    assert plane.distance > 0
    assert plane.ring_radius > 0
    assert abs(sum(v * v for v in plane.direction) - 1.0) < 1e-6


def test_tiny_cpu_run(tmp_path: Path):
    cfg = AppConfig()
    cfg.backend.device = "cpu"
    cfg.backend.auto_batch = False
    cfg.dmd.nx = 16
    cfg.dmd.ny = 12
    cfg.dmd.aperture_sides = 6
    cfg.optimization.n_patterns = 3
    cfg.optimization.n_steps = 2
    cfg.optimization.points_per_step = 6
    cfg.optimization.pixel_chunk = 64
    cfg.optimization.log_every = 1
    cfg.target.target_type = "siemens"
    cfg.reconstruction.enabled = False
    cfg.output.base_dir = str(tmp_path)
    cfg.validate()
    summary = run_simulation(cfg)
    assert summary["patterns"].shape == (3, 12, 16)
    assert Path(summary["out_dir"]).is_dir()
    assert any(Path(summary["out_dir"]).glob("patterns_*.mat"))


def test_plane_wave_json_roundtrip(tmp_path: Path):
    cfg = AppConfig()
    cfg.incident.wavefront_radius_zi = float("inf")
    path = tmp_path / "plane_wave.json"
    cfg.save_json(path)
    text = path.read_text(encoding="utf-8")
    assert '"inf"' in text
    loaded = AppConfig.load_json(path)
    assert loaded.incident.wavefront_radius_zi == float("inf")


def test_tensor_adam_first_step():
    from leaf_finch.adam import TensorAdam

    value = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    value.grad = torch.tensor([0.25, -0.5], dtype=torch.float64)
    optimizer = TensorAdam(value, lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    optimizer.step()

    # On Adam's first step, m_hat == grad and sqrt(v_hat) == abs(grad).
    expected = torch.tensor([0.9, -1.9], dtype=torch.float64)
    assert torch.allclose(value, expected, rtol=1e-7, atol=1e-7)


def test_project_checkpoint_matches_direct_gradient():
    from leaf_finch.config import DMDConfig
    from leaf_finch.propagation import rs_chunk_field

    torch.manual_seed(3)
    dmd = DMDConfig(nx=4, ny=3, pitch=13.68e-6, wavelength=532e-9)
    points = torch.tensor(
        [[0.0, 0.0, 0.2], [1e-4, -2e-4, 0.2]], dtype=torch.float32
    )
    sources = torch.tensor(
        [[-1e-5, 0.0, 0.0], [1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0]],
        dtype=torch.float32,
    )
    incident = torch.ones(3, dtype=torch.complex64)
    initial = torch.randn(2, 3, dtype=torch.float32)

    direct_patterns = initial.clone().requires_grad_(True)
    direct = rs_chunk_field(
        points, sources, direct_patterns, incident, dmd, 2, use_checkpoint=False
    )
    direct.square().abs().sum().backward()

    checkpoint_patterns = initial.clone().requires_grad_(True)
    checkpointed = rs_chunk_field(
        points, sources, checkpoint_patterns, incident, dmd, 2, use_checkpoint=True
    )
    checkpointed.square().abs().sum().backward()

    assert torch.allclose(direct, checkpointed, rtol=0.0, atol=0.0)
    assert torch.allclose(
        direct_patterns.grad, checkpoint_patterns.grad, rtol=2e-5, atol=2e-5
    )


def test_tensor_adam_state_roundtrip():
    from leaf_finch.adam import TensorAdam

    first = torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=True)
    first.grad = torch.tensor([0.2, -0.4], dtype=torch.float32)
    optimizer = TensorAdam(first, lr=0.03)
    optimizer.step()
    state = optimizer.state_dict(cpu=True)

    second = first.detach().clone().requires_grad_(True)
    restored = TensorAdam(second, lr=0.01)
    restored.load_state_dict(state, keep_current_lr=True)
    assert restored.step_count == optimizer.step_count
    assert restored.lr == 0.01
    assert torch.allclose(restored.exp_avg, optimizer.exp_avg)
    assert torch.allclose(restored.exp_avg_sq, optimizer.exp_avg_sq)


def test_graceful_stop_saves_and_resume_continues(tmp_path: Path):
    import threading

    from leaf_finch.training_state import load_training_checkpoint

    cfg = AppConfig()
    cfg.backend.device = "cpu"
    cfg.backend.auto_batch = False
    cfg.dmd.nx = 16
    cfg.dmd.ny = 12
    cfg.dmd.aperture_sides = 6
    cfg.optimization.n_patterns = 3
    cfg.optimization.n_steps = 3
    cfg.optimization.points_per_step = 6
    cfg.optimization.pixel_chunk = 64
    cfg.optimization.log_every = 1
    cfg.target.target_type = "siemens"
    cfg.reconstruction.enabled = False
    cfg.output.base_dir = str(tmp_path)

    stop_event = threading.Event()

    def request_stop(state):
        if state.get("phase") == "optimization" and state.get("epoch") == 1:
            stop_event.set()

    first = run_simulation(
        cfg,
        stop_after_epoch_event=stop_event,
        progress_callback=request_stop,
    )
    assert first["stopped_early"] is True
    assert first["completed_epochs"] == 1
    assert first["model_path"] is not None
    model_path = Path(first["model_path"])
    assert model_path.is_file()
    saved = load_training_checkpoint(model_path)
    assert saved["completed_epochs"] == 1
    assert len(saved["history"]) == 1

    second = run_simulation(cfg, model_checkpoint=model_path, resume_optimizer=True)
    assert second["stopped_early"] is False
    assert second["completed_epochs"] == 3
    assert len(second["history"]) == 3
    resumed = load_training_checkpoint(second["model_path"])
    assert resumed["completed_epochs"] == 3
    assert len(resumed["history"]) == 3


def test_restart_from_weights_resets_epoch_counter(tmp_path: Path):
    cfg = AppConfig()
    cfg.backend.device = "cpu"
    cfg.backend.auto_batch = False
    cfg.dmd.nx = 16
    cfg.dmd.ny = 12
    cfg.optimization.n_patterns = 3
    cfg.optimization.n_steps = 1
    cfg.optimization.points_per_step = 4
    cfg.optimization.pixel_chunk = 64
    cfg.reconstruction.enabled = False
    cfg.output.base_dir = str(tmp_path)
    first = run_simulation(cfg)

    cfg.optimization.n_steps = 2
    second = run_simulation(
        cfg,
        model_checkpoint=first["model_path"],
        resume_optimizer=False,
    )
    assert second["completed_epochs"] == 2
    assert len(second["history"]) == 2


def test_loss_reports_and_info_txt_include_binary_terms(tmp_path: Path):
    from types import SimpleNamespace

    from leaf_finch.io import save_convergence, write_run_metadata

    history = [
        {
            "step": 0,
            "epoch": 1,
            "total_loss": -0.49995,
            "data_loss": -0.5,
            "shape_loss_unweighted": -0.48,
            "shape_loss_weighted": -0.48,
            "mode_power_loss_unweighted": -1.0,
            "mode_power_loss_weighted": -0.02,
            "binarization_loss_unweighted": 0.25,
            "binarization_loss_weighted": 5e-5,
            "mean_shape_score": 0.48,
            "mean_mode_power": 1.0,
            "temperature": 1.0,
            "elapsed_s": 0.1,
        }
    ]
    files = save_convergence(history, tmp_path, "test")
    assert files is not None
    assert Path(files["pdf"]).is_file()
    assert Path(files["png"]).is_file()

    cfg = AppConfig()
    result = {
        "src_pos": torch.zeros((5, 3)),
        "aperture_sides": 6,
        "aperture_radius_m": 1e-3,
        "pixel_chunk": 64,
        "completed_epochs": 1,
        "stopped_early": False,
        "resumed_from": None,
        "history": history,
    }
    accelerator = SimpleNamespace(backend="cpu", device="cpu", name="test CPU")
    write_run_metadata(tmp_path, cfg, accelerator, result, files={}, reconstruction=None)
    text = (tmp_path / "info.txt").read_text(encoding="utf-8")
    assert "total_loss:" in text
    assert "binarization_loss_unweighted:" in text
    assert "binarization_loss_weighted:" in text
    assert "Real compute dtype: torch.float32" in text
    assert "Complex compute dtype: torch.complex64" in text


def test_rs_hot_path_preserves_float32_complex64():
    from leaf_finch.config import DMDConfig
    from leaf_finch.propagation import rs_chunk_field

    dmd = DMDConfig(nx=4, ny=3, pitch=13.68e-6, wavelength=532e-9)
    points = torch.tensor([[0.0, 0.0, 0.2], [1e-4, -2e-4, 0.2]], dtype=torch.float32)
    sources = torch.tensor(
        [[-1e-5, 0.0, 0.0], [1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0]],
        dtype=torch.float32,
    )
    patterns = torch.ones((2, 3), dtype=torch.float32)
    incident = torch.ones(3, dtype=torch.complex64)
    field = rs_chunk_field(points, sources, patterns, incident, dmd, pixel_chunk=2)
    assert field.dtype == torch.complex64
    assert field.device == points.device


def test_reconstruction_outputs_float32_on_cpu():
    from leaf_finch.geometry import largest_regular_polygon_mask, plane_from_angles
    from leaf_finch.propagation import make_incident_field_fn
    from leaf_finch.reconstruction import reconstruct_field_on_grid

    cfg = AppConfig()
    cfg.backend.device = "cpu"
    cfg.backend.auto_batch = False
    cfg.dmd.nx = 16
    cfg.dmd.ny = 12
    cfg.dmd.use_aperture = False
    cfg.optimization.n_patterns = 3
    cfg.reconstruction.grid_size = 16
    cfg.reconstruction.point_chunk = 32
    cfg.reconstruction.pixel_chunk = 64
    accelerator = resolve_accelerator("cpu")
    device = torch.device("cpu")
    mask, src_pos, _, _ = largest_regular_polygon_mask(cfg.dmd, device, torch.float32)
    patterns = torch.ones((3, cfg.dmd.ny, cfg.dmd.nx), dtype=torch.uint8)
    plane = plane_from_angles(cfg.plane_angles)
    incident_fn = make_incident_field_fn(cfg.dmd.wavelength, cfg.incident.wavefront_radius_zi)
    reconstruction = reconstruct_field_on_grid(
        patterns,
        mask,
        src_pos,
        cfg,
        plane,
        accelerator,
        incident_fn,
    )
    assert reconstruction["intensity"].dtype == torch.float32
    assert reconstruction["phase"].dtype == torch.float32
    assert reconstruction["intensity"].device.type == "cpu"
    assert reconstruction["phase"].device.type == "cpu"
