# Contributing to LEAF_FINCH

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
# Install a suitable PyTorch build first.
pip install -e .[dev]
python -m pytest
```

## Pull requests

1. Open an issue first for substantial changes to the physical model, file formats, or public API.
2. Keep numerical code device-independent and compatible with CPU, CUDA, and ROCm.
3. Use `torch.float32` for performance-critical real calculations and `torch.complex64` for optical fields unless a change has a documented numerical reason.
4. Add or update tests for behavioral changes.
5. Update the README and LaTeX documentation when user-facing behavior changes.
6. Do not commit generated result directories, virtual environments, caches, or large checkpoints.

## Style

- Use type annotations for new public functions.
- Prefer small modules with explicit responsibilities.
- Avoid hidden CPU/GPU transfers and repeated accelerator synchronization in hot loops.
- Raise descriptive exceptions for invalid geometry and incompatible checkpoints.

By submitting a contribution, you agree that it may be distributed under the repository's MIT License.
