# LEAF-FINCH (Learned Engineering of Amplitude Fields for Fresnel Incoherent Correlation Holography)

Project homepage: https://github.com/rkotynski/leaf_finch

**LEAF-FINCH** designs binary-amplitude masks for a digital micromirror device (DMD) using differentiable Rayleigh–Sommerfeld propagation. The masks are optimized such that the complex field generated in an oblique observation plane that is not parallel to the DMD has a strong projection onto a prescribed real-valued target field. Three or more phase-shifted patterns are optimized jointly.

The program includes a PyQt5 graphical interface, a command-line interface, CPU/CUDA/ROCm execution, automatic GPU-memory-aware chunk sizing, field reconstruction, convergence reports, and resumable checkpoints.

Version: **1.0.0**

Project homepage: [https://github.com/rkotynski/leaf_finch](https://github.com/rkotynski/leaf_finch)

## Features

- Nonparaxial point-to-point Rayleigh-Sommerfeld scalar propagation between non-parallel planes.
- Circular optimization target region in a plane with arbitrary direction and radius.
- Binary DMD masks trained with a straight-through estimator.
- Cosine, spherical-wave, Siemens-star, and deterministic Fresnel-Zone-Plate (FZP) targets. Cosine targets are used in the Opt. Lett. paper. FZP targets with Lee hologram encoding are included as a reference [V. Anand and R. Kotyński, “Experimental full-field Fresnel incoherent correlation holography using a digital micromirror device,” Appl. Opt. 65, 5479–5490 (2026)].
- CPU, NVIDIA CUDA, and AMD ROCm support through PyTorch.
- `torch.float32` real tensors and `torch.complex64` optical fields in performance-critical paths.
- Automatic source-pixel and reconstruction chunk selection from available memory.
- Live loss plots, graceful stop after the current epoch, checkpoint save/load, and exact continuation of the local Adam state.
- Pattern export to MATLAB, NumPy, PDF, and PNG formats.
- Reconstruction and convergence output in MAT, CSV, JSON, PDF, PNG, and TXT formats.

## Relation to Lee holography and to superpixel methods

- In this approach, the diffraction image is directly optimized at a selected angle and distance from the binary-amplitude spatial modulator (DMD). This differs from Lee holography, in which the amplitude hologram is encoded from an original complex field modulated by a linear Lee phase and then projected onto the coding domain. In Lee holography, spatial filtering in the Fourier domain is required to retain only the selected diffraction order. In the present case, spatial filtering may help select one of multiple diffraction orders; however, LEAF-FINCH may also be applied directly without spatial filtering.


- In superpixel methods [S. A. Goorden, J. Bertolotti, and A. P. Mosk, “Superpixel-based spatial amplitude and phase modulation using a digital micromirror device,” Opt. Express 22, 17999–18009 (2014)], the amplitude hologram is obtained from a complex field, as in Lee holography, whereas in our approach it is optimized directly. Direct optimization is considerably more computationally demanding; however, it does not produce aberrations at larger observation angles and enables target fields to be optimized with high angular resolution near the DMD’s specular-reflection angle, where the diffraction efficiency is highest.
 

- The Python code assumes that the distance from the DMD center to the target center ($L$), the curvatures of the interfering waves at the target—characterized by the distance to the focus, equal to $2z_r$—and the target radius are specified for a setup without additional lenses, particularly without magnification. These parameters may be affected by the presence of Fourier lenses and a spatial filter between the DMD and the observation target.


## Documentation

The complete physical model, equations, algorithms, GUI workflow, configuration reference, and output format are described in:

- [`docs/LEAF_FINCH_Documentation.pdf`](docs/LEAF_FINCH_Documentation.pdf)
- [`docs/LEAF_FINCH_Documentation.tex`](docs/LEAF_FINCH_Documentation.tex)

## Screenshots


<table>
<tr>
<td width="50%" align="center">
<img src="docs/assets/screenshots/main_window.png" alt="LEAF_FINCH Parameters tab"><br>
<em>Figure 1. Parameters tab: device selection, DMD geometry, target definition, optimization settings, and output controls.</em>
</td>
<td width="50%" align="center">
<img src="docs/assets/screenshots/live_optimization.png" alt="LEAF_FINCH live optimization tab"><br>
<em>Figure 2. Simulation tab showing optimization progress, the textual log, and the three-panel plot of the total loss and its components.</em>
</td>
</tr>
<tr>
<td width="50%" align="center">
<img src="docs/assets/screenshots/results_browser.png" alt="Generated LEAF_FINCH binary masks"><br>
<em>Figure 3. Generated binary masks encoding three phase-shifted DMD patterns.</em>
</td>
<td width="50%" align="center">
<img src="docs/assets/screenshots/results_reconstructed.png" alt="Reconstructed LEAF_FINCH observation-plane fields"><br>
<em>Figure 4. Reconstructed intensity and phase in the circular off-axis target area. The phase-shifted target images jointly encode a complex-valued FINCH hologram.</em>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="docs/assets/screenshots/finch_psf.png" alt="Fresnel backward propagation of the LEAF_FINCH hologram"><br>
<em>Figure 5. Reconstructed PSF of the complex FINCH hologram obtained by phase-shifted hologram reconstruction followed by Fresnel backward propagation.</em>
</td>
</tr>
</table>


## Installation

Python 3.10 or newer is required. PyTorch is installed separately because CPU, CUDA, and ROCm builds use different packages.

```bash
python -m venv .venv
source .venv/bin/activate

# Install the PyTorch build suitable for this computer first.
# See https://pytorch.org/get-started/locally/

pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
python -m pytest
```

The application intentionally does not use `torch.compile`. The launchers disable TorchDynamo probing to avoid importing optional compiler components that are not required by the numerical method.

## Running the GUI

```bash
leaf-finch-gui
```

The repository also contains a direct launcher:

```bash
python run_gui.py
```

The **Device** selector distinguishes CPU, CUDA, and ROCm. In a ROCm build, PyTorch exposes AMD accelerators through the `torch.cuda` API, so devices retain names such as `cuda:0` internally while the GUI labels the backend as ROCm.

## Command-line interface

```bash
leaf-finch --config examples/default_config.json
leaf-finch --list-devices
leaf-finch --write-default my_config.json
```

Resume an optimization from a checkpoint:

```bash
leaf-finch --config config.json --model model_checkpoint.pt
```

Use saved logits as initialization while resetting Adam, the epoch counter, and the loss history:

```bash
leaf-finch --config config.json --model model_checkpoint.pt --restart-from-weights
```

Equivalent direct launchers are available as `python run_cli.py ...` and `python -m leaf_finch ...`.

## Output

Each run creates a unique directory under the configured output path. Depending on enabled options, it contains:

- `config.json`: exact run configuration;
- `patterns_*.mat`: bit-packed `uint8` DMD masks in big-endian bit order;
- `patterns_*.npz`: unpacked binary masks;
- `patterns_*.pdf` and `patterns_*.png`: mask previews;
- `convergence_*.csv`, `.json`, `.pdf`, `.png`: complete per-epoch history;
- `model_*.pt`: portable logits, Adam state, history, and continuation metadata;
- `reconstruction_*.mat`, `.pdf`, `.png`: observation-plane fields;
- `hologram_*.pdf`: complex hologram and optional Fresnel reconstructions;
- `run.json` and `info.txt`: configuration, device, numerical types, chunk sizes, and final metrics.

## Repository layout

```text
leaf_finch/
  adam.py             local Adam implementation for the optimized logits tensor
  backend.py          accelerator discovery and automatic chunk sizing
  cli.py              command-line interface
  config.py           validated configuration dataclasses and JSON I/O
  geometry.py         DMD aperture and oblique observation-plane geometry
  gui/                PyQt5 interface and worker thread
  io.py               pattern, reconstruction, report, and metadata output
  optimization.py     straight-through binary optimization
  propagation.py      Rayleigh-Sommerfeld and Fresnel propagation
  reconstruction.py   observation-plane field reconstruction
  runner.py           complete simulation orchestration
  targets.py          target modes and deterministic FZP masks
  training_state.py   portable checkpoint save/load

docs/                 PDF documentation and its LaTeX source
examples/             example JSON configuration
tests/                automated tests
```

## Citation

The accompanying manuscript has been provisionally accepted for Optics Letters. Until final bibliographic data are available, cite it as:

> Vijayakumar Anand, Rafał Stojek, and Rafał Kotyński, “Learned binary amplitude mask designs for Fresnel incoherent correlation holography,” *Optics Letters* (2026) (submitted; provisionally accepted).

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff), and a BibTeX entry is provided in [`CITATION.bib`](CITATION.bib).

## Use of AI

Parts of the code and its documentation were developed with the assistance of large language model (LLM) tools.

## Acknowledgement

Vijayakumar Anand acknowledges financial support from the NAWA ULAM project of the Polish National Agency for Academic Exchange (BPN/ULM/2025/1/00097/U/DRAFT/00001).

## License

LEAF_FINCH is released under the [MIT License](LICENSE). This permits academic, commercial, and private use, modification, and redistribution, provided that the copyright and license notices are retained.

## Contributing

Bug reports and contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Security-sensitive reports should follow [`SECURITY.md`](SECURITY.md).
