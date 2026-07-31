from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..backend import list_accelerators
from ..config import (
    AppConfig,
    BackendConfig,
    DMDConfig,
    IncidentConfig,
    OptimizationConfig,
    OutputConfig,
    PlaneAnglesConfig,
    ReconstructionConfig,
    TargetConfig,
)
from ..training_state import checkpoint_summary, load_training_checkpoint
from .worker import SimulationWorker


def double_spin(minimum, maximum, value, decimals=6, step=None, suffix="") -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    if step is not None:
        widget.setSingleStep(step)
    if suffix:
        widget.setSuffix(suffix)
    return widget


def int_spin(minimum, maximum, value, step=1) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setSingleStep(step)
    widget.setKeyboardTracking(False)
    return widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEAF_FINCH")
        self.resize(1220, 820)
        self.thread: QThread | None = None
        self.worker: SimulationWorker | None = None
        self.last_output_dir: Path | None = None
        self.last_model_path: Path | None = None
        self.loaded_model_path: Path | None = None
        self._loss_history: list[dict] = []
        self._loaded_checkpoint_history: list[dict] = []
        self._close_when_finished = False

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.parameters_tab = self._build_parameters_tab()
        self.progress_tab = self._build_progress_tab()
        self.results_tab = self._build_results_tab()
        self.tabs.addTab(self.parameters_tab, "Parameters")
        self.tabs.addTab(self.progress_tab, "Simulation")
        self.tabs.addTab(self.results_tab, "Results")
        self.apply_config(AppConfig())

    def _build_parameters_tab(self) -> QWidget:
        content = QWidget()
        columns = QHBoxLayout(content)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        backend_box = QGroupBox("Accelerator and memory")
        backend_form = QFormLayout(backend_box)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Automatic", "auto")
        for info in list_accelerators():
            self.device_combo.addItem(info.label, info.device)
        self.auto_batch = QCheckBox("Determine chunks from free memory")
        self.memory_fraction = double_spin(0.05, 0.95, 0.65, 2, 0.05)
        self.pixel_chunk = int_spin(64, 1_048_576, 32768, 1024)
        backend_form.addRow("Device", self.device_combo)
        backend_form.addRow("Automatic chunks", self.auto_batch)
        backend_form.addRow("Usable memory fraction", self.memory_fraction)
        backend_form.addRow("Manual optimization chunk", self.pixel_chunk)
        self.auto_batch.toggled.connect(lambda checked: self.pixel_chunk.setEnabled(not checked))
        left.addWidget(backend_box)

        dmd_box = QGroupBox("DMD")
        dmd_form = QFormLayout(dmd_box)
        self.nx = int_spin(8, 16384, 1024, 8)
        self.ny = int_spin(8, 16384, 768, 8)
        self.pitch_um = double_spin(0.01, 1000, 13.68, 6, 0.01, " µm")
        self.wavelength_nm = double_spin(1, 100000, 532, 4, 1, " nm")
        self.use_aperture = QCheckBox("Use regular polygon aperture")
        self.aperture_sides = int_spin(4, 256, 12, 2)
        dmd_form.addRow("Width", self.nx)
        dmd_form.addRow("Height", self.ny)
        dmd_form.addRow("Pixel pitch", self.pitch_um)
        dmd_form.addRow("Wavelength", self.wavelength_nm)
        dmd_form.addRow("Aperture", self.use_aperture)
        dmd_form.addRow("Polygon sides", self.aperture_sides)
        left.addWidget(dmd_box)

        plane_box = QGroupBox("Observation geometry")
        plane_form = QFormLayout(plane_box)
        self.distance = double_spin(1e-5, 1000, 0.275, 8, 0.005, " m")
        self.theta_x = double_spin(-89, 89, 15.0, 6, 0.1, "°")
        self.theta_y = double_spin(-89, 89, 16.4, 6, 0.1, "°")
        self.theta_ring = double_spin(1e-5, 45, 0.15, 6, 0.01, "°")
        self.zi = double_spin(-1e9, 1e9, 1000.0, 6, 1, " m")
        self.plane_wave = QCheckBox("Plane wave (zi = ∞)")
        self.plane_wave.toggled.connect(lambda checked: self.zi.setEnabled(not checked))
        plane_form.addRow("Distance L", self.distance)
        plane_form.addRow("θx", self.theta_x)
        plane_form.addRow("θy", self.theta_y)
        plane_form.addRow("Disk angular radius", self.theta_ring)
        plane_form.addRow("Incident wavefront radius", self.zi)
        plane_form.addRow("Incident field", self.plane_wave)
        left.addWidget(plane_box)
        left.addStretch(1)

        target_box = QGroupBox("Target")
        target_form = QFormLayout(target_box)
        self.target_type = QComboBox()
        for text, value in (
            ("Cosine / quadratic phase", "cosine"),
            ("Spherical waves", "spherical"),
            ("Deterministic two-lens FZP", "fzp"),
            ("Rotated Siemens stars", "siemens"),
        ):
            self.target_type.addItem(text, value)
        self.distance_to_focus = double_spin(-100, 100, 0.025, 8, 0.001, " m")
        self.apodization = double_spin(0, 1, 0.15, 4, 0.01)
        self.siemens_spokes = int_spin(2, 2000, 36)
        target_form.addRow("Type", self.target_type)
        target_form.addRow("Distance to focus", self.distance_to_focus)
        target_form.addRow("Edge apodization fraction", self.apodization)
        target_form.addRow("Siemens spokes", self.siemens_spokes)
        right.addWidget(target_box)

        optim_box = QGroupBox("Optimization")
        optim_form = QFormLayout(optim_box)
        self.n_patterns = int_spin(3, 128, 3)
        self.n_steps = int_spin(1, 1_000_000, 1000, 100)
        self.points_per_step = int_spin(1, 1_000_000, 2048, 128)
        self.learning_rate = double_spin(1e-7, 10, 0.05, 7, 0.005)
        self.shape_weight = double_spin(0, 1000, 1.0, 6, 0.1)
        self.mode_power_weight = double_spin(0, 1000, 0.02, 6, 0.01)
        self.jitter = QCheckBox("Randomize pixel positions each step")
        self.jitter_fraction = double_spin(0, 0.5, 0.5, 3, 0.05)
        self.radial_sampling = double_spin(0, 1, 0.0, 3, 0.1)
        self.seed = int_spin(0, 2_147_483_647, 1)
        optim_form.addRow("Number of patterns", self.n_patterns)
        optim_form.addRow("Steps", self.n_steps)
        optim_form.addRow("Points per step", self.points_per_step)
        optim_form.addRow("Learning rate", self.learning_rate)
        optim_form.addRow("Shape weight", self.shape_weight)
        optim_form.addRow("Mode-power weight", self.mode_power_weight)
        optim_form.addRow("Pixel jitter", self.jitter)
        optim_form.addRow("Jitter half-range / pitch", self.jitter_fraction)
        optim_form.addRow("Radial sampling f", self.radial_sampling)
        optim_form.addRow("Random seed", self.seed)
        right.addWidget(optim_box)

        model_box = QGroupBox("Model checkpoint")
        model_layout = QVBoxLayout(model_box)
        model_row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.model_path.setPlaceholderText("No model loaded; initialize random logits")
        load_model_button = QPushButton("Load model…")
        clear_model_button = QPushButton("Clear")
        load_model_button.clicked.connect(self.load_model_checkpoint)
        clear_model_button.clicked.connect(self.clear_model_checkpoint)
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(load_model_button)
        model_row.addWidget(clear_model_button)
        self.resume_optimizer = QCheckBox("Continue optimizer state, history, and epoch counter")
        self.resume_optimizer.setChecked(True)
        self.model_status = QLabel("A completed or gracefully stopped run saves a portable .pt model automatically.")
        self.model_status.setWordWrap(True)
        model_layout.addLayout(model_row)
        model_layout.addWidget(self.resume_optimizer)
        model_layout.addWidget(self.model_status)
        right.addWidget(model_box)

        recon_box = QGroupBox("Reconstruction")
        recon_form = QFormLayout(recon_box)
        self.reconstruction_enabled = QCheckBox("Reconstruct observation plane")
        self.grid_size = int_spin(16, 4096, 300, 16)
        self.extent_factor = double_spin(0.1, 100, 1.25, 4, 0.05)
        self.fresnel_zr = double_spin(-100, 100, 0.0125, 8, 0.001, " m")
        recon_form.addRow("Enabled", self.reconstruction_enabled)
        recon_form.addRow("Grid size", self.grid_size)
        recon_form.addRow("Extent / disk radius", self.extent_factor)
        recon_form.addRow("Fresnel ±zr", self.fresnel_zr)
        right.addWidget(recon_box)

        output_box = QGroupBox("Output")
        output_layout = QHBoxLayout(output_box)
        self.output_dir = QLineEdit("results")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self.choose_output_dir)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(browse_button)
        right.addWidget(output_box)

        buttons = QHBoxLayout()
        load_button = QPushButton("Load config…")
        save_button = QPushButton("Save config…")
        self.start_button = QPushButton("Start simulation")
        self.start_button.setDefault(True)
        load_button.clicked.connect(self.load_config)
        save_button.clicked.connect(self.save_config)
        self.start_button.clicked.connect(self.start_simulation)
        buttons.addWidget(load_button)
        buttons.addWidget(save_button)
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        right.addLayout(buttons)
        right.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(scroll)
        return wrapper

    def _build_progress_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.phase_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.loss_value_label = QLabel("Loss components will appear after the first epoch.")

        self.loss_figure = Figure(figsize=(8, 6), tight_layout=True)
        self.loss_canvas = FigureCanvas(self.loss_figure)
        grid = self.loss_figure.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.9))
        self.loss_axes_total = self.loss_figure.add_subplot(grid[0, 0])
        self.loss_axes_components = self.loss_figure.add_subplot(
            grid[1, 0], sharex=self.loss_axes_total
        )
        self.loss_axes_binary = self.loss_figure.add_subplot(
            grid[2, 0], sharex=self.loss_axes_total
        )
        self.loss_axes_all = (
            self.loss_axes_total,
            self.loss_axes_components,
            self.loss_axes_binary,
        )

        self.loss_axes_total.set_ylabel("loss")
        self.loss_axes_components.set_ylabel("weighted component")
        self.loss_axes_binary.set_ylabel("binary penalty")
        self.loss_axes_binary.set_xlabel("epoch")
        self.loss_axes_binary.set_yscale("symlog", linthresh=1e-12)
        self.loss_axes_binary.set_title(
            "Binary penalty on symmetric-log scale (exact zeros remain visible)",
            fontsize=9,
        )
        self.loss_axes_total.tick_params(labelbottom=False)
        self.loss_axes_components.tick_params(labelbottom=False)
        for axis in self.loss_axes_all:
            axis.grid(True, alpha=0.3, which="both")

        self.loss_lines = {}
        line, = self.loss_axes_total.plot(
            [], [], label="total loss", linewidth=1.6, marker="o", markersize=2.5,
            markevery=5, zorder=2,
        )
        self.loss_lines["total_loss"] = line
        line, = self.loss_axes_total.plot(
            [], [], label="data loss", linestyle="--", linewidth=1.8, zorder=3,
        )
        self.loss_lines["data_loss"] = line
        line, = self.loss_axes_components.plot(
            [], [], label="weighted shape component", linewidth=1.5,
        )
        self.loss_lines["shape_loss_weighted"] = line
        line, = self.loss_axes_components.plot(
            [], [], label="weighted mode-power component", linewidth=1.5,
        )
        self.loss_lines["mode_power_loss_weighted"] = line
        line, = self.loss_axes_binary.plot(
            [], [], label="weighted binary component", linewidth=1.5,
        )
        self.loss_lines["binarization_loss_weighted"] = line
        line, = self.loss_axes_binary.plot(
            [], [], label="unweighted binary penalty", linestyle="--", linewidth=1.3,
        )
        self.loss_lines["binarization_loss_unweighted"] = line
        for axis in self.loss_axes_all:
            axis.legend(loc="best", fontsize=8)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.loss_canvas)
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        button_row = QHBoxLayout()
        self.stop_epoch_button = QPushButton("Stop after current epoch")
        self.stop_epoch_button.setEnabled(False)
        self.stop_epoch_button.clicked.connect(self.stop_after_current_epoch)
        self.cancel_button = QPushButton("Cancel immediately")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulation)
        button_row.addStretch(1)
        button_row.addWidget(self.stop_epoch_button)
        button_row.addWidget(self.cancel_button)

        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.loss_value_label)
        layout.addWidget(splitter, 1)
        layout.addLayout(button_row)
        return widget

    def _build_results_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        top = QHBoxLayout()
        self.results_path = QLineEdit()
        self.results_path.setReadOnly(True)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(self.open_results_folder)
        self.save_model_button = QPushButton("Save model as…")
        self.save_model_button.setEnabled(False)
        self.save_model_button.clicked.connect(self.save_last_model_as)
        top.addWidget(self.results_path, 1)
        top.addWidget(self.save_model_button)
        top.addWidget(open_folder)
        splitter = QSplitter()
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.preview_selected_file)
        self.file_list.itemDoubleClicked.connect(lambda _: self.open_selected_file())
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        self.image_preview = QLabel("No result selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setScaledContents(False)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.hide()
        self.open_file_button = QPushButton("Open selected file")
        self.open_file_button.clicked.connect(self.open_selected_file)
        preview_layout.addWidget(self.image_preview, 1)
        preview_layout.addWidget(self.text_preview, 1)
        preview_layout.addWidget(self.open_file_button, 0, Qt.AlignRight)
        splitter.addWidget(self.file_list)
        splitter.addWidget(preview_container)
        splitter.setStretchFactor(1, 1)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        return widget

    def collect_config(self) -> AppConfig:
        config = AppConfig(
            backend=BackendConfig(
                device=str(self.device_combo.currentData()),
                auto_batch=self.auto_batch.isChecked(),
                memory_fraction=self.memory_fraction.value(),
            ),
            dmd=DMDConfig(
                nx=self.nx.value(),
                ny=self.ny.value(),
                pitch=self.pitch_um.value() * 1e-6,
                wavelength=self.wavelength_nm.value() * 1e-9,
                aperture_sides=self.aperture_sides.value(),
                use_aperture=self.use_aperture.isChecked(),
            ),
            plane_angles=PlaneAnglesConfig(
                L=self.distance.value(),
                theta_x_deg=self.theta_x.value(),
                theta_y_deg=self.theta_y.value(),
                theta_ring_deg=self.theta_ring.value(),
            ),
            incident=IncidentConfig(
                wavefront_radius_zi=math.inf if self.plane_wave.isChecked() else self.zi.value()
            ),
            optimization=OptimizationConfig(
                n_patterns=self.n_patterns.value(),
                n_steps=self.n_steps.value(),
                points_per_step=self.points_per_step.value(),
                pixel_chunk=None if self.auto_batch.isChecked() else self.pixel_chunk.value(),
                lr=self.learning_rate.value(),
                shape_weight=self.shape_weight.value(),
                mode_power_weight=self.mode_power_weight.value(),
                seed=self.seed.value(),
                dmd_pixel_jitter=self.jitter.isChecked(),
                jitter_fraction=self.jitter_fraction.value(),
                radial_sampling_f=self.radial_sampling.value(),
            ),
            target=TargetConfig(
                target_type=str(self.target_type.currentData()),
                distance_to_focus=self.distance_to_focus.value(),
                edge_apodization_fraction=self.apodization.value(),
                siemens_spokes=self.siemens_spokes.value(),
            ),
            reconstruction=ReconstructionConfig(
                enabled=self.reconstruction_enabled.isChecked(),
                theta_ring_deg=self.theta_ring.value(),
                grid_size=self.grid_size.value(),
                extent_factor=self.extent_factor.value(),
                fresnel_zr=self.fresnel_zr.value(),
            ),
            output=OutputConfig(
                base_dir=self.output_dir.text().strip() or "results",
                save_convergence=True,
                save_reconstruction=True,
            ),
        )
        config.validate()
        return config

    def apply_config(self, config: AppConfig) -> None:
        index = self.device_combo.findData(config.backend.device)
        if index < 0:
            self.device_combo.addItem(config.backend.device, config.backend.device)
            index = self.device_combo.count() - 1
        self.device_combo.setCurrentIndex(index)
        self.auto_batch.setChecked(config.backend.auto_batch)
        self.memory_fraction.setValue(config.backend.memory_fraction)
        self.pixel_chunk.setValue(config.optimization.pixel_chunk or 32768)
        self.nx.setValue(config.dmd.nx); self.ny.setValue(config.dmd.ny)
        self.pitch_um.setValue(config.dmd.pitch * 1e6)
        self.wavelength_nm.setValue(config.dmd.wavelength * 1e9)
        self.use_aperture.setChecked(config.dmd.use_aperture)
        self.aperture_sides.setValue(config.dmd.aperture_sides)
        self.distance.setValue(config.plane_angles.L)
        self.theta_x.setValue(config.plane_angles.theta_x_deg)
        self.theta_y.setValue(config.plane_angles.theta_y_deg)
        self.theta_ring.setValue(config.plane_angles.theta_ring_deg)
        is_plane_wave = not math.isfinite(config.incident.wavefront_radius_zi)
        self.plane_wave.setChecked(is_plane_wave)
        if not is_plane_wave:
            self.zi.setValue(config.incident.wavefront_radius_zi)
        self.target_type.setCurrentIndex(max(0, self.target_type.findData(config.target.target_type)))
        self.distance_to_focus.setValue(config.target.distance_to_focus)
        self.apodization.setValue(config.target.edge_apodization_fraction)
        self.siemens_spokes.setValue(config.target.siemens_spokes)
        self.n_patterns.setValue(config.optimization.n_patterns)
        self.n_steps.setValue(config.optimization.n_steps)
        self.points_per_step.setValue(config.optimization.points_per_step)
        self.learning_rate.setValue(config.optimization.lr)
        self.shape_weight.setValue(config.optimization.shape_weight)
        self.mode_power_weight.setValue(config.optimization.mode_power_weight)
        self.jitter.setChecked(config.optimization.dmd_pixel_jitter)
        self.jitter_fraction.setValue(config.optimization.jitter_fraction)
        self.radial_sampling.setValue(config.optimization.radial_sampling_f)
        self.seed.setValue(config.optimization.seed)
        self.reconstruction_enabled.setChecked(config.reconstruction.enabled)
        self.grid_size.setValue(config.reconstruction.grid_size)
        self.extent_factor.setValue(config.reconstruction.extent_factor)
        zr = config.reconstruction.fresnel_zr
        self.fresnel_zr.setValue(config.target.distance_to_focus / 2 if zr is None else zr)
        self.output_dir.setText(config.output.base_dir)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.apply_config(AppConfig.load_json(path))
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))

    def save_config(self) -> None:
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "config.json", "JSON (*.json)")
        if path:
            config.save_json(path)

    def load_model_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load optimizer model",
            "",
            "LEAF_FINCH model (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        try:
            state = load_training_checkpoint(path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid model checkpoint", str(exc))
            return
        self.loaded_model_path = Path(path)
        self.model_path.setText(path)
        self.model_status.setText(f"Loaded {checkpoint_summary(state)}")
        self._loaded_checkpoint_history = [dict(row) for row in state.get("history", [])]
        self.resume_optimizer.setChecked(True)
        saved_config = state.get("config")
        if isinstance(saved_config, dict):
            answer = QMessageBox.question(
                self,
                "Restore model configuration",
                "Apply the simulation configuration stored with this model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                try:
                    self.apply_config(AppConfig.from_dict(saved_config))
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Configuration not restored",
                        f"The model was loaded, but its saved configuration could not be applied:\n{exc}",
                    )
        self._reset_loss_plot(
            self._loaded_checkpoint_history if self.resume_optimizer.isChecked() else []
        )

    def clear_model_checkpoint(self) -> None:
        self.loaded_model_path = None
        self.model_path.clear()
        self.model_status.setText(
            "A completed or gracefully stopped run saves a portable .pt model automatically."
        )
        self._loaded_checkpoint_history = []

    def save_last_model_as(self) -> None:
        if self.last_model_path is None or not self.last_model_path.is_file():
            QMessageBox.information(self, "No model", "The last run did not produce a model checkpoint.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save model checkpoint",
            self.last_model_path.name,
            "LEAF_FINCH model (*.pt)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".pt":
            destination = destination.with_suffix(".pt")
        try:
            shutil.copy2(self.last_model_path, destination)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save model", str(exc))
            return
        self.model_status.setText(f"Model copied to {destination}")

    def _reset_loss_plot(self, history: list[dict] | None = None) -> None:
        self._loss_history = [dict(row) for row in (history or [])]
        self._redraw_loss_plot()

    def _redraw_loss_plot(self) -> None:
        epochs = [row.get("epoch", int(row.get("step", 0)) + 1) for row in self._loss_history]
        for key, line in self.loss_lines.items():
            line.set_data(epochs, [row.get(key, math.nan) for row in self._loss_history])
        for axis in self.loss_axes_all:
            axis.relim()
            axis.autoscale_view()
        self.loss_canvas.draw_idle()
        if self._loss_history:
            row = self._loss_history[-1]
            self.loss_value_label.setText(
                f"Epoch {int(row.get('epoch', int(row.get('step', 0)) + 1))}: "
                f"total={row.get('total_loss', math.nan):.5e}, "
                f"data={row.get('data_loss', math.nan):.5e}, "
                f"shape={row.get('shape_loss_weighted', math.nan):.5e}, "
                f"mode={row.get('mode_power_loss_weighted', math.nan):.5e}, "
                f"binary(w)={row.get('binarization_loss_weighted', math.nan):.5e}, "
                f"binary(raw)={row.get('binarization_loss_unweighted', math.nan):.5e}"
            )
        else:
            self.loss_value_label.setText("Loss components will appear after the first epoch.")

    def start_simulation(self) -> None:
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid parameters", str(exc))
            return
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("Starting…")
        if self.loaded_model_path is not None and self.resume_optimizer.isChecked():
            self._reset_loss_plot(self._loaded_checkpoint_history)
        else:
            self._reset_loss_plot()
        self.start_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(str(self.target_type.currentData()) != "fzp")
        self.cancel_button.setEnabled(True)
        self.tabs.setCurrentWidget(self.progress_tab)

        self.thread = QThread(self)
        self.worker = SimulationWorker(
            config,
            model_checkpoint=self.loaded_model_path,
            resume_optimizer=self.resume_optimizer.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_worker)
        self.thread.start()

    def stop_after_current_epoch(self) -> None:
        if self.worker is not None:
            self.worker.stop_after_current_epoch()
            self.stop_epoch_button.setEnabled(False)
            self.phase_label.setText("Stop requested; finishing current epoch…")
            self.append_log(
                "Graceful stop requested. The current epoch will finish, then the model and results will be saved."
            )

    def cancel_simulation(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.stop_epoch_button.setEnabled(False)
            self.append_log(
                "Immediate cancellation requested; unlike graceful stop, this may not save the current model."
            )

    def on_progress(self, state: dict) -> None:
        phase = state.get("phase", "")
        fraction = float(state.get("fraction", 0.0))
        if phase == "optimization":
            value = int(750 * fraction)
            epoch = state.get("epoch")
            target_epochs = state.get("target_epochs")
            label = (
                f"Optimizing binary patterns — epoch {epoch}/{target_epochs}"
                if epoch is not None and target_epochs is not None
                else "Optimizing binary patterns"
            )
            if "total_loss" in state:
                row = dict(state)
                row_epoch = int(row.get("epoch", int(row.get("step", 0)) + 1))
                if not self._loss_history or int(
                    self._loss_history[-1].get(
                        "epoch", int(self._loss_history[-1].get("step", 0)) + 1
                    )
                ) < row_epoch:
                    self._loss_history.append(row)
                elif self._loss_history:
                    self._loss_history[-1] = row
                self._redraw_loss_plot()
        elif phase == "reconstruction":
            value = int(750 + 240 * fraction)
            label = "Reconstructing the observation plane"
            self.stop_epoch_button.setEnabled(False)
        elif phase == "finished":
            value = 1000
            label = "Finished"
            self.stop_epoch_button.setEnabled(False)
        elif phase == "stopped":
            value = int(750 * fraction)
            label = f"Stopped cleanly after epoch {state.get('completed_epochs', '')}"
        else:
            value = int(1000 * fraction)
            label = phase.title()
        self.progress_bar.setValue(max(0, min(1000, value)))
        self.phase_label.setText(label)

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def on_finished(self, summary: dict) -> None:
        stopped = bool(summary.get("stopped_early", False))
        self.progress_bar.setValue(750 if stopped else 1000)
        self.phase_label.setText(
            f"Stopped cleanly after epoch {summary.get('completed_epochs', 0)}; model saved"
            if stopped
            else "Finished"
        )
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        self.last_output_dir = Path(summary["out_dir"])
        model_path = summary.get("model_path")
        self.last_model_path = Path(model_path) if model_path else None
        self.save_model_button.setEnabled(
            self.last_model_path is not None and self.last_model_path.is_file()
        )
        if self.last_model_path is not None:
            self.model_status.setText(f"Last saved model: {self.last_model_path}")
        self.populate_results(self.last_output_dir)
        self.tabs.setCurrentWidget(self.results_tab)

    def on_failed(self, traceback_text: str) -> None:
        self.append_log(traceback_text)
        self.phase_label.setText("Failed")
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        QMessageBox.critical(self, "Simulation failed", traceback_text.splitlines()[-1])

    def on_cancelled(self) -> None:
        self.phase_label.setText("Cancelled")
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        self.append_log("Simulation cancelled.")

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_epoch_button.setEnabled(False)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def populate_results(self, directory: Path) -> None:
        self.results_path.setText(str(directory))
        self.file_list.clear()
        for path in sorted(directory.iterdir()):
            if path.is_file():
                item = QListWidgetItem(path.name)
                item.setData(Qt.UserRole, str(path))
                self.file_list.addItem(item)
        if self.file_list.count():
            preferred = next(
                (i for i in range(self.file_list.count()) if self.file_list.item(i).text().startswith("patterns_") and self.file_list.item(i).text().endswith(".png")),
                0,
            )
            self.file_list.setCurrentRow(preferred)

    def preview_selected_file(self, current: QListWidgetItem | None, previous=None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.UserRole))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.text_preview.hide(); self.image_preview.show()
                self.image_preview.setPixmap(
                    pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        if path.suffix.lower() in {".txt", ".json", ".csv"}:
            self.image_preview.hide(); self.text_preview.show()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                self.text_preview.setPlainText(text[:200_000])
            except Exception as exc:
                self.text_preview.setPlainText(str(exc))
        else:
            self.text_preview.hide(); self.image_preview.show()
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText(f"No built-in preview for {path.suffix or 'this file'}\nDouble-click to open it.")

    def open_selected_file(self) -> None:
        item = self.file_list.currentItem()
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))

    def open_results_folder(self) -> None:
        if self.last_output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            answer = QMessageBox.question(
                self,
                "Simulation running",
                "Cancel the simulation and close after the worker stops?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._close_when_finished = True
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.stop_epoch_button.setEnabled(False)
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LEAF_FINCH")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
