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
from ..training_state import load_training_checkpoint
from .i18n import LANGUAGES, tr, translate_runtime_message
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
        self.language = "en"
        self._text_bindings: list[tuple[object, str, str]] = []
        self._combo_bindings: list[tuple[QComboBox, object, str]] = []
        self._phase_key = "ready"
        self._phase_args: dict[str, object] = {}
        self._model_status_key = "model_auto_save"
        self._model_status_args: dict[str, object] = {}
        self._log_entries: list[tuple[str, str]] = []

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.parameters_tab = self._build_parameters_tab()
        self.progress_tab = self._build_progress_tab()
        self.results_tab = self._build_results_tab()
        self.tabs.addTab(self.parameters_tab, "")
        self.tabs.addTab(self.progress_tab, "")
        self.tabs.addTab(self.results_tab, "")
        self.retranslate_ui()
        self.apply_config(AppConfig())

    def _t(self, key: str, **kwargs) -> str:
        return tr(self.language, key, **kwargs)

    def _bind_text(self, widget, method: str, key: str):
        getattr(widget, method)(self._t(key))
        self._text_bindings.append((widget, method, key))
        return widget

    def _group_box(self, key: str) -> QGroupBox:
        return self._bind_text(QGroupBox(), "setTitle", key)

    def _check_box(self, key: str) -> QCheckBox:
        return self._bind_text(QCheckBox(), "setText", key)

    def _button(self, key: str) -> QPushButton:
        return self._bind_text(QPushButton(), "setText", key)

    def _add_form_row(self, form: QFormLayout, key: str, field) -> None:
        label = self._bind_text(QLabel(), "setText", key)
        form.addRow(label, field)

    def _add_combo_item(self, combo: QComboBox, key: str, data) -> None:
        combo.addItem(self._t(key), data)
        self._combo_bindings.append((combo, data, key))

    def _set_phase(self, key: str, **kwargs) -> None:
        self._phase_key = key
        self._phase_args = dict(kwargs)
        if hasattr(self, "phase_label"):
            self.phase_label.setText(self._t(key, **kwargs))

    def _set_model_status(self, key: str, **kwargs) -> None:
        self._model_status_key = key
        self._model_status_args = dict(kwargs)
        if hasattr(self, "model_status"):
            self.model_status.setText(self._t(key, **kwargs))

    def _checkpoint_values(self, state: dict) -> dict[str, int]:
        completed = int(state.get("completed_epochs", 0))
        target = int(state.get("target_epochs", completed))
        logits = state.get("logits")
        if getattr(logits, "ndim", 0) == 2:
            rows, cols = (int(v) for v in logits.shape)
        else:
            rows = cols = 0
        return {"completed": completed, "target": target, "rows": rows, "cols": cols}

    def _ask_yes_no(self, title_key: str, text_key: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(self._t(title_key))
        box.setText(self._t(text_key))
        yes_button = box.addButton(self._t("yes"), QMessageBox.YesRole)
        box.addButton(self._t("no"), QMessageBox.NoRole)
        box.setDefaultButton(yes_button)
        box.exec_()
        return box.clickedButton() is yes_button

    def _on_language_changed(self, _index: int = -1) -> None:
        self.language = str(self.language_combo.currentData() or "en")
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        if not hasattr(self, "tabs"):
            return
        self.tabs.setTabText(0, self._t("tab_parameters"))
        self.tabs.setTabText(1, self._t("tab_simulation"))
        self.tabs.setTabText(2, self._t("tab_results"))
        for widget, method, key in self._text_bindings:
            getattr(widget, method)(self._t(key))
        for combo, data, key in self._combo_bindings:
            index = combo.findData(data)
            if index >= 0:
                combo.setItemText(index, self._t(key))
        if hasattr(self, "loss_axes_total"):
            self.loss_axes_total.set_ylabel(self._t("plot_loss"))
            self.loss_axes_components.set_ylabel(self._t("plot_weighted_component"))
            self.loss_axes_binary.set_ylabel(self._t("plot_binary_penalty"))
            self.loss_axes_binary.set_xlabel(self._t("plot_epoch"))
            self.loss_axes_binary.set_title(self._t("plot_binary_title"), fontsize=9)
            labels = {
                "total_loss": "legend_total",
                "data_loss": "legend_data",
                "shape_loss_weighted": "legend_shape",
                "mode_power_loss_weighted": "legend_mode",
                "binarization_loss_weighted": "legend_binary_weighted",
                "binarization_loss_unweighted": "legend_binary_raw",
            }
            for name, key in labels.items():
                self.loss_lines[name].set_label(self._t(key))
            for axis in self.loss_axes_all:
                axis.legend(loc="best", fontsize=8)
            self.loss_canvas.draw_idle()
        self._set_phase(self._phase_key, **self._phase_args)
        self._set_model_status(self._model_status_key, **self._model_status_args)
        if hasattr(self, "_loss_history"):
            self._redraw_loss_plot()
        if hasattr(self, "log_view"):
            self._render_log_history()
        if hasattr(self, "file_list"):
            current = self.file_list.currentItem()
            if current is not None:
                self.preview_selected_file(current)
            elif hasattr(self, "image_preview"):
                self.image_preview.setText(self._t("no_result_selected"))

    def _build_parameters_tab(self) -> QWidget:
        content = QWidget()
        columns = QHBoxLayout(content)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        backend_box = self._group_box("group_backend")
        backend_form = QFormLayout(backend_box)
        self.device_combo = QComboBox()
        self._add_combo_item(self.device_combo, "automatic", "auto")
        for info in list_accelerators():
            self.device_combo.addItem(info.label, info.device)
        self.language_combo = QComboBox()
        for key, code in LANGUAGES:
            self._add_combo_item(self.language_combo, key, code)
        self.language_combo.setCurrentIndex(0)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.auto_batch = self._check_box("determine_chunks")
        self.memory_fraction = double_spin(0.05, 0.95, 0.65, 2, 0.05)
        self.pixel_chunk = int_spin(64, 1_048_576, 32768, 1024)
        self._add_form_row(backend_form, "interface_language", self.language_combo)
        self._add_form_row(backend_form, "device", self.device_combo)
        self._add_form_row(backend_form, "automatic_chunks", self.auto_batch)
        self._add_form_row(backend_form, "usable_memory_fraction", self.memory_fraction)
        self._add_form_row(backend_form, "manual_optimization_chunk", self.pixel_chunk)
        self.auto_batch.toggled.connect(lambda checked: self.pixel_chunk.setEnabled(not checked))
        left.addWidget(backend_box)

        dmd_box = self._group_box("group_dmd")
        dmd_form = QFormLayout(dmd_box)
        self.nx = int_spin(8, 16384, 1024, 8)
        self.ny = int_spin(8, 16384, 768, 8)
        self.pitch_um = double_spin(0.01, 1000, 13.68, 6, 0.01, " µm")
        self.wavelength_nm = double_spin(1, 100000, 532, 4, 1, " nm")
        self.use_aperture = self._check_box("use_polygon_aperture")
        self.aperture_sides = int_spin(4, 256, 12, 2)
        self._add_form_row(dmd_form, "width", self.nx)
        self._add_form_row(dmd_form, "height", self.ny)
        self._add_form_row(dmd_form, "pixel_pitch", self.pitch_um)
        self._add_form_row(dmd_form, "wavelength", self.wavelength_nm)
        self._add_form_row(dmd_form, "aperture", self.use_aperture)
        self._add_form_row(dmd_form, "polygon_sides", self.aperture_sides)
        left.addWidget(dmd_box)

        plane_box = self._group_box("group_geometry")
        plane_form = QFormLayout(plane_box)
        self.distance = double_spin(1e-5, 1000, 0.275, 8, 0.005, " m")
        self.theta_x = double_spin(-89, 89, 15.0, 6, 0.1, "°")
        self.theta_y = double_spin(-89, 89, 16.4, 6, 0.1, "°")
        self.theta_ring = double_spin(1e-5, 45, 0.15, 6, 0.01, "°")
        self.zi = double_spin(-1e9, 1e9, 1000.0, 6, 1, " m")
        self.plane_wave = self._check_box("plane_wave")
        self.plane_wave.toggled.connect(lambda checked: self.zi.setEnabled(not checked))
        self._add_form_row(plane_form, "distance_l", self.distance)
        self._add_form_row(plane_form, "theta_x", self.theta_x)
        self._add_form_row(plane_form, "theta_y", self.theta_y)
        self._add_form_row(plane_form, "disk_angular_radius", self.theta_ring)
        self._add_form_row(plane_form, "incident_wavefront_radius", self.zi)
        self._add_form_row(plane_form, "incident_field", self.plane_wave)
        left.addWidget(plane_box)
        left.addStretch(1)

        target_box = self._group_box("group_target")
        target_form = QFormLayout(target_box)
        self.target_type = QComboBox()
        for key, value in (
            ("target_cosine", "cosine"),
            ("target_spherical", "spherical"),
            ("target_fzp", "fzp"),
            ("target_siemens", "siemens"),
        ):
            self._add_combo_item(self.target_type, key, value)
        self.distance_to_focus = double_spin(-100, 100, 0.025, 8, 0.001, " m")
        self.apodization = double_spin(0, 1, 0.15, 4, 0.01)
        self.siemens_spokes = int_spin(2, 2000, 36)
        self._add_form_row(target_form, "type", self.target_type)
        self._add_form_row(target_form, "distance_to_focus", self.distance_to_focus)
        self._add_form_row(target_form, "edge_apodization_fraction", self.apodization)
        self._add_form_row(target_form, "siemens_spokes", self.siemens_spokes)
        right.addWidget(target_box)

        optim_box = self._group_box("group_optimization")
        optim_form = QFormLayout(optim_box)
        self.n_patterns = int_spin(3, 128, 3)
        self.n_steps = int_spin(1, 1_000_000, 1000, 100)
        self.points_per_step = int_spin(1, 1_000_000, 2048, 128)
        self.learning_rate = double_spin(1e-7, 10, 0.05, 7, 0.005)
        self.shape_weight = double_spin(0, 1000, 1.0, 6, 0.1)
        self.mode_power_weight = double_spin(0, 1000, 0.02, 6, 0.01)
        self.jitter = self._check_box("randomize_pixels")
        self.jitter_fraction = double_spin(0, 0.5, 0.5, 3, 0.05)
        self.radial_sampling = double_spin(0, 1, 0.0, 3, 0.1)
        self.seed = int_spin(0, 2_147_483_647, 1)
        self._add_form_row(optim_form, "number_of_patterns", self.n_patterns)
        self._add_form_row(optim_form, "steps", self.n_steps)
        self._add_form_row(optim_form, "points_per_step", self.points_per_step)
        self._add_form_row(optim_form, "learning_rate", self.learning_rate)
        self._add_form_row(optim_form, "shape_weight", self.shape_weight)
        self._add_form_row(optim_form, "mode_power_weight", self.mode_power_weight)
        self._add_form_row(optim_form, "pixel_jitter", self.jitter)
        self._add_form_row(optim_form, "jitter_half_range", self.jitter_fraction)
        self._add_form_row(optim_form, "radial_sampling_f", self.radial_sampling)
        self._add_form_row(optim_form, "random_seed", self.seed)
        right.addWidget(optim_box)

        model_box = self._group_box("group_checkpoint")
        model_layout = QVBoxLayout(model_box)
        model_row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self._bind_text(self.model_path, "setPlaceholderText", "model_placeholder")
        load_model_button = self._button("load_model")
        clear_model_button = self._button("clear")
        load_model_button.clicked.connect(self.load_model_checkpoint)
        clear_model_button.clicked.connect(self.clear_model_checkpoint)
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(load_model_button)
        model_row.addWidget(clear_model_button)
        self.resume_optimizer = self._check_box("resume_optimizer")
        self.resume_optimizer.setChecked(True)
        self.model_status = QLabel()
        self._set_model_status("model_auto_save")
        self.model_status.setWordWrap(True)
        model_layout.addLayout(model_row)
        model_layout.addWidget(self.resume_optimizer)
        model_layout.addWidget(self.model_status)
        right.addWidget(model_box)

        recon_box = self._group_box("group_reconstruction")
        recon_form = QFormLayout(recon_box)
        self.reconstruction_enabled = self._check_box("reconstruct_plane")
        self.grid_size = int_spin(16, 4096, 300, 16)
        self.extent_factor = double_spin(0.1, 100, 1.25, 4, 0.05)
        self.fresnel_zr = double_spin(-100, 100, 0.0125, 8, 0.001, " m")
        self._add_form_row(recon_form, "enabled", self.reconstruction_enabled)
        self._add_form_row(recon_form, "grid_size", self.grid_size)
        self._add_form_row(recon_form, "extent_disk_radius", self.extent_factor)
        self._add_form_row(recon_form, "fresnel_zr", self.fresnel_zr)
        right.addWidget(recon_box)

        output_box = self._group_box("group_output")
        output_layout = QHBoxLayout(output_box)
        self.output_dir = QLineEdit("results")
        browse_button = self._button("browse")
        browse_button.clicked.connect(self.choose_output_dir)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(browse_button)
        right.addWidget(output_box)

        buttons = QHBoxLayout()
        load_button = self._button("load_config")
        save_button = self._button("save_config")
        self.start_button = self._button("start_simulation")
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
        self.phase_label = QLabel()
        self._set_phase("ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.loss_value_label = QLabel(self._t("loss_wait"))

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

        self.loss_axes_total.set_ylabel(self._t("plot_loss"))
        self.loss_axes_components.set_ylabel(self._t("plot_weighted_component"))
        self.loss_axes_binary.set_ylabel(self._t("plot_binary_penalty"))
        self.loss_axes_binary.set_xlabel(self._t("plot_epoch"))
        self.loss_axes_binary.set_yscale("symlog", linthresh=1e-12)
        self.loss_axes_binary.set_title(
            self._t("plot_binary_title"),
            fontsize=9,
        )
        self.loss_axes_total.tick_params(labelbottom=False)
        self.loss_axes_components.tick_params(labelbottom=False)
        for axis in self.loss_axes_all:
            axis.grid(True, alpha=0.3, which="both")

        self.loss_lines = {}
        line, = self.loss_axes_total.plot(
            [], [], label=self._t("legend_total"), linewidth=1.6, marker="o", markersize=2.5,
            markevery=5, zorder=2,
        )
        self.loss_lines["total_loss"] = line
        line, = self.loss_axes_total.plot(
            [], [], label=self._t("legend_data"), linestyle="--", linewidth=1.8, zorder=3,
        )
        self.loss_lines["data_loss"] = line
        line, = self.loss_axes_components.plot(
            [], [], label=self._t("legend_shape"), linewidth=1.5,
        )
        self.loss_lines["shape_loss_weighted"] = line
        line, = self.loss_axes_components.plot(
            [], [], label=self._t("legend_mode"), linewidth=1.5,
        )
        self.loss_lines["mode_power_loss_weighted"] = line
        line, = self.loss_axes_binary.plot(
            [], [], label=self._t("legend_binary_weighted"), linewidth=1.5,
        )
        self.loss_lines["binarization_loss_weighted"] = line
        line, = self.loss_axes_binary.plot(
            [], [], label=self._t("legend_binary_raw"), linestyle="--", linewidth=1.3,
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
        self.stop_epoch_button = self._button("stop_after_epoch")
        self.stop_epoch_button.setEnabled(False)
        self.stop_epoch_button.clicked.connect(self.stop_after_current_epoch)
        self.cancel_button = self._button("cancel_immediately")
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
        open_folder = self._button("open_folder")
        open_folder.clicked.connect(self.open_results_folder)
        self.save_model_button = self._button("save_model_as")
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
        self.image_preview = QLabel(self._t("no_result_selected"))
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setScaledContents(False)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.hide()
        self.open_file_button = self._button("open_selected_file")
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
        path = QFileDialog.getExistingDirectory(self, self._t("select_output_directory"), self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self._t("load_configuration"), "", "JSON (*.json)")
        if not path:
            return
        try:
            self.apply_config(AppConfig.load_json(path))
        except Exception as exc:
            QMessageBox.critical(self, self._t("invalid_configuration"), str(exc))

    def save_config(self) -> None:
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, self._t("invalid_configuration"), str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, self._t("save_configuration"), "config.json", "JSON (*.json)")
        if path:
            config.save_json(path)

    def load_model_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._t("load_optimizer_model"),
            "",
            self._t("model_filter"),
        )
        if not path:
            return
        try:
            state = load_training_checkpoint(path)
        except Exception as exc:
            QMessageBox.critical(self, self._t("invalid_model_checkpoint"), str(exc))
            return
        self.loaded_model_path = Path(path)
        self.model_path.setText(path)
        self._set_model_status("loaded_model", **self._checkpoint_values(state))
        self._loaded_checkpoint_history = [dict(row) for row in state.get("history", [])]
        self.resume_optimizer.setChecked(True)
        saved_config = state.get("config")
        if isinstance(saved_config, dict):
            if self._ask_yes_no("restore_model_configuration", "restore_model_configuration_question"):
                try:
                    self.apply_config(AppConfig.from_dict(saved_config))
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        self._t("configuration_not_restored"),
                        self._t("configuration_not_restored_detail", error=exc),
                    )
        self._reset_loss_plot(
            self._loaded_checkpoint_history if self.resume_optimizer.isChecked() else []
        )

    def clear_model_checkpoint(self) -> None:
        self.loaded_model_path = None
        self.model_path.clear()
        self._set_model_status("model_auto_save")
        self._loaded_checkpoint_history = []

    def save_last_model_as(self) -> None:
        if self.last_model_path is None or not self.last_model_path.is_file():
            QMessageBox.information(self, self._t("no_model"), self._t("no_model_detail"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._t("save_model_checkpoint"),
            self.last_model_path.name,
            self._t("model_save_filter"),
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".pt":
            destination = destination.with_suffix(".pt")
        try:
            shutil.copy2(self.last_model_path, destination)
        except Exception as exc:
            QMessageBox.critical(self, self._t("could_not_save_model"), str(exc))
            return
        self._set_model_status("model_copied", path=destination)

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
                self._t(
                    "loss_summary",
                    epoch=int(row.get("epoch", int(row.get("step", 0)) + 1)),
                    total=row.get("total_loss", math.nan),
                    data=row.get("data_loss", math.nan),
                    shape=row.get("shape_loss_weighted", math.nan),
                    mode=row.get("mode_power_loss_weighted", math.nan),
                    binary_w=row.get("binarization_loss_weighted", math.nan),
                    binary_raw=row.get("binarization_loss_unweighted", math.nan),
                )
            )
        else:
            self.loss_value_label.setText(self._t("loss_wait"))

    def start_simulation(self) -> None:
        try:
            config = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, self._t("invalid_parameters"), str(exc))
            return
        self._log_entries.clear()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self._set_phase("starting")
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
            self._set_phase("stop_requested")
            self.append_log_key("graceful_stop_log")

    def cancel_simulation(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.stop_epoch_button.setEnabled(False)
            self.append_log_key("cancel_log")

    def on_progress(self, state: dict) -> None:
        phase = state.get("phase", "")
        fraction = float(state.get("fraction", 0.0))
        if phase == "optimization":
            value = int(750 * fraction)
            epoch = state.get("epoch")
            target_epochs = state.get("target_epochs")
            if epoch is not None and target_epochs is not None:
                phase_key = "optimizing_epoch"
                phase_args = {"epoch": epoch, "target": target_epochs}
            else:
                phase_key = "optimizing"
                phase_args = {}
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
            phase_key = "reconstructing"
            phase_args = {}
            self.stop_epoch_button.setEnabled(False)
        elif phase == "finished":
            value = 1000
            phase_key = "finished"
            phase_args = {}
            self.stop_epoch_button.setEnabled(False)
        elif phase == "stopped":
            value = int(750 * fraction)
            phase_key = "stopped_after"
            phase_args = {"epoch": state.get("completed_epochs", "")}
        else:
            value = int(1000 * fraction)
            phase_key = "finished" if phase == "finished" else "ready"
            phase_args = {}
        self.progress_bar.setValue(max(0, min(1000, value)))
        self._set_phase(phase_key, **phase_args)

    def _render_log_history(self) -> None:
        if not hasattr(self, "log_view"):
            return
        self.log_view.clear()
        for kind, value in self._log_entries:
            if kind == "key":
                message = self._t(value)
            else:
                message = translate_runtime_message(value, self.language)
            self.log_view.appendPlainText(message)

    def append_log(self, text: str) -> None:
        self._log_entries.append(("runtime", text))
        self.log_view.appendPlainText(translate_runtime_message(text, self.language))

    def append_log_key(self, key: str) -> None:
        self._log_entries.append(("key", key))
        self.log_view.appendPlainText(self._t(key))

    def on_finished(self, summary: dict) -> None:
        stopped = bool(summary.get("stopped_early", False))
        self.progress_bar.setValue(750 if stopped else 1000)
        if stopped:
            self._set_phase("stopped_saved", epoch=summary.get("completed_epochs", 0))
        else:
            self._set_phase("finished")
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        self.last_output_dir = Path(summary["out_dir"])
        model_path = summary.get("model_path")
        self.last_model_path = Path(model_path) if model_path else None
        self.save_model_button.setEnabled(
            self.last_model_path is not None and self.last_model_path.is_file()
        )
        if self.last_model_path is not None:
            self._set_model_status("last_saved_model", path=self.last_model_path)
        self.populate_results(self.last_output_dir)
        self.tabs.setCurrentWidget(self.results_tab)

    def on_failed(self, traceback_text: str) -> None:
        self.append_log(traceback_text)
        self._set_phase("failed")
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        QMessageBox.critical(self, self._t("simulation_failed"), traceback_text.splitlines()[-1])

    def on_cancelled(self) -> None:
        self._set_phase("cancelled")
        self.cancel_button.setEnabled(False)
        self.stop_epoch_button.setEnabled(False)
        self.append_log_key("simulation_cancelled")

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
            self.image_preview.setText(self._t("no_preview", suffix=path.suffix or self._t("this_file")))

    def open_selected_file(self) -> None:
        item = self.file_list.currentItem()
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))

    def open_results_folder(self) -> None:
        if self.last_output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            if not self._ask_yes_no("simulation_running", "close_running_question"):
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
