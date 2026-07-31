from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..config import AppConfig
from ..propagation import CancelledError
from ..runner import run_simulation


class SimulationWorker(QObject):
    progress = pyqtSignal(object)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        *,
        model_checkpoint: str | Path | None = None,
        resume_optimizer: bool = True,
    ):
        super().__init__()
        self.config = config
        self.model_checkpoint = None if model_checkpoint is None else str(model_checkpoint)
        self.resume_optimizer = bool(resume_optimizer)
        self.cancel_event = threading.Event()
        self.stop_after_epoch_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = run_simulation(
                self.config,
                cancel_event=self.cancel_event,
                stop_after_epoch_event=self.stop_after_epoch_event,
                progress_callback=self.progress.emit,
                log_callback=self.log.emit,
                model_checkpoint=self.model_checkpoint,
                resume_optimizer=self.resume_optimizer,
            )
        except CancelledError:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)

    @pyqtSlot()
    def cancel(self) -> None:
        self.cancel_event.set()

    @pyqtSlot()
    def stop_after_current_epoch(self) -> None:
        self.stop_after_epoch_event.set()
