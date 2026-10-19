"""Embedded YAML-driven TiSAM training/testing and inference pages.

WorkflowSwitcher mounts this widget beside the untouched CNN form. A QProcess
runs tisam_pipeline; native Mammoth events provide training progress. The existing
TiSAM result viewer is embedded instead of opening another model window.
"""

from __future__ import annotations

import codecs
import json
import sys
import tempfile
from pathlib import Path

import yaml
from mammoth.core.events import ExecutionEventTailReader
from PySide6 import QtCore, QtGui, QtWidgets

from base.models.tisam_model import TiSAMModel
from base.models.tisam_workflow import INPUT_NAMES, load_workflow

from .segmentation_result import SegmentationResult
from .tisam_classification import TiSAMClassificationWindow


class PathField(QtWidgets.QWidget):
    """One aligned, editable path with a standard file/directory picker."""

    changed = QtCore.Signal()

    def __init__(self, title, *, file=False, parent=None):
        super().__init__(parent)
        self.title, self.file = title, file
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText(title)
        self.edit.editingFinished.connect(self.changed)
        layout.addWidget(self.edit, 1)
        button = QtWidgets.QPushButton("Browse…")
        button.clicked.connect(self.browse)
        layout.addWidget(button)

    def browse(self):
        if self.file:
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, self.title, self.edit.text(), "YAML (*.yml *.yaml)"
            )
        else:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, self.title, self.edit.text()
            )
        if selected:
            self.edit.setText(selected)
            self.changed.emit()


class TiSAMPage(QtWidgets.QWidget):
    """Own the complete TiSAM lifecycle without mutating CNN UI state."""

    busy_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = self.viewer = self.config = None
        self.request_directory = None
        self.output = None
        self.stopped = self.failed = False
        self.result_config = None
        self.event_reader = None
        self.previous_streams = set()
        self.pending_text = ""
        self.native_timer = QtCore.QTimer(self)
        self.native_timer.setInterval(1000)
        self.native_timer.timeout.connect(self.read_training_events)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.setup_pages = []
        for name in ("Data", "Model", "Classes"):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            page = QtWidgets.QWidget()
            scroll.setWidget(page)
            self.setup_pages.append(page)
            self.tabs.addTab(scroll, name)

        form = QtWidgets.QFormLayout(self.setup_pages[0])
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(12)
        self.yaml = PathField("TiSAM workflow YAML", file=True)
        self.yaml.changed.connect(self.load_config)
        form.addRow("Configuration", self.yaml)
        self.fields = {}
        for key, label in zip(
            INPUT_NAMES,
            (
                "Training images",
                "Training annotations",
                "Testing images",
                "Testing annotations",
            ),
        ):
            field = PathField(label)
            field.edit.textChanged.connect(self.invalidate_check)
            self.fields[key] = field
            form.addRow(label, field)
        hint = QtWidgets.QLabel(
            "Finds NDPI files in training/testing image folders and matching Aperio XML files in annotation folders."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        model_layout = QtWidgets.QVBoxLayout(self.setup_pages[1])
        model_form = QtWidgets.QFormLayout()
        model_form.setVerticalSpacing(12)
        model_form.setHorizontalSpacing(18)
        self.model_values = {}
        for name in (
            "Model name",
            "Encoder",
            "Decoder",
            "Input / output tiles",
            "Device",
            "Precision",
            "Epochs",
            "Batch / accumulation",
            "Learning rate",
            "Loss",
            "Downsample",
            "Validation split",
            "Output folder",
        ):
            value = QtWidgets.QLabel("—")
            value.setWordWrap(True)
            value.setTextFormat(QtCore.Qt.PlainText)
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.model_values[name] = value
            model_form.addRow(name, value)
        model_layout.addLayout(model_form)
        model_layout.addStretch()

        classes_layout = QtWidgets.QVBoxLayout(self.setup_pages[2])
        self.classes_note = QtWidgets.QLabel(
            "Colors from YAML. Black → 0 (void)."
        )
        self.classes_note.setWordWrap(True)
        self.classes_note.setTextFormat(QtCore.Qt.PlainText)
        class_content = QtWidgets.QHBoxLayout()
        self.class_table = QtWidgets.QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["ID", "Class", "YAML color (RGB)"])
        self.class_table.verticalHeader().hide()
        self.class_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.class_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        header = self.class_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        class_content.addWidget(self.class_table, 1)
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.hide()
        class_content.addWidget(self.preview)
        classes_layout.addLayout(class_content, 1)
        classes_layout.addWidget(self.classes_note)

        self.training_controls = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self.training_controls)
        body.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.training_controls)
        actions = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("Check data")
        self.back_button = QtWidgets.QPushButton("Back")
        self.continue_button = QtWidgets.QPushButton("Continue")
        self.start_button = QtWidgets.QPushButton("Start training and testing")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.check_button.clicked.connect(lambda: self.start("check"))
        self.back_button.clicked.connect(
            lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex() - 1)
        )
        self.continue_button.clicked.connect(self.continue_setup)
        self.start_button.clicked.connect(lambda: self.start("run"))
        self.stop_button.clicked.connect(self.stop)
        actions.addWidget(self.check_button)
        actions.addStretch()
        for button in (
            self.back_button,
            self.continue_button,
            self.start_button,
            self.stop_button,
        ):
            actions.addWidget(button)
        body.addLayout(actions)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        body.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        body.addWidget(self.progress)
        self.metrics_status = QtWidgets.QLabel()
        self.metrics_status.setWordWrap(True)
        body.addWidget(self.metrics_status)
        self.details = QtWidgets.QToolButton()
        self.details.setText("Show detailed logs")
        self.details.setCheckable(True)
        body.addWidget(self.details)
        self.logs = QtWidgets.QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2000)
        self.logs.setMaximumHeight(180)
        self.logs.hide()
        self.details.toggled.connect(self.logs.setVisible)
        body.addWidget(self.logs)
        results = QtWidgets.QHBoxLayout()
        self.results_button = QtWidgets.QPushButton("View test results")
        self.segment_button = QtWidgets.QPushButton("Use model for segmentation")
        self.folder_button = QtWidgets.QPushButton("Open output folder")
        for button in (self.results_button, self.segment_button, self.folder_button):
            button.setEnabled(False)
            results.addWidget(button)
        self.results_button.clicked.connect(self.open_results)
        self.segment_button.clicked.connect(self.use_trained_model)
        self.folder_button.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self.output))
            )
        )
        body.addLayout(results)
        self.inference = QtWidgets.QWidget()
        self.inference_layout = QtWidgets.QVBoxLayout(self.inference)
        self.inference_index = self.tabs.addTab(
            self.inference, "Segmentation and viewer"
        )
        self.model_yaml = PathField("Workflow or saved model YAML", file=True)
        self.model_yaml.changed.connect(self.load_viewer)
        self.inference_layout.addWidget(self.model_yaml)
        self.inference_hint = QtWidgets.QLabel(
            "Load the same workflow YAML after training, or a saved TiSAM model YAML. Colors always come from YAML."
        )
        self.inference_hint.setWordWrap(True)
        self.inference_layout.addWidget(self.inference_hint)
        self.inference_spacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.inference_layout.addItem(self.inference_spacer)
        self.tabs.currentChanged.connect(self.update_navigation)
        self.reset_run_status()
        self.update_navigation()

    def show_status(self, message):
        """Show actionable errors and worker updates without idle setup messages."""
        self.status.setText(message)
        self.status.setVisible(bool(message))

    def reset_run_status(self):
        """Keep the configuration pages free of unused progress and log controls."""
        self.show_status("")
        self.progress.hide()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.metrics_status.clear()
        self.metrics_status.hide()
        self.details.setChecked(False)
        self.details.hide()
        self.logs.clear()

    def update_navigation(self):
        """Keep review navigation and run actions attached to the relevant setup page."""
        index = self.tabs.currentIndex()
        busy = self.is_busy()
        self.training_controls.setVisible(index != self.inference_index)
        self.back_button.setEnabled(not busy and index > 0)
        self.continue_button.setVisible(index < 2)
        self.continue_button.setEnabled(not busy)
        self.start_button.setVisible(index == 2)
        self.stop_button.setVisible(self.process is not None)

    def continue_setup(self):
        """Review Data, Model and Classes in order without changing YAML settings."""
        index = self.tabs.currentIndex()
        if self.config is None:
            self.tabs.setCurrentIndex(0)
            self.show_status("Select a valid workflow YAML before continuing.")
            return
        if index == 0:
            for field in self.fields.values():
                value = field.edit.text().strip()
                if not value or not Path(value).expanduser().is_dir():
                    self.show_status(f"Select an existing directory for {field.title}.")
                    field.edit.setFocus()
                    return
        self.tabs.setCurrentIndex(index + 1)

    def invalidate_check(self):
        if self.process is None:
            self.reset_run_status()
            self.preview.hide()

    def load_config(self):
        """Load the selected YAML once for review and subsequent worker requests."""
        self.reset_run_status()
        self.preview.hide()
        try:
            config = load_workflow(self.yaml.edit.text())
            self.config = config
            native = config.training
            self.output = config.directory
            saved = self.output / "workflow.yml"
            self.result_config = str(saved) if saved.is_file() else None
            for key, value in config.paths.items():
                self.fields[key].edit.setText(str(value))
            values = {
                "Model name": native.name,
                "Encoder": native.model.extra_encoder,
                "Decoder": f"{native.model.num_layers} layers · {native.model.total_queries} queries",
                "Input / output tiles": f"{native.model.input_hw} → {native.model.output_hw}",
                "Device": native.device,
                "Precision": native.precision,
                "Epochs": native.epochs,
                "Batch / accumulation": f"{native.batch_size} / {native.accumulation_steps}",
                "Learning rate": native.learning_rate,
                "Loss": native.loss,
                "Downsample": f"{config.preprocessing.downsample:g}×",
                "Validation split": f"{config.preprocessing.validation_fraction:.0%} of training slides",
                "Output folder": config.directory,
            }
            for name, value in values.items():
                self.model_values[name].setText(str(value))
            self.class_table.setRowCount(len(native.data.class_names))
            for row, name in enumerate(native.data.class_names):
                color = config.colormap[name]
                items = [str(row), name, ", ".join(map(str, color))]
                for column, text in enumerate(items):
                    self.class_table.setItem(
                        row, column, QtWidgets.QTableWidgetItem(text)
                    )
                item = self.class_table.item(row, 2)
                item.setBackground(QtGui.QColor(*color))
                r, g, b = color
                item.setForeground(
                    QtGui.QColor(
                        "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"
                    )
                )
            self.classes_note.setText("Colors from YAML. Black → 0 (void).")
            self.set_busy(False)
            return True
        except Exception as error:
            self.config = None
            self.result_config = None
            self.output = None
            self.preview.hide()
            self.set_busy(False)
            for value in self.model_values.values():
                value.setText("—")
            self.class_table.setRowCount(0)
            self.classes_note.setText("Select a valid workflow YAML on the Data tab.")
            self.show_status(str(error))
            return False

    def start(self, mode):
        if self.is_busy():
            return
        try:
            config = self.config
            if config is None:
                raise ValueError("Select a valid workflow YAML before starting")
            paths = {
                key: field.edit.text().strip() for key, field in self.fields.items()
            }
            if not all(paths.values()):
                raise ValueError("Select all four training and testing directories")
        except Exception as error:
            self.show_status(str(error))
            return
        self.config, self.output = config, config.directory
        if self.request_directory:
            self.request_directory.cleanup()
        self.request_directory = tempfile.TemporaryDirectory(prefix="codavision-tisam-")
        snapshot = Path(self.request_directory.name) / "workflow.yml"
        snapshot.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
        )
        request = Path(self.request_directory.name) / "request.json"
        request.write_text(
            json.dumps(
                {
                    "config": str(snapshot),
                    "paths": paths,
                }
            )
        )
        self.stopped = self.failed = False
        self.mode = mode
        self.pending_text = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.logs.clear()
        self.metrics_status.clear()
        self.metrics_status.hide()
        self.progress.hide()
        self.details.show()
        self.result_config = None
        self.previous_streams = set(
            self.output.glob("logs/executions/*/rank-0.jsonl.zst")
        )
        self.event_reader = None
        self.process = QtCore.QProcess(self)
        self.set_busy(True)
        self.show_status("Checking inputs and requested device…")
        self.progress.setRange(0, 0)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        worker = Path(__file__).resolve().parents[2] / "base/models/tisam_pipeline.py"
        self.process.start(sys.executable, ["-u", str(worker), mode, str(request)])

    def set_busy(self, busy):
        for page in self.setup_pages:
            page.setEnabled(not busy)
        for index in range(self.tabs.count()):
            self.tabs.setTabEnabled(
                index, not busy or index == self.tabs.currentIndex()
            )
        self.check_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.results_button.setEnabled(
            not busy
            and self.result_config is not None
            and bool(list((self.output / "test/predictions").glob("*.tif")))
        )
        self.segment_button.setEnabled(
            not busy
            and self.result_config is not None
            and (self.output / "checkpoints/best.safetensors").is_file()
        )
        self.folder_button.setEnabled(
            not busy and self.output is not None and self.output.is_dir()
        )
        self.update_navigation()
        self.busy_changed.emit(busy)

    def is_busy(self):
        return self.process is not None or (
            self.viewer is not None and self.viewer.is_busy()
        )

    def read_output(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.pending_text += text
        while "\n" in self.pending_text:
            line, self.pending_text = self.pending_text.split("\n", 1)
            if line.startswith("CODAVISION_EVENT "):
                self.handle_event(json.loads(line[len("CODAVISION_EVENT ") :]))
            elif line.strip():
                self.logs.appendPlainText(line)

    def handle_event(self, event):
        stage = event["stage"]
        self.progress.setVisible(stage in ("train", "test"))
        self.show_status(event["message"])
        self.logs.appendPlainText(event["message"])
        if event.get("total"):
            self.progress.setRange(0, event["total"])
            self.progress.setValue(event.get("completed") or 0)
        else:
            self.progress.setRange(0, 0)
        if stage == "checked":
            pixmap = QtGui.QPixmap(event["preview"])
            self.preview.setPixmap(
                pixmap.scaled(
                    400, 240, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
            )
            self.preview.setToolTip("Annotation preview; class colors from YAML")
            self.preview.show()
        elif stage == "train":
            self.native_timer.start()
        elif stage in ("test", "completed", "failed"):
            self.native_timer.stop()
        if stage == "failed":
            self.failed = True
            self.details.setChecked(True)
        elif stage == "completed":
            self.result_config = event["config"]
            scores = event["metrics"]
            self.show_status(
                "Completed · Test: "
                + " · ".join(
                    f"{key.capitalize()} {scores[key]:.3f}"
                    for key in ("accuracy", "precision", "recall", "f1")
                )
            )

    def read_training_events(self):
        if self.event_reader is None:
            streams = (
                set(self.output.glob("logs/executions/*/rank-0.jsonl.zst"))
                - self.previous_streams
            )
            if not streams:
                return
            self.event_reader = ExecutionEventTailReader(
                max(streams, key=lambda p: p.stat().st_mtime_ns)
            )
        try:
            for item in self.event_reader.poll():
                record = item.to_dict()
                if record.get("event") == "progress" and record.get("total"):
                    done = record.get("completed", 0)
                    self.progress.setRange(0, int(record["total"]))
                    self.progress.setValue(int(done))
                    epoch = record.get("coordinates", {}).get("epoch", 0) + 1
                    self.show_status(
                        f"{self.config.training.device} · epoch {epoch}/{self.config.training.epochs} · {record.get('phase', 'Training')} · {done}/{record['total']}"
                    )
                if record.get("epoch_metrics"):
                    metrics = ", ".join(
                        f"{name}: {value:.4g}"
                        for name, value in record["epoch_metrics"].items()
                        if isinstance(value, (int, float))
                    )
                    text = f"Epoch {record.get('epoch', 0) + 1}: {metrics}"
                    self.logs.appendPlainText(text)
                    self.metrics_status.setText(text)
                    self.metrics_status.show()
        except (OSError, ValueError, RuntimeError) as error:
            self.logs.appendPlainText(f"Progress reader: {error}")

    def stop(self):
        self.stopped = True
        self.show_status("Stopping… completed data and checkpoints will be retained.")
        self.stop_button.setEnabled(False)
        process = self.process
        if process:
            process.terminate()
            QtCore.QTimer.singleShot(
                3000, lambda: process.kill() if self.process is process else None
            )

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.failed = True
            self.show_status(self.process.errorString())
            self.process_finished(-1, QtCore.QProcess.CrashExit)

    def process_finished(self, code, status):
        self.read_output()
        if self.pending_text.strip():
            self.logs.appendPlainText(self.pending_text)
        self.native_timer.stop()
        self.process.deleteLater()
        self.process = None
        self.progress.hide()
        self.progress.setRange(0, 1)
        self.progress.setValue(int(code == 0 and not self.stopped))
        if self.stopped:
            self.show_status(
                "Stopped. Restart with the same configuration to reuse prepared data and resume the last completed checkpoint."
            )
        elif code != 0 and not self.failed:
            self.show_status(
                "Worker failed. See detailed logs; completed outputs were retained."
            )
            self.details.setChecked(True)
        if (self.output / "workflow.yml").is_file() and (
            self.output / "checkpoints/best.safetensors"
        ).is_file():
            self.result_config = str(self.output / "workflow.yml")
        self.set_busy(False)
        if self.mode == "run" and code == 0 and not self.stopped and not self.failed:
            self.open_results()

    def load_viewer(self):
        if self.viewer is not None and self.viewer.is_busy():
            self.inference_hint.setText(
                "Wait for the current viewer operation to finish before changing models."
            )
            return
        try:
            model = TiSAMModel.load(self.model_yaml.edit.text())
            new_viewer = TiSAMClassificationWindow(model, self, embedded=True)
        except Exception as error:
            self.inference_hint.setText(str(error))
            return
        if self.viewer:
            self.inference_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
        if self.inference_spacer is not None:
            self.inference_layout.removeItem(self.inference_spacer)
            self.inference_spacer = None
        self.viewer = new_viewer
        self.viewer.busy_changed.connect(self.viewer_busy)
        self.inference_layout.addWidget(self.viewer, 1)
        self.inference_hint.setText(
            "Select source images or inspect completed masks. Class colors come from the selected YAML."
        )

    def viewer_busy(self, busy):
        self.model_yaml.setEnabled(not busy)
        for index in range(len(self.setup_pages)):
            self.tabs.setTabEnabled(index, not busy)
        self.busy_changed.emit(busy)

    def use_trained_model(self):
        self.model_yaml.edit.setText(str(self.result_config))
        self.load_viewer()
        self.tabs.setCurrentIndex(self.inference_index)

    def open_results(self):
        self.use_trained_model()
        if self.viewer:
            root = self.output / "test"
            pairs = [
                (root / "images" / mask.name, mask)
                for mask in sorted((root / "predictions").glob("*.tif"))
            ]
            report = root / "metrics.json"
            if report.is_file():
                metrics = json.loads(report.read_text())
                self.inference_hint.setText(
                    "Independent Test · "
                    + " · ".join(
                        f"{key.capitalize()} {metrics[key]:.3f}"
                        for key in ("accuracy", "precision", "recall", "f1")
                    )
                    + " · GT void excluded"
                )
            self.viewer.show_results(
                [
                    SegmentationResult.from_model(image, mask, self.viewer.model)
                    for image, mask in pairs
                ]
            )


class WorkflowSwitcher(QtCore.QObject):
    """Mount two workflows in one MainWindow; preserve the original CNN form intact."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        original = window.takeCentralWidget()
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Workflow"))
        self.selector = QtWidgets.QComboBox()
        self.selector.addItems(["DeepLab / UNet", "TiSAM"])
        header.addWidget(self.selector)
        header.addStretch()
        layout.addLayout(header)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(original)
        self.tisam = TiSAMPage()
        self.stack.addWidget(self.tisam)
        layout.addWidget(self.stack)
        window.setCentralWidget(root)
        self.selector.currentIndexChanged.connect(self.switch)
        self.tisam.busy_changed.connect(lambda busy: self.selector.setEnabled(not busy))

    def switch(self, index):
        self.stack.setCurrentIndex(index)
        if index == 1:
            self.window.resize(
                max(960, self.window.width()), max(760, self.window.height())
            )

    def can_close(self):
        if self.tisam.is_busy():
            self.window.statusBar().showMessage(
                "Stop the TiSAM operation or wait for it to finish before closing."
            )
            return False
        return True
