"""Checkpoint segmentation dialog launched by CODAvision's main window.

A QProcess runs base/models/tisam_inference.py with CODAvision's interpreter
and installed dependencies, keeping inference off the GUI event loop.
"""

from __future__ import annotations

import codecs
from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets


class TiSAMDialog(QtWidgets.QDialog):
    """Collect inference inputs and manage the native TiSAM worker lifecycle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TiSAM segmentation")
        self.resize(760, 620)
        layout = QtWidgets.QVBoxLayout(self)
        help_text = QtWidgets.QLabel(
            "Segment an RGB TIFF or PNG with an existing TiSAM checkpoint.\n"
            "Output labels retain the checkpoint class order; 0 is void."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.form_widget)
        self.checkpoint = self.path_row(
            form, "Checkpoint", "TiSAM weights (*.safetensors *.pt)"
        )
        self.config = self.path_row(
            form, "Configuration (if needed)", "TiSAM configuration (*.yaml *.yml)"
        )
        self.config.setPlaceholderText(
            "Optional when checkpoint includes architecture, classes, mean and std"
        )
        self.source = self.path_row(
            form, "RGB image", "RGB images (*.tif *.tiff *.png)"
        )
        self.output = self.path_row(
            form, "Output mask", "TIFF mask (*.tif *.tiff)", save=True
        )
        self.device = QtWidgets.QLineEdit("cuda:0")
        self.device.setPlaceholderText("cuda:0, cuda:1, or cpu")
        form.addRow("Device", self.device)
        self.patch_size = self.spin_row(form, "Patch size (pixels)", 1024)
        self.effective_size = self.spin_row(form, "Effective size (pixels)", 800)
        self.batch_size = self.spin_row(form, "Batch size", 1)
        layout.addWidget(self.form_widget)
        self.status = QtWidgets.QLabel(
            "Ready. Interrupted segmentation resumes when the same output is selected."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        layout.addWidget(self.log)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Segment")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.close_button = QtWidgets.QPushButton("Close")
        for button in (self.start_button, self.stop_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.process = QtCore.QProcess(self)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.worker_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.kill_timer = QtCore.QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.process.kill)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.close_button.clicked.connect(self.reject)
        self.stopping = False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def path_row(self, form, label, file_filter, save=False):
        """Add a file selector; QProcess receives paths as individual arguments."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        field = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse…")
        layout.addWidget(field)
        layout.addWidget(button)

        def browse():
            choose = (
                QtWidgets.QFileDialog.getSaveFileName
                if save
                else QtWidgets.QFileDialog.getOpenFileName
            )
            path, _ = choose(self, label, field.text(), file_filter)
            if path:
                if save and not Path(path).suffix:
                    path += ".tif"
                field.setText(path)

        button.clicked.connect(browse)
        form.addRow(label, row)
        return field

    def spin_row(self, form, label, value):
        field = QtWidgets.QSpinBox()
        field.setRange(1, 65536)
        field.setValue(value)
        form.addRow(label, field)
        return field

    def start(self):
        """Validate paths and launch the packaged worker without a shell."""
        if self.process.state() != QtCore.QProcess.NotRunning:
            return
        try:
            for label, field in (
                ("Checkpoint", self.checkpoint),
                ("RGB image", self.source),
            ):
                if (
                    not field.text().strip()
                    or not Path(field.text()).expanduser().is_file()
                ):
                    raise ValueError(f"{label}: select an existing file")
            if (
                self.config.text()
                and not Path(self.config.text()).expanduser().is_file()
            ):
                raise ValueError("Configuration: select an existing YAML file")
            output = Path(self.output.text()).expanduser().resolve()
            if not self.output.text().strip() or output.suffix.lower() not in (
                ".tif",
                ".tiff",
            ):
                raise ValueError("Choose a .tif or .tiff output")
            if output.exists() or output.with_suffix(output.suffix + ".json").exists():
                raise ValueError(
                    "Output or class legend already exists; choose another output"
                )
            difference = self.patch_size.value() - self.effective_size.value()
            if difference < 0 or difference % 2:
                raise ValueError(
                    "Patch size minus effective size must be nonnegative and even"
                )
            if not self.device.text().strip():
                raise ValueError("Specify a device, such as cuda:0 or cpu")
        except ValueError as error:
            self.status.setText(str(error))
            return
        worker = (
            Path(__file__).resolve().parents[2]
            / "base"
            / "models"
            / "tisam_inference.py"
        )
        args = ["-u", str(worker)]
        for option, field in (
            ("checkpoint", self.checkpoint),
            ("source", self.source),
            ("output", self.output),
            ("config", self.config),
        ):
            if field.text():
                args.extend(
                    [f"--{option}", str(Path(field.text()).expanduser().resolve())]
                )
        args.extend(
            [
                "--device",
                self.device.text().strip(),
                "--patch-size",
                str(self.patch_size.value()),
                "--effective-size",
                str(self.effective_size.value()),
                "--batch-size",
                str(self.batch_size.value()),
            ]
        )
        self.log.clear()
        self.decoder.reset()
        self.stopping = False
        self.set_running(True)
        self.status.setText(
            "Running. Model loading and segmentation progress appear below."
        )
        self.process.start(sys.executable, args)

    def set_running(self, running):
        self.form_widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.close_button.setEnabled(not running)

    def read_output(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.set_running(False)
            self.status.setText(f"Could not start Python: {self.process.errorString()}")

    def worker_finished(self, exit_code, exit_status):
        """Release controls only after the worker exits and GPU allocations are released."""
        self.kill_timer.stop()
        self.read_output()
        self.set_running(False)
        if self.stopping:
            self.status.setText(
                "Stopped. Select the same inputs and output to resume incomplete segmentation."
            )
        elif exit_status == QtCore.QProcess.NormalExit and exit_code == 0:
            self.status.setText(
                "Complete. Saved the TIFF mask and its .tif.json/.tiff.json class legend."
            )
        else:
            self.status.setText(
                "Segmentation failed. See the log; check TiSAM installation, model access and checkpoint settings."
            )

    def stop(self):
        """Stop only this dialog's worker, retaining TiSAM's resumable scratch."""
        self.stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping…")
        self.process.terminate()
        self.kill_timer.start(3000)

    def reject(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.status.setText("Stop segmentation before closing this window.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.process.state() != QtCore.QProcess.NotRunning:
            event.ignore()
            self.status.setText("Stop segmentation before closing this window.")
        else:
            super().closeEvent(event)
