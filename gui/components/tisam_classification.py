"""TiSAM inference in CODAvision's existing classification UI.

Reuse MainWindowClassify's canvas, zoom, color editing and directory removal.
Only model-specific IDs, result I/O and the native inference worker differ;
the original CNN class and its training/quantification pipeline stay unchanged.
"""

from __future__ import annotations

import codecs
import re
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPixmap
from PySide6.QtWidgets import QHeaderView

from base.models.tisam_metadata import visualization_settings
from base.models.tisam_model import TiSAMModel, image_pairs

from .classification_window import MainWindowClassify
from .segmentation_result import (
    ResultExportWorker,
    ResultPreviewLoader,
    SegmentationResult,
    render_preview,
)
from .ui_definitions import Ui_classify_im


class TiSAMClassificationWindow(MainWindowClassify):
    """Bind a YAML-selected TiSAM model to the shared directory classification UI."""

    busy_changed = QtCore.Signal(bool)

    def __init__(self, model, parent=None, *, embedded=False):
        # The CNN constructor reads net.pkl and training tables; use its UI directly.
        QtWidgets.QMainWindow.__init__(
            self, parent, Qt.Widget if embedded else Qt.Window
        )
        self.embedded = embedded
        self.ui = Ui_classify_im()
        self.ui.setupUi(self)
        self.setCentralWidget(self.ui.centralwidget)
        self.model = model
        self.train_im_fold = str(model.path.parent)
        self.classify = []
        self.results = []
        self.result = None
        self.preview_rgb = self.preview_labels = None
        self.preview_loader = self.export_worker = self.process = None
        self.progress = None
        self.queue = []
        self.process_log = ""
        self.stopped = False
        self.ui.browseCl_PB.clicked.connect(self.select_imagedir)
        self.ui.deleteCl_PB.clicked.connect(self.remove_imagedir)
        self.ui.classify_PB.clicked.connect(self.apply_changes)
        self.ui.classify_LE.setToolTip(
            "NDPI / SVS / RGB image directory; press Enter to add"
        )
        self.ui.classify_LE.returnPressed.connect(
            lambda: self.add_directory(self.ui.classify_LE.text())
        )
        self.ui.change_color_PB.clicked.connect(self.change_color)
        self.ui.reset_PB.clicked.connect(self.reset_cmap)
        self.ui.zoom_in_PB.clicked.connect(self.zoom_in)
        self.ui.zoom_out_PB.clicked.connect(self.zoom_out)
        # CNN component quantification assumes different label semantics.
        self.ui.groupBox_5.hide()
        self.ui.overlay.setGeometry(245, 200, 585, 325)
        self.ui.overlay.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.toolbar = self.addToolBar("Results")
        self.toolbar.setMovable(False)
        self.save_mask_action = self.toolbar.addAction(
            "Save mask as…", lambda: self.export_result("mask")
        )
        self.save_overlay_action = self.toolbar.addAction(
            "Export overlay JPG…", lambda: self.export_result("overlay")
        )
        self.save_overlay_action.setToolTip(
            "Export full resolution using the selected YAML colors"
        )
        self.open_folder_action = self.toolbar.addAction(
            "Open results folder", self.open_results_folder
        )
        self.view_mode = QtWidgets.QComboBox(self)
        self.toolbar.addWidget(self.view_mode)
        self.view_mode.currentTextChanged.connect(self.show_image)
        self.image_selector = QtWidgets.QComboBox(self)
        self.image_selector.setMinimumWidth(180)
        self.image_selector.setToolTip("Choose a classified image")
        self.toolbar.addWidget(self.image_selector)
        self.image_selector.currentIndexChanged.connect(self.select_result)
        self.configure_model()
        self.build_responsive_layout()
        self.set_busy(False)
        self.statusBar().showMessage(
            "Add image directories, then click Apply. Masks are saved in classification_"
            + model.name
            + "_TiSAM."
        )

    def build_responsive_layout(self):
        """Reuse the canvas and source controls inside a resizeable embedded layout."""
        ui = self.ui
        layout = QtWidgets.QVBoxLayout(ui.centralwidget)
        source = QtWidgets.QHBoxLayout()
        for widget in (ui.label, ui.classify_LE, ui.browseCl_PB):
            source.addWidget(widget)
        layout.addLayout(source)
        ui.classify_LW.setMaximumHeight(80)
        layout.addWidget(ui.classify_LW)
        actions = QtWidgets.QHBoxLayout()
        for widget in (ui.deleteCl_PB, ui.classify_PB, ui.zoom_in_PB, ui.zoom_out_PB):
            actions.addWidget(widget)
        layout.addLayout(actions)
        content = QtWidgets.QHBoxLayout()
        ui.colormap_TW.setParent(ui.centralwidget)
        ui.colormap_TW.setMaximumWidth(220)
        content.addWidget(ui.colormap_TW)
        content.addWidget(ui.overlay, 1)
        layout.addLayout(content, 1)
        # YAML is the only palette source; no divergent preview-only color edits.
        ui.groupBox_4.hide()
        ui.change_color_PB.hide()
        ui.reset_PB.hide()
        self.inline_progress = QtWidgets.QWidget()
        progress_layout = QtWidgets.QHBoxLayout(self.inline_progress)
        self.inline_label = QtWidgets.QLabel()
        self.inline_label.setWordWrap(True)
        progress_layout.addWidget(self.inline_label, 1)
        stop = QtWidgets.QPushButton("Stop")
        stop.clicked.connect(self.stop_inference)
        progress_layout.addWidget(stop)
        layout.addWidget(self.inline_progress)
        self.inline_progress.hide()

    def is_busy(self):
        return any(
            item is not None
            for item in (self.process, self.preview_loader, self.export_worker)
        )

    def show_results(self, results):
        """Display explicit Test source/mask pairs in the same viewer."""
        self.results = results
        for widget in (
            self.ui.label,
            self.ui.classify_LE,
            self.ui.browseCl_PB,
            self.ui.classify_LW,
            self.ui.deleteCl_PB,
            self.ui.classify_PB,
        ):
            widget.hide()
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        for result in results:
            self.image_selector.addItem(result.source.name)
        self.image_selector.blockSignals(False)
        self.result = None
        self.preview_rgb = self.preview_labels = None
        self.ui.scene.clear()
        self.set_busy(False)
        if results:
            self.select_result(0)

    def configure_model(self):
        """Use canonical class IDs and the YAML palette for preview and exports."""
        self.setWindowTitle(f"CODAvision — {self.model.name} (TiSAM)")
        self.classNames, self.cmap = visualization_settings(self.model.config)
        self.data = {"cmap": None if self.cmap is None else self.cmap.copy()}
        self.view_mode.blockSignals(True)
        self.view_mode.clear()
        self.view_mode.addItems(["Original", "Mask"])
        if self.cmap is not None:
            self.view_mode.addItem("Overlay")
            self.view_mode.setCurrentText("Overlay")
        self.view_mode.blockSignals(False)
        self.ui.change_color_PB.setEnabled(self.cmap is not None)
        self.ui.reset_PB.setEnabled(self.cmap is not None)
        self.populate_cmap()

    def populate_cmap(self):
        """Display TiSAM's zero-based IDs, including void and the final class."""
        table = self.ui.colormap_TW
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Annotation Class"])
        table.setRowCount(len(self.classNames))
        table.setVerticalHeaderLabels([str(i) for i in range(len(self.classNames))])
        for row, name in enumerate(self.classNames):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if self.cmap is not None:
                color = self.cmap[row]
                item.setBackground(QColor(*color))
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                item.setForeground(QBrush(Qt.black if brightness > 128 else Qt.white))
            table.setItem(row, 0, item)
        table.resizeRowsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def select_imagedir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Classify Image Directory", self.train_im_fold
        )
        if directory:
            self.add_directory(directory)

    def add_directory(self, directory):
        """Populate the existing directory list and display masks already on disk."""
        if not directory.strip():
            return
        try:
            directory = str(Path(directory).expanduser().resolve())
            image_pairs(directory, self.model.classification_dir(directory))
            if directory in self.classify:
                raise ValueError("The selected folder is already on the list")
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Source directory", str(error))
            return
        self.classify.append(directory)
        self.ui.classify_LW.addItem(directory)
        self.ui.classify_LE.clear()
        self.train_im_fold = directory
        self.refresh_results()

    def remove_imagedir(self):
        super().remove_imagedir()
        self.refresh_results()

    def refresh_results(self):
        """View completed masks without needing the checkpoint or starting inference."""
        self.results = []
        for directory in self.classify:
            try:
                self.results.extend(
                    SegmentationResult.from_model(source, mask, self.model)
                    for source, mask in image_pairs(
                        directory, self.model.classification_dir(directory)
                    )
                    if mask.is_file()
                )
            except (OSError, ValueError) as error:
                self.statusBar().showMessage(str(error))
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        for result in self.results:
            self.image_selector.addItem(
                f"{result.source.parent.name}/{result.source.name}"
            )
        self.image_selector.blockSignals(False)
        self.result = None
        self.preview_rgb = self.preview_labels = None
        self.ui.scene.clear()
        self.set_busy(False)
        if self.results:
            self.select_result(0)
        else:
            self.statusBar().showMessage(
                "No completed masks in the selected directories. Click Apply to classify."
            )

    def select_result(self, index):
        if index < 0:
            return
        self.result = self.results[index]
        self.image_selector.setToolTip(
            f"Image: {self.result.source}\nMask: {self.result.mask}"
        )
        self.preview_rgb = self.preview_labels = None
        self.ui.scene.clear()
        self.set_busy(True)
        self.statusBar().showMessage("Loading preview…")
        self.preview_loader = ResultPreviewLoader(self.result, self)
        self.preview_loader.loaded.connect(self.result_loaded)
        self.preview_loader.failed.connect(self.result_load_failed)
        self.preview_loader.finished.connect(self.preview_finished)
        self.preview_loader.start()

    def preview_finished(self):
        self.preview_loader.deleteLater()
        self.preview_loader = None
        self.set_busy(False)

    def export_finished(self):
        self.export_worker.deleteLater()
        self.export_worker = None
        self.set_busy(False)

    def set_busy(self, busy):
        self.busy_changed.emit(busy)
        for widget in (
            self.ui.browseCl_PB,
            self.ui.deleteCl_PB,
            self.ui.classify_PB,
            self.ui.classify_LE,
        ):
            widget.setEnabled(not busy)
        self.image_selector.setEnabled(not busy and bool(self.results))
        ready = not busy and self.preview_rgb is not None
        self.save_mask_action.setEnabled(ready)
        self.save_overlay_action.setEnabled(ready)
        self.open_folder_action.setEnabled(ready)
        self.ui.zoom_in_PB.setEnabled(ready)
        if not ready:
            self.ui.zoom_out_PB.setEnabled(False)

    def show_image(self):
        if self.preview_rgb is None:
            return
        image = render_preview(
            self.preview_rgb,
            self.preview_labels,
            self.view_mode.currentText(),
            self.cmap,
        )
        if self.ui.scene.items():
            self.ui.overlay.image_item.setPixmap(QPixmap.fromImage(image))
        else:
            self.disp_image(image)

    def apply_changes(self):
        """Run native inference for the shared directory list, then refresh this viewer."""
        if not self.classify:
            QtWidgets.QMessageBox.warning(
                self, "Source directory", "Add an RGB image directory first."
            )
            return
        try:
            model = TiSAMModel.load(self.model.path)
            queue = [
                directory
                for directory in self.classify
                if any(
                    not mask.exists()
                    for _, mask in image_pairs(
                        directory, model.classification_dir(directory)
                    )
                )
            ]
            if queue and not model.checkpoint.is_file():
                raise ValueError(f"Checkpoint not found: {model.checkpoint}")
            self.model, self.queue = model, queue
            self.configure_model()
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Cannot classify", str(error))
            return
        if not self.queue:
            self.refresh_results()
            return
        self.stopped = False
        self.set_busy(True)
        self.inline_progress.show()
        self.inline_label.setText("Loading TiSAM…")
        self.start_directory()

    def start_directory(self):
        """Use the shared interpreter in a cancellable process, without a new environment."""
        directory = self.queue.pop(0)
        self.process_log = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_progress)
        self.process.finished.connect(self.inference_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.inline_label.setText(f"Classifying {directory}\nLoading TiSAM…")
        worker = Path(__file__).resolve().parents[2] / "base/models/tisam_inference.py"
        self.process.start(
            sys.executable,
            [
                "-u",
                str(worker),
                "--profile",
                str(self.model.path),
                "--source-dir",
                directory,
                "--device",
                self.model.device,
            ],
        )

    def read_progress(self):
        self.process_log = (
            self.process_log
            + self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        )[-65536:]
        messages = re.findall(r"Image \d+/\d+: [^\r\n]+", self.process_log)
        if messages:
            self.inline_label.setText(messages[-1])

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.process_log += self.process.errorString()
            self.inference_finished(-1, QtCore.QProcess.CrashExit)

    def stop_inference(self):
        self.stopped = True
        if self.process is not None:
            process = self.process
            process.terminate()
            QtCore.QTimer.singleShot(
                3000, lambda: process.kill() if self.process is process else None
            )

    def inference_finished(self, code, status):
        """Keep partial outputs accessible on errors or cancellation; never overwrite masks."""
        self.read_progress()
        self.process.deleteLater()
        self.process = None
        if (
            code == 0
            and status == QtCore.QProcess.NormalExit
            and self.queue
            and not self.stopped
        ):
            self.start_directory()
            return
        self.queue = []
        self.inline_progress.hide()
        if not self.stopped and (code != 0 or status != QtCore.QProcess.NormalExit):
            error = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Critical,
                "TiSAM classification failed",
                "Inference failed. Completed masks were kept; see details for the worker error.",
                parent=self,
            )
            error.setDetailedText(self.process_log)
            error.exec()
        self.refresh_results()

    def closeEvent(self, event):
        if any(
            item is not None
            for item in (self.process, self.preview_loader, self.export_worker)
        ):
            self.statusBar().showMessage(
                "Stop inference or wait for preview/export to finish before closing."
            )
            event.ignore()
            return
        super().closeEvent(event)

    def result_loaded(self, rgb, labels, step):
        """Cache preview pixels so recoloring and view switching do not reread the slide."""
        self.preview_rgb, self.preview_labels = rgb, labels
        detail = (
            f"Preview sampled every {step} pixel(s); saved mask is full resolution."
        )
        if self.cmap is None:
            detail += " No config colormap: mask is grayscale; overlay unavailable."
        else:
            detail += " Colors come from the selected YAML."
        self.statusBar().showMessage(detail)
        self.show_image()

    def result_load_failed(self, message):
        """Expose file/config mismatch without leaving a modal loading dialog open."""
        self.statusBar().showMessage(f"Cannot display result: {message}")

    def export_result(self, kind):
        """Choose a destination only after viewing; reuse the shared full-resolution export."""
        if kind == "overlay" and self.result.colormap is None:
            self.statusBar().showMessage(
                "Cannot export overlay: the model YAML requires tisam_config.colormap."
            )
            return
        suffix = ".jpg" if kind == "overlay" else ".tif"
        suggested = (
            self.result.mask.with_suffix(".jpg")
            if kind == "overlay"
            else self.result.mask.with_name(self.result.mask.stem + "_copy.tif")
        )
        destination, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export overlay JPG" if kind == "overlay" else "Save mask as",
            str(suggested),
            "JPEG overlay (*.jpg)" if kind == "overlay" else "TIFF mask (*.tif *.tiff)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if not destination:
            return
        destination = Path(destination).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(suffix)
        if destination.suffix.lower() not in (
            (".jpg",) if kind == "overlay" else (".tif", ".tiff")
        ):
            self.statusBar().showMessage(f"Choose a {suffix} filename.")
            return
        if destination.exists():
            self.statusBar().showMessage(
                "File already exists. Choose a new filename; existing results are never overwritten."
            )
            return
        self.save_mask_action.setEnabled(False)
        self.save_overlay_action.setEnabled(False)
        self.set_busy(True)
        self.statusBar().showMessage("Exporting full-resolution result…")
        self.export_worker = ResultExportWorker(self.result, destination, kind, self)
        self.export_worker.saved.connect(
            lambda path: self.statusBar().showMessage(f"Saved: {path}")
        )
        self.export_worker.failed.connect(
            lambda message: self.statusBar().showMessage(f"Export failed: {message}")
        )
        self.export_worker.finished.connect(self.export_finished)
        self.export_worker.start()

    def open_results_folder(self):
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.result.mask.parent))
        )
