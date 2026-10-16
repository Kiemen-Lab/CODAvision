"""Native TiSAM result inputs and preview loading for the existing viewer.

TiSAMClassificationWindow supplies explicit source/mask/config paths.
This module reads visualization metadata and image pixels without importing
TiSAM encoders or requiring CODAvision's historical net.pkl layout.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image
from PySide6 import QtCore, QtGui

from base.models.tisam_metadata import (
    blend_overlay,
    save_overlay,
    visualization_settings,
)


@dataclass
class SegmentationResult:
    """Describe files consumed by TiSAMClassificationWindow."""

    source: Path
    mask: Path
    class_names: list[str]
    colormap: np.ndarray | None

    @classmethod
    def from_model(cls, source, mask, model):
        """Create a viewer result from the saved model's embedded config."""
        names, palette = visualization_settings(model.config)
        source, mask = Path(source), Path(mask)
        prepared = mask.parent / "images" / (source.stem + ".tif")
        if prepared.is_file():
            source = prepared
        return cls(source, mask, names, palette)

    def read_preview(self):
        """Read aligned samples with maximum side 2048; preserve uint16 class IDs."""
        with tifffile.TiffFile(self.mask) as handle:
            shape = handle.series[0].shape
        if len(shape) != 2:
            raise ValueError("Result mask must be a two-dimensional class-ID TIFF")
        step = max(1, (max(shape) + 2047) // 2048)
        labels = read_tiff_sample(self.mask, shape, step)
        if labels.dtype.kind not in "ui":
            raise ValueError("Result mask must contain integer class IDs")
        if int(labels.min()) < 0:
            raise ValueError("Result mask class IDs must be nonnegative")
        if self.class_names and int(labels.max()) >= len(self.class_names):
            raise ValueError("Mask class IDs exceed the configured class list")
        if self.source.suffix.lower() in (".tif", ".tiff"):
            rgb = read_tiff_sample(self.source, (*shape, 3), step)
        else:
            with Image.open(self.source) as handle:
                if handle.mode != "RGB" or handle.size != (shape[1], shape[0]):
                    raise ValueError("RGB source and mask dimensions must match")
                rgb = np.asarray(handle)[::step, ::step].copy()
        if rgb.dtype != np.uint8:
            raise ValueError("Viewer requires uint8 RGB source pixels")
        return np.ascontiguousarray(rgb), np.ascontiguousarray(labels), step


def read_tiff_sample(path, expected_shape, step):
    """Slice the base TIFF level lazily and close its backing Zarr store."""
    store = tifffile.imread(path, aszarr=True, level=0)
    try:
        array = zarr.open(store, mode="r")
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError("RGB source and mask dimensions must match")
        return np.asarray(array[::step, ::step])
    finally:
        store.close()


def render_preview(rgb, labels, mode, palette):
    """Render configured class colors without shifting IDs; void stays unchanged in overlay."""
    if mode == "Original":
        pixels = rgb
    elif palette is None:
        # Display numeric labels as grayscale when no explicit colors are available.
        maximum = max(1, int(labels.max()))
        pixels = np.repeat(
            (labels.astype(np.float32) * 255 / maximum).astype(np.uint8)[..., None],
            3,
            axis=2,
        )
    elif mode == "Mask":
        pixels = palette[labels]
    else:
        pixels = blend_overlay(rgb, labels, palette)
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    return QtGui.QImage(
        pixels.data, width, height, pixels.strides[0], QtGui.QImage.Format_RGB888
    ).copy()


class ResultPreviewLoader(QtCore.QThread):
    """Load bounded preview arrays off the GUI thread for TiSAMClassificationWindow."""

    loaded = QtCore.Signal(object, object, int)
    failed = QtCore.Signal(str)

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result

    def run(self):
        try:
            self.loaded.emit(*self.result.read_preview())
        except Exception as error:  # noqa: BLE001 - report worker failures to the GUI
            self.failed.emit(str(error))


class ResultExportWorker(QtCore.QThread):
    """Export the original mask or full-resolution config-colored JPG off the UI thread."""

    saved = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, result, destination, kind, parent=None):
        super().__init__(parent)
        self.result = result
        self.destination = Path(destination)
        self.kind = kind

    def run(self):
        try:
            if self.kind == "overlay":
                save_overlay(
                    self.result.source,
                    self.result.mask,
                    self.destination,
                    self.result.colormap,
                )
            else:
                # Copy bytes to preserve the original IDs, compression and full resolution.
                with (
                    self.result.mask.open("rb") as source,
                    self.destination.open("xb") as target,
                ):
                    try:
                        shutil.copyfileobj(source, target)
                    except Exception:
                        target.close()
                        self.destination.unlink()
                        raise
            self.saved.emit(str(self.destination))
        except Exception as error:  # noqa: BLE001 - report worker failures to the GUI
            self.failed.emit(str(error))
