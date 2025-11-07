"""
Mock Dialog Implementations for GUI Testing

This module provides mock implementations of GUI dialogs to prevent
Qt event loop blocking during tests.
"""

from unittest.mock import Mock, MagicMock
from PySide6 import QtCore, QtWidgets


class MockDialogBase:
    """Base class for mock dialogs."""

    def __init__(self):
        self.finished = Mock()  # Mock signal
        self.accepted = Mock()  # Mock signal
        self.rejected = Mock()  # Mock signal
        self._visible = False
        self._result = QtWidgets.QDialog.Accepted

    def show(self):
        """Mock show method."""
        self._visible = True

    def close(self):
        """Mock close method."""
        self._visible = False
        self.finished.emit(self._result)

    def isVisible(self):
        """Mock isVisible method."""
        return self._visible

    def deleteLater(self):
        """Mock deleteLater method."""
        pass

    def setWindowModality(self, modality):
        """Mock setWindowModality method."""
        pass

    def exec_(self):
        """Mock exec_ method - non-blocking."""
        return self._result


class MockImageDisplayDialog(MockDialogBase):
    """Mock for ImageDisplayDialog."""

    def __init__(self, shape=(100, 100), resize_factor=1.0):
        super().__init__()
        self.clicked_position = None
        self.resize_factor = resize_factor
        self.ui = MagicMock()
        self.ui.whole_im = MagicMock()

    def update_image(self, image):
        """Mock update_image method."""
        self.ui.whole_im.pixmap = Mock(return_value=MagicMock())

    def get_clicked_region(self, size):
        """Mock get_clicked_region method."""
        from base.tissue_area.models import RegionSelection
        return RegionSelection(50, 50, size)

    def mousePressEvent(self, event):
        """Mock mouse press event."""
        self.clicked_position = event.position()


class MockRegionCheckDialog(MockDialogBase):
    """Mock for RegionCheckDialog."""

    def __init__(self, size=600):
        super().__init__()
        self.do_again = True
        self.ui = MagicMock()
        self.ui.cropped = MagicMock()

    def update_image(self, cropped):
        """Mock update_image method."""
        self.ui.cropped.pixmap = Mock(return_value=MagicMock())

    def on_good(self):
        """Mock on_good method."""
        self.do_again = False

    def on_new(self):
        """Mock on_new method."""
        self.do_again = True


class MockThresholdSelectionDialog(MockDialogBase):
    """Mock for ThresholdSelectionDialog."""

    def __init__(self, threshold=205, mode=None, size=600):
        super().__init__()
        from base.tissue_area.models import ThresholdMode
        self.threshold = threshold
        self.mode = mode or ThresholdMode.HE
        self.stop = True
        self.ui = MagicMock()
        self.ui.TA_selection = MagicMock()
        self.ui.TA_selection.value = Mock(return_value=threshold)
        self.ui.TA_selection.setValue = Mock()

    def update_images(self, cropped):
        """Mock update_images method."""
        pass

    def on_raise(self):
        """Mock on_raise method."""
        self.threshold += 1
        self.ui.TA_selection.setValue(self.threshold)

    def on_decrease(self):
        """Mock on_decrease method."""
        self.threshold -= 1
        self.ui.TA_selection.setValue(self.threshold)

    def on_mode(self):
        """Mock on_mode method."""
        from base.tissue_area.models import ThresholdMode
        if self.mode == ThresholdMode.HE:
            self.mode = ThresholdMode.GRAYSCALE
            self.threshold = 50
        else:
            self.mode = ThresholdMode.HE
            self.threshold = 205
        self.ui.TA_selection.setValue(self.threshold)

    def on_apply(self):
        """Mock on_apply method."""
        self.stop = False

    def update_slider(self):
        """Mock update_slider method."""
        self.threshold = self.ui.TA_selection.value()


class MockImageSelectionDialog(MockDialogBase):
    """Mock for ImageSelectionDialog."""

    def __init__(self, base_path='/test/path'):
        super().__init__()
        self.base_path = base_path
        self.images = []
        self.apply_all = False
        self.ui = MagicMock()
        self.ui.image_LW = MagicMock()

    def on_apply_all(self):
        """Mock on_apply_all method."""
        self.apply_all = True

    def on_delete_image(self):
        """Mock on_delete_image method."""
        row = self.ui.image_LW.currentRow()
        if 0 <= row < len(self.images):
            del self.images[row]
            self.ui.image_LW.takeItem(row)

    def windowTitle(self):
        """Mock windowTitle method."""
        return "Confirm tissue mask evaluation"


class MockTissueMaskExistsDialog(MockDialogBase):
    """Mock for TissueMaskExistsDialog."""

    def __init__(self, keep_current=False):
        super().__init__()
        self.keep_current = keep_current


def create_mock_show_dialog_modal(qtbot=None):
    """
    Create a mock for _show_dialog_modal that doesn't block.

    Args:
        qtbot: Optional qtbot fixture for more advanced testing

    Returns:
        Mock function for _show_dialog_modal
    """

    def mock_show_dialog_modal(dialog):
        """Mock implementation that doesn't block."""
        # Just mark dialog as shown without blocking
        if hasattr(dialog, 'show'):
            dialog.show()

        # If we have qtbot, process events briefly
        if qtbot:
            qtbot.wait(10)  # Small delay to simulate dialog interaction

        # Mark dialog as closed
        if hasattr(dialog, 'close'):
            dialog._visible = False

    return mock_show_dialog_modal


def patch_qt_dialogs(monkeypatch):
    """
    Patch all Qt dialog classes with mock implementations.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr(
        'gui.tissue_area.dialogs.ImageDisplayDialog',
        MockImageDisplayDialog
    )
    monkeypatch.setattr(
        'gui.tissue_area.dialogs.RegionCheckDialog',
        MockRegionCheckDialog
    )
    monkeypatch.setattr(
        'gui.tissue_area.dialogs.ThresholdSelectionDialog',
        MockThresholdSelectionDialog
    )
    monkeypatch.setattr(
        'gui.tissue_area.dialogs.ImageSelectionDialog',
        MockImageSelectionDialog
    )
    monkeypatch.setattr(
        'gui.tissue_area.dialogs.TissueMaskExistsDialog',
        MockTissueMaskExistsDialog
    )