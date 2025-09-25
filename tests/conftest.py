"""
Pytest Configuration and Fixtures for CODAvision Tests

This module provides shared fixtures and configuration for all tests,
with special handling for Qt/GUI tests.
"""

import os
import sys
import tempfile
import shutil
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

# Configure Qt for testing
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')  # For headless testing

# Try to import Qt components
try:
    from PySide6 import QtWidgets, QtCore
    from PySide6.QtTest import QTest
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    QtWidgets = None
    QtCore = None
    QTest = None


# Mark tests that require Qt
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "qt: mark test as requiring Qt/PySide6"
    )
    config.addinivalue_line(
        "markers", "qt_no_app: mark test as Qt test that doesn't need QApplication"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip Qt tests if Qt is not available."""
    if not QT_AVAILABLE:
        skip_qt = pytest.mark.skip(reason="Qt/PySide6 not available")
        for item in items:
            if "qt" in item.keywords or "gui" in item.keywords:
                item.add_marker(skip_qt)


# Qt Application Management
@pytest.fixture(scope='session')
def qapp_session():
    """Session-scoped Qt application for all Qt tests."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")

    # Check if an application already exists
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Create application with test-specific arguments
        app = QtWidgets.QApplication([
            'pytest',
            '-platform', 'offscreen',  # Headless mode
            '-style', 'fusion'  # Consistent style across platforms
        ])

    # Configure application for testing
    app.setQuitOnLastWindowClosed(False)  # Prevent app from quitting during tests

    yield app

    # Cleanup is handled by pytest-qt


@pytest.fixture
def qapp(qapp_session):
    """Test-function scoped Qt application."""
    yield qapp_session
    # Process any remaining events
    if QT_AVAILABLE:
        qapp_session.processEvents()


# pytest-qt compatibility fixtures
@pytest.fixture
def qtbot(qapp):
    """Mock qtbot fixture for tests that expect pytest-qt."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")

    class MockQtBot:
        """Minimal qtbot implementation for compatibility."""

        def __init__(self, app):
            self.app = app
            self._widgets = []

        def addWidget(self, widget):
            """Register a widget for cleanup."""
            self._widgets.append(widget)
            return widget

        def wait(self, ms):
            """Wait for specified milliseconds."""
            QtCore.QThread.msleep(ms)
            self.app.processEvents()

        def waitUntil(self, callback, timeout=5000):
            """Wait until callback returns True."""
            timer = QtCore.QElapsedTimer()
            timer.start()
            while not callback():
                if timer.elapsed() > timeout:
                    raise TimeoutError(f"waitUntil timed out after {timeout}ms")
                self.wait(10)
                self.app.processEvents()

        def waitSignal(self, signal, timeout=5000):
            """Wait for a signal to be emitted."""
            received = []

            def on_signal(*args):
                received.append(args)

            signal.connect(on_signal)

            timer = QtCore.QElapsedTimer()
            timer.start()
            while not received:
                if timer.elapsed() > timeout:
                    raise TimeoutError(f"Signal not received within {timeout}ms")
                self.wait(10)
                self.app.processEvents()

            return received[0] if received[0] else None

        def mouseClick(self, widget, button=QtCore.Qt.LeftButton, pos=None):
            """Simulate mouse click."""
            if pos is None:
                pos = widget.rect().center()
            QTest.mouseClick(widget, button, pos=pos)

        def keyClick(self, widget, key):
            """Simulate key press."""
            QTest.keyClick(widget, key)

        def cleanup(self):
            """Clean up registered widgets."""
            for widget in self._widgets:
                if widget and not widget.isHidden():
                    widget.close()
                    widget.deleteLater()
            self._widgets.clear()
            self.app.processEvents()

    bot = MockQtBot(qapp)
    yield bot
    bot.cleanup()


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for a single test."""
    temp_path = tempfile.mkdtemp(prefix='coda_test_')
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_dirs(temp_dir):
    """Create training and testing directories for tests."""
    training_path = os.path.join(temp_dir, 'training')
    testing_path = os.path.join(temp_dir, 'testing')
    os.makedirs(training_path)
    os.makedirs(testing_path)

    yield training_path, testing_path


# Image fixtures
@pytest.fixture
def sample_image():
    """Generate a sample RGB image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Generate a sample grayscale image for testing."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


# Mock fixtures for GUI components
@pytest.fixture
def mock_dialog():
    """Create a mock dialog for testing."""
    dialog = MagicMock()
    dialog.isVisible.return_value = False
    dialog.show = Mock()
    dialog.close = Mock()
    dialog.deleteLater = Mock()
    dialog.exec_ = Mock(return_value=QtWidgets.QDialog.Accepted if QT_AVAILABLE else 1)
    dialog.finished = Mock()  # Mock the finished signal
    return dialog


@pytest.fixture
def mock_message_box(monkeypatch):
    """Mock QMessageBox for testing."""
    mock = Mock()
    mock.information = Mock(return_value=QtWidgets.QMessageBox.Ok if QT_AVAILABLE else 1)
    mock.warning = Mock(return_value=QtWidgets.QMessageBox.Ok if QT_AVAILABLE else 1)
    mock.critical = Mock(return_value=QtWidgets.QMessageBox.Ok if QT_AVAILABLE else 1)
    mock.question = Mock(return_value=QtWidgets.QMessageBox.Yes if QT_AVAILABLE else 1)

    if QT_AVAILABLE:
        monkeypatch.setattr(QtWidgets, 'QMessageBox', mock)

    return mock


# TensorFlow mocking
@pytest.fixture
def mock_tensorflow(monkeypatch):
    """Mock TensorFlow for tests that don't need real TF."""
    mock_tf = MagicMock()
    mock_keras = MagicMock()

    # Mock common TF operations
    mock_tf.constant = Mock(side_effect=lambda x: x)
    mock_tf.Variable = Mock(side_effect=lambda x: x)
    mock_tf.keras = mock_keras

    # Mock Keras components
    mock_model = MagicMock()
    mock_model.predict = Mock(return_value=np.random.rand(1, 10))
    mock_model.fit = Mock()
    mock_keras.models.load_model = Mock(return_value=mock_model)
    mock_keras.Model = Mock(return_value=mock_model)

    monkeypatch.setattr('tensorflow', mock_tf)
    sys.modules['tensorflow'] = mock_tf

    yield mock_tf

    # Cleanup
    if 'tensorflow' in sys.modules:
        del sys.modules['tensorflow']


# Test data fixtures
@pytest.fixture
def sample_threshold_config():
    """Create a sample ThresholdConfig for testing."""
    from base.tissue_area.models import ThresholdConfig

    return ThresholdConfig(
        training_path='/tmp/training',
        testing_path='/tmp/testing',
        num_images=5,
        redo=False
    )


# Utility functions for testing
def process_qt_events(app=None, max_time=100):
    """Process Qt events for a maximum time."""
    if not QT_AVAILABLE:
        return

    if app is None:
        app = QtWidgets.QApplication.instance()

    if app:
        timer = QtCore.QElapsedTimer()
        timer.start()
        while timer.elapsed() < max_time:
            app.processEvents()
            QtCore.QThread.msleep(10)


# Skip markers for conditional testing
requires_display = pytest.mark.skipif(
    os.environ.get('DISPLAY') is None and sys.platform != 'win32',
    reason="Requires display (not running in headless mode)"
)

requires_gpu = pytest.mark.skipif(
    not any(os.path.exists(f"/dev/nvidia{i}") for i in range(8)),
    reason="Requires GPU"
)