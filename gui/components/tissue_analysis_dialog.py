"""
Tissue Analysis Dialog Components for CODAvision

This module provides GUI components for interactive tissue threshold selection,
allowing users to visually determine the optimal threshold for tissue segmentation.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QRect, QPointF, Signal
from PySide6.QtWidgets import QMainWindow, QLabel, QPushButton, QSlider, QMessageBox

from base.image.tissue_analysis import TissueAnalyzer


class ImageSelectionDialog(QMainWindow):
    """Dialog for selecting a location on a whole slide image."""
    
    def __init__(self, shape: Tuple[int, int], scale_factor: float):
        super().__init__()
        self.setWindowTitle("Click on a location at the edge of tissue and whitespace")
        
        # Set up the UI
        self.whole_im = QLabel(self)
        self.whole_im.setCursor(QtCore.Qt.CrossCursor)
        
        # Position window
        width, height = shape[1], shape[0]
        self.setGeometry(
            30 + int(np.round((1500 - width * scale_factor) / 2)), 
            30 + int(np.round((800 - height * scale_factor) / 2)),
            int(np.round(width * scale_factor)), 
            int(np.round(height * scale_factor))
        )
        
        self.clicked_position = None
        
    def update_image(self, image: np.ndarray, scale_factor: float) -> None:
        """Update the displayed image."""
        image_array = np.ascontiguousarray(image)
        height, width = image_array.shape[:2]
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image_array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        
        self.whole_im.setGeometry(0, 0, 
                                int(np.round(image.shape[1] * scale_factor)), 
                                int(np.round(image.shape[0] * scale_factor)))
        self.whole_im.setPixmap(pixmap.scaled(
            self.whole_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.whole_im.setScaledContents(True)
        
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse clicks."""
        if self.whole_im.geometry().contains(event.position().toPoint()):
            self.clicked_position = event.position() - QPointF(self.whole_im.geometry().topLeft())
            self.close()


class CropCheckDialog(QMainWindow):
    """Dialog for checking the selected image crop region."""
    
    def __init__(self, size: int):
        super().__init__()
        self.setWindowTitle("Check selected region")
        
        # Setup window geometry
        self.setGeometry(30 + int(np.round((1500 - size + 100) / 2)), 
                         50, int(np.round(size + 100)), 
                         int(np.round(size + 300)))
        
        # UI components
        self.cropped = QLabel(self)
        self.cropped.setGeometry(QRect(50, 40, size, size))
        
        self.looks_good = QPushButton("Looks good", self)
        self.looks_good.setGeometry(QRect(50, 50 + size, size // 2, 90))
        
        self.new_loc = QPushButton("No, select a new location", self)
        self.new_loc.setGeometry(QRect(50 + size // 2, 50 + size, size // 2, 90))
        
        self.text = QLabel("Is this a good location to evaluate tissue and whitespace detection?", self)
        self.text.setGeometry(QRect(50, 10, 900, 20))
        
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text.setFont(font)
        
        # State tracking
        self.do_again = True
        
        # Connect signals
        self.looks_good.clicked.connect(self.on_good)
        self.new_loc.clicked.connect(self.on_new)
        
    def on_good(self) -> None:
        """Accept the current crop."""
        self.do_again = False
        self.close()
        
    def on_new(self) -> None:
        """Reject the current crop."""
        self.close()
        
    def update_image(self, size: int, cropped: np.ndarray) -> None:
        """Update the displayed crop image."""
        image_array = np.ascontiguousarray(cropped)
        height, width = image_array.shape[:2]
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image_array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        
        self.cropped.setGeometry(QRect(50, 40, size, size))
        self.cropped.setPixmap(pixmap.scaled(
            self.cropped.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.cropped.setScaledContents(True)


class ThresholdSelectionDialog(QMainWindow):
    """Dialog for selecting intensity threshold for tissue segmentation."""
    
    def __init__(self, size: int, initial_threshold: int, mode: str):
        super().__init__()
        self.setWindowTitle("Select an appropriate intensity threshold for the binary mask")
        
        # State variables
        self.TA = initial_threshold  # Final threshold value
        self.CT0 = initial_threshold  # Current threshold value
        self.mode = mode
        self.stop = True
        
        # Window geometry
        self.setGeometry(
            30 + int(np.round((1500 - 1.5 * size + 100) / 2)),
            50,
            int(np.round(1.5 * size + 100)),
            int(np.round(230 + size / 2))
        )
        
        # UI Components
        self.TA_im = QLabel(self)
        self.TA_im.setFrameShape(QtWidgets.QFrame.Box)
        self.TA_im.setGeometry(QRect(54 + size * 0.75, 70, size // 2, size // 2))
        
        self.medium_im = QLabel(self)
        self.medium_im.setFrameShape(QtWidgets.QFrame.Box)
        self.medium_im.setGeometry(QRect(50 + size // 4, 70, size // 2, size // 2))
        
        self.apply = QPushButton("Save", self)
        self.apply.setGeometry(QRect(212 + size, 160 + size // 2, 150, 50))
        
        self.slider_container = QtWidgets.QWidget(self)
        self.slider_container.setGeometry(QRect(176, 80 + size // 2, size + 56, 60))
        
        self.TA_selection = QSlider(Qt.Horizontal, self.slider_container)
        self.TA_selection.setMinimum(0)
        self.TA_selection.setMaximum(255)
        self.TA_selection.setValue(initial_threshold)
        self.TA_selection.setGeometry(0, 30, size + 56, 30)
        
        self.slider_label = QLabel(str(initial_threshold), self.slider_container)
        self.slider_label.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.slider_label.setStyleSheet("background-color: white; color: black;")
        self.slider_label.setAlignment(Qt.AlignCenter)
        
        # Position the slider label
        handle_pos = self.slider_handle_position()
        self.slider_label.setGeometry(handle_pos - 15, 0, 30, 20)
        
        self.text = QLabel("Select an intensity threshold so that the tissue in the binary image is marked in black", self)
        self.text.setGeometry(QRect(50, 10, int(np.round(8 + size * 1.5)), 20))
        
        self.text_mid = QLabel("Original image", self)
        self.text_mid.setGeometry(QRect(50 + size // 4, 40, 3 + size // 2, 20))
        
        self.text_TA = QLabel("Binary mask", self)
        self.text_TA.setGeometry(QRect(54 + size * 0.75, 40, 3 + size // 2, 20))
        
        # Set fonts for labels
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text.setFont(font)
        self.text_mid.setFont(font)
        self.text_TA.setFont(font)
        
        self.text_mode = QLabel(f"Current mode: {mode}", self)
        self.text_mode.setGeometry(QRect(58 + size * 1.25, 125, 155, 30))
        self.text_mode.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        if mode == 'H&E':
            self.text_mode.setStyleSheet("background-color: white; color: black;")
        else:
            self.text_mode.setStyleSheet("background-color: black; color: white;")
            
        self.change_mode = QPushButton("Change mode", self)
        self.change_mode.setGeometry(QRect(58 + size * 1.25, 70, 155, 50))
        
        self.raise_ta = QPushButton("More tissue" if mode == 'H&E' else "More whitespace", self)
        self.raise_ta.setGeometry(QRect(242 + size, 105 + size // 2, 120, 30))
        
        self.decrease_ta = QPushButton("More whitespace" if mode == 'H&E' else "More tissue", self)
        self.decrease_ta.setGeometry(QRect(50, 105 + size // 2, 120, 30))
        
        # Connect signals
        self.apply.clicked.connect(self.on_apply)
        self.TA_selection.valueChanged.connect(self.update_slider)
        self.change_mode.clicked.connect(self.on_mode)
        self.raise_ta.clicked.connect(self.on_raise)
        self.decrease_ta.clicked.connect(self.on_decrease)
    
    def slider_handle_position(self) -> int:
        """Get the current position of the slider handle."""
        option = QtWidgets.QStyleOptionSlider()
        self.TA_selection.initStyleOption(option)
        handle_rect = self.TA_selection.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option,
            QtWidgets.QStyle.SC_SliderHandle, self.TA_selection
        )
        
        slider_x = self.TA_selection.pos().x()
        return slider_x + handle_rect.x() + (handle_rect.width() // 2)
    
    def on_apply(self) -> None:
        """Handle apply button click."""
        self.TA = self.CT0
        self.stop = False
        self.close()
        
    def update_slider(self) -> None:
        """Update slider and threshold display."""
        self.slider_label.setText(str(self.TA_selection.value()))
        handle_pos = self.slider_handle_position()
        self.slider_label.setGeometry(handle_pos - 15, 0, 30, 20)
        self.on_change_TA()
        
    def on_raise(self) -> None:
        """Increase threshold value."""
        self.CT0 = self.CT0 + 10
        self.TA_selection.setValue(self.CT0)
        
    def on_decrease(self) -> None:
        """Decrease threshold value."""
        self.CT0 = self.CT0 - 10
        self.TA_selection.setValue(self.CT0)
        
    def on_change_TA(self) -> None:
        """Handle threshold value changes."""
        self.CT0 = self.TA_selection.value()
        self.change_TA('TA')
        
    def on_mode(self) -> None:
        """Toggle between H&E and Grayscale modes."""
        if self.mode == 'H&E':
            self.mode = 'Grayscale'
            self.TA_selection.setValue(50)
            self.CT0 = 50
        else:
            self.mode = 'H&E'
            self.TA_selection.setValue(205)
            self.CT0 = 205
        self.change_TA('mode')
        
    def change_TA(self, change: str) -> None:
        """Update display based on current threshold and mode."""
        if change == 'mode':
            if self.mode == 'H&E':
                self.text_mode.setStyleSheet("background-color: white; color: black;")
                self.text_mode.setText(f'Current mode: H&E')
                self.decrease_ta.setText('More whitespace')
                self.raise_ta.setText('More tissue')
            else:
                self.raise_ta.setText('More whitespace')
                self.decrease_ta.setText('More tissue')
                self.text_mode.setText(f'Current mode: {self.mode}')
                self.text_mode.setStyleSheet("background-color: black; color: white;")
    
    def update_image(self, size: int, cropped: np.ndarray) -> None:
        """Update both the original and threshold images."""
        # Update original image display
        image_array_medium = np.ascontiguousarray(cropped)
        height, width = image_array_medium.shape[:2]
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image_array_medium.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        
        self.medium_im.setPixmap(pixmap.scaled(
            self.medium_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.medium_im.setScaledContents(True)
        
        # Update threshold image
        if self.mode == 'H&E':
            image_array = np.ascontiguousarray(((cropped[:, :, 1] > self.CT0) * 255).astype(np.uint8))
        else:
            image_array = np.ascontiguousarray(((cropped[:, :, 1] < self.CT0) * 255).astype(np.uint8))
            
        height, width = image_array.shape[:2]
        q_image = QtGui.QImage(image_array.data, width, height, image_array.strides[0], 
                               QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        
        self.TA_im.setPixmap(pixmap.scaled(
            self.TA_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.TA_im.setScaledContents(True)


class ConfirmThresholdDialog(QMainWindow):
    """Dialog for confirming existing threshold usage."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confirm tissue mask evaluation")
        self.setGeometry(550, 300, 550, 130)
        
        # UI components
        self.text = QLabel("The tissue mask has already been evaluated.\n"
                          "Do you want to choose a new tissue mask evaluation?", self)
        self.text.setGeometry(QRect(25, 10, 500, 60))
        
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text.setFont(font)
        self.text.setAlignment(Qt.AlignCenter)
        
        self.keep_ta = QPushButton("Keep current tissue mask evaluation", self)
        self.keep_ta.setGeometry(QRect(25, 80, 248, 40))
        
        self.new_ta = QPushButton("Evaluate tissue mask again", self)
        self.new_ta.setGeometry(QRect(275, 80, 248, 40))
        
        # State tracking
        self.keep_TA = False
        
        # Connect signals
        self.keep_ta.clicked.connect(self.on_keep_TA)
        self.new_ta.clicked.connect(self.on_new_TA)
        
    def on_keep_TA(self):
        """Keep the current tissue mask evaluation."""
        self.keep_TA = True
        self.close()
        
    def on_new_TA(self):
        """Select a new tissue mask evaluation."""
        self.keep_TA = False
        self.close()


class ImageSelectionForEvaluationDialog(QMainWindow):
    """Dialog for selecting images to re-evaluate."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confirm tissue mask evaluation")
        self.setGeometry(450, 300, 600, 250)
        
        # UI components
        self.image_LE = QtWidgets.QLineEdit(self)
        self.image_LE.setGeometry(QRect(25, 10, 448, 30))
        
        self.image_LW = QtWidgets.QListWidget(self)
        self.image_LW.setGeometry(QRect(25, 45, 550, 100))
        self.image_LW.setEnabled(True)
        self.image_LW.setStyleSheet("QListWidget { border: 1px solid white; }")
        
        self.browse_PB = QPushButton("Browse", self)
        self.browse_PB.setGeometry(QRect(475, 10, 100, 30))
        
        self.delete_PB = QPushButton("Delete", self)
        self.delete_PB.setGeometry(QRect(375, 150, 200, 30))
        
        self.apply_PB = QPushButton("Accept", self)
        self.apply_PB.setGeometry(QRect(477, 185, 100, 30))
        
        self.apply_all_PB = QPushButton("Redo all images", self)
        self.apply_all_PB.setGeometry(QRect(375, 185, 100, 30))
        
        # State tracking
        self.images = []
        self.apply_all = False
        
        # Connect signals
        self.delete_PB.clicked.connect(self.on_delete_image)
        self.browse_PB.clicked.connect(self.on_browse)
        self.apply_PB.clicked.connect(self.on_apply)
        self.apply_all_PB.clicked.connect(self.on_apply_all)
        
    def closeEvent(self, event):
        """Handle window close event."""
        if event.spontaneous():
            self.images = []
            QtWidgets.QMessageBox.information(self, 'Info',
                                          'Keeping previous tissue mask evaluation')
            
    def on_delete_image(self):
        """Remove selected image from the list."""
        delete_row = self.image_LW.currentRow()
        if delete_row == -1:
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                      'Please select path to delete')
            return
            
        del self.images[delete_row]
        self.image_LW.takeItem(delete_row)
        
    def on_browse(self):
        """Browse for an image to add to the list."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image to Reevaluate", "", "TIFF Images (*.tif *.tiff)"
        )
        
        if file_path:
            if os.path.isfile(file_path):
                self.image_LE.setText(file_path)
                
                if not (file_path.endswith(('.tif', '.tiff'))):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'Select a .tif image.')
                elif file_path in self.images:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'The selected image is already on the list')
                else:
                    self.images.append(file_path)
                    self.image_LW.addItem(file_path)

                self.image_LE.setText('')
            else:
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                         'Selected image does not exist')

    def on_apply(self):
        """Apply selection of specific images."""
        # Convert full paths to base filenames
        for i, image in enumerate(self.images):
            self.images[i] = os.path.basename(image)
        self.close()

    def on_apply_all(self):
        """Apply selection to all images."""
        self.apply_all = True
        self.close()


class TissueAnalysisController:
    """
    Controller that manages the tissue analysis workflow.

    This class coordinates the user interaction flow through multiple
    dialogs to determine optimal tissue detection thresholds.
    """

    def __init__(self, image_path: str, num_images: int = 0):
        """
        Initialize the tissue analysis workflow.

        Args:
            image_path: Path to directory containing images
            num_images: Number of images to analyze (0 for all)
        """
        self.image_path = image_path
        self.num_images = num_images
        self.analyzer = TissueAnalyzer(image_path)
        self.crop_size = 600  # Size of crop region

    def run(self) -> None:
        """Run the complete tissue analysis workflow."""
        print('   ')

        # Get list of images
        imlist = self.analyzer.load_images()
        if not imlist:
            print(f"No image files found in {self.image_path}")
            return

        # Normalize paths
        for i, image in enumerate(imlist):
            imlist[i] = image[len(self.image_path) + 1:] if image.startswith(self.image_path) else image

        # Check for existing thresholds
        outpath = os.path.join(self.image_path, 'TA')
        os.makedirs(outpath, exist_ok=True)

        cts = {}
        mode = 'H&E'

        # Check if threshold data already exists
        if os.path.isfile(os.path.join(outpath, 'TA_cutoff.pkl')):
            if self.num_images > 0:
                # If analyzing specific number of images, ask user whether to keep existing thresholds
                keep_TA = self._confirm_threshold_usage()
                if keep_TA:
                    print('   Optimal cutoff already chosen, skip this step')
                    return

                # Load existing thresholds
                with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    mode = data['mode']
            else:
                # If analyzing all images, check which ones need thresholds
                with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    cts = data['cts']
                    mode = data['mode']

                # Find images that don't have thresholds yet
                done = list(cts.keys())
                imlist_temp = list(set(imlist) - set(done))

                if not imlist_temp:
                    # All images have thresholds, ask if user wants to redo any
                    keep_TA = self._confirm_threshold_usage()
                    if keep_TA:
                        print('   Optimal cutoff already chosen for all images, skip this step')
                        return
                    else:
                        # Ask which images to redo
                        apply_all, redo_list = self._select_images_to_reevaluate()
                        if not apply_all:
                            imlist = redo_list

        # Set default threshold based on mode
        CT0 = 50 if mode == 'Grayscale' else 205

        # Random selection if analyzing specific number of images
        if self.num_images > 0:
            self.num_images = min(self.num_images, len(imlist))
            imlist = np.random.choice(imlist, size=self.num_images, replace=False)
            print(f'Evaluating {self.num_images} randomly selected images to choose a good whitespace detection...')
            average_TA = True
        else:
            self.num_images = len(imlist)
            print(f'Evaluating all training images to choose a good whitespace detection...')
            average_TA = False

        # Process each image
        count = 0
        for nm in imlist:
            count += 1
            print(f'    Loading image {count} of {self.num_images}: {nm}')

            # Load image
            try:
                im0 = cv2.imread(os.path.join(self.image_path, nm))
                im0 = im0[:, :, ::-1]  # Convert BGR to RGB
                print('     Image loaded')
            except Exception as e:
                print(f"    Error loading image {nm}: {e}")
                continue

            # Scale factor for display
            rsf = min(1500 / im0.shape[1], 780 / im0.shape[0])

            # Get user to select a point on the image
            do_again = True
            while do_again:
                # Show image and get user click
                click = self._display_image(im0, rsf)
                if click is None:
                    break

                x = click.x()
                y = click.y()
                x_norm = int(np.round(x / rsf))
                y_norm = int(np.round(y / rsf))

                # Crop around the selected point
                cropped = self._crop_image(im0, x_norm, y_norm)

                # Check if crop is good
                do_again = self._check_crop_region(self.crop_size, cropped)

            # Let user select threshold
            sstop, CT0, mode = self._select_threshold(self.crop_size, cropped, CT0, mode)
            if sstop:
                print('Whitespace detection process stopped by the user')
                return

            # Save threshold for this image
            if nm in cts:
                cts[nm] = CT0
            else:
                cts[nm] = CT0

        # Save thresholds to file
        with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'wb') as f:
            pickle.dump({'cts': cts, 'imlist': imlist, 'mode': mode, 'average_TA': average_TA}, f)

    def _get_qt_application(self) -> QtWidgets.QApplication:
        """Get or create a Qt application instance."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app

    def _display_image(self, im0: np.ndarray, rsf: float) -> Optional[QPointF]:
        """Display image and get user click location."""
        app = self._get_qt_application()
        window = ImageSelectionDialog(im0.shape, rsf)
        window.show()
        window.update_image(im0, rsf)
        app.exec()
        return window.clicked_position

    def _check_crop_region(self, size: int, cropped: np.ndarray) -> bool:
        """Check if the cropped region is suitable."""
        app = self._get_qt_application()
        window = CropCheckDialog(size)
        window.show()
        window.update_image(size, cropped)
        app.exec()
        return window.do_again

    def _select_threshold(self, size: int, cropped: np.ndarray,
                         initial_threshold: int, mode: str) -> Tuple[bool, int, str]:
        """Select threshold for tissue detection."""
        app = self._get_qt_application()
        window = ThresholdSelectionDialog(size, initial_threshold, mode)
        window.show()
        window.update_image(size, cropped)
        app.exec()
        return window.stop, window.TA, window.mode

    def _confirm_threshold_usage(self) -> bool:
        """Ask user whether to use existing thresholds."""
        app = self._get_qt_application()
        window = ConfirmThresholdDialog()
        window.show()
        app.exec()
        return window.keep_TA

    def _select_images_to_reevaluate(self) -> Tuple[bool, List[str]]:
        """Let user select which images to reevaluate."""
        app = self._get_qt_application()
        window = ImageSelectionForEvaluationDialog()
        window.show()
        app.exec()
        return window.apply_all, window.images

    def _crop_image(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """Crop a region from the image around the given point."""
        size = self.crop_size
        cropped_temp = image.copy()

        # Handle images smaller than crop size
        if 2 * size > image.shape[0] and 2 * size > image.shape[1]:
            # Both dimensions too small, pad to square
            max_dim = max(image.shape[0], image.shape[1])
            pad_x = (max_dim - image.shape[0]) // 2
            pad_y = (max_dim - image.shape[1]) // 2

            # Ensure padding is even
            pad_x1, pad_x2 = pad_x, pad_x + (max_dim - image.shape[0]) % 2
            pad_y1, pad_y2 = pad_y, pad_y + (max_dim - image.shape[1]) % 2

            return np.pad(image, ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)), mode='constant')

        elif 2 * size > image.shape[0]:
            # Height too small, pad vertically
            pad_x = (2 * size - image.shape[0]) // 2
            pad_x1, pad_x2 = pad_x, pad_x + (2 * size - image.shape[0]) % 2

            padded = np.pad(image, ((pad_x1, pad_x2), (0, 0), (0, 0)), mode='constant')
            cropped_temp = padded[y_norm - size:y_norm + size, x_norm - size:x_norm + size, :]

        elif 2 * size > image.shape[1]:
            # Width too small, pad horizontally
            pad_y = (2 * size - image.shape[1]) // 2
            pad_y1, pad_y2 = pad_y, pad_y + (2 * size - image.shape[1]) % 2

            padded = np.pad(image, ((0, 0), (pad_y1, pad_y2), (0, 0)), mode='constant')
            cropped_temp = padded[y_norm - size:y_norm + size, x_norm - size:x_norm + size, :]

        # Handle points near edges
        if y < size:
            cropped_temp = cropped_temp[0:2 * size, :, :]
        elif y + size > image.shape[0]:
            cropped_temp = cropped_temp[-2 * size:, :, :]
        else:
            cropped_temp = cropped_temp[y - size:y + size, :, :]

        if x < size:
            cropped_temp = cropped_temp[:, 0:2 * size, :]
        elif x + size > image.shape[1]:
            cropped_temp = cropped_temp[:, -2 * size:, :]
        else:
            cropped_temp = cropped_temp[:, x - size:x + size, :]

        return cropped_temp