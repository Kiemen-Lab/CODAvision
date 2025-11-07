"""
Dialog Classes for Tissue Area Threshold Selection

This module contains the GUI dialog classes used for interactive tissue area
threshold selection process.
"""

import os
import sys
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QRect, QPointF
from typing import Optional, List, Tuple

from base.tissue_area.models import ThresholdMode, RegionSelection
from base.tissue_area.utils import create_tissue_mask

import logging
logger = logging.getLogger(__name__)


class ImageDisplayDialog(QtWidgets.QMainWindow):
    """Dialog for displaying full image and selecting a region."""
    
    def __init__(self, image_shape: Tuple[int, ...], resize_factor: float, parent=None):
        super().__init__(parent)
        
        # Import here to avoid circular imports
        from ..components.dialogs import Ui_choose_area
        
        self.ui = Ui_choose_area()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Click on a location at the edge of tissue and whitespace")
        self.clicked_position = None
        self.resize_factor = resize_factor
        
        # Set cursor and geometry
        self.ui.whole_im.setCursor(QtCore.Qt.CrossCursor)
        
        # Center the window
        width = int(np.round(image_shape[1] * resize_factor))
        height = int(np.round(image_shape[0] * resize_factor))
        x_offset = int(np.round((1500 - width) / 2))
        y_offset = int(np.round((800 - height) / 2)) 
        self.setGeometry(30 + x_offset, 30 + y_offset, width, height)
    
    def update_image(self, image: np.ndarray):
        """Update the displayed image."""
        image_array = np.ascontiguousarray(image)
        height, width = image_array.shape[:2]
        bytes_per_line = 3 * width
        
        qimage = QtGui.QImage(
            image_array.data, width, height, bytes_per_line, 
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        # Set geometry and scale
        scaled_width = int(np.round(width * self.resize_factor))
        scaled_height = int(np.round(height * self.resize_factor))
        self.ui.whole_im.setGeometry(0, 0, scaled_width, scaled_height)
        self.ui.whole_im.setPixmap(pixmap.scaled(
            self.ui.whole_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.ui.whole_im.setScaledContents(True)
    
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if self.ui.whole_im.geometry().contains(event.position().toPoint()):
            self.clicked_position = event.position() - QPointF(self.ui.whole_im.geometry().topLeft())
            self.close()
    
    def get_clicked_region(self, region_size: int = 600) -> Optional[RegionSelection]:
        """Get the clicked region selection."""
        if self.clicked_position is None:
            return None
        
        return RegionSelection(
            x=int(self.clicked_position.x()),
            y=int(self.clicked_position.y()),
            size=region_size
        )


class RegionCheckDialog(QtWidgets.QMainWindow):
    """Dialog for checking if the selected region is appropriate."""
    
    def __init__(self, region_size: int = 600, parent=None):
        super().__init__(parent)
        
        # Import here to avoid circular imports
        from ..components.dialogs import Ui_disp_crop
        
        self.ui = Ui_disp_crop()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Check selected region")
        self.region_size = region_size
        self.do_again = True
        
        # Setup layout
        self._setup_layout()
        
        # Connect signals
        self.ui.looks_good.clicked.connect(self.on_good)
        self.ui.new_loc.clicked.connect(self.on_new)
    
    def _setup_layout(self):
        """Setup the dialog layout."""
        central_widget = self.ui.centralwidget
        main_layout = QtWidgets.QVBoxLayout()
        self.setCentralWidget(central_widget)
        
        # Text label
        self.ui.text.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Fixed
        )
        main_layout.addWidget(self.ui.text, alignment=Qt.AlignCenter)
        
        # Image display
        self.ui.cropped.setFixedSize(self.region_size, self.region_size)
        self.ui.cropped.setAlignment(Qt.AlignCenter)
        self.ui.cropped.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        main_layout.addWidget(self.ui.cropped, alignment=Qt.AlignCenter)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(5)
        
        button_height = 90
        for button in [self.ui.looks_good, self.ui.new_loc]:
            button.setFixedHeight(button_height)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed
            )
        
        button_layout.addWidget(self.ui.looks_good)
        button_layout.addWidget(self.ui.new_loc)
        main_layout.addLayout(button_layout)
        
        central_widget.setLayout(main_layout)
    
    def on_good(self):
        """Handle 'looks good' button click."""
        self.do_again = False
        self.close()
    
    def on_new(self):
        """Handle 'new location' button click."""
        self.close()
    
    def update_image(self, cropped: np.ndarray):
        """Update the displayed cropped region."""
        image_array = np.ascontiguousarray(cropped)
        height, width = image_array.shape[:2]
        bytes_per_line = 3 * width
        
        qimage = QtGui.QImage(
            image_array.data, width, height, bytes_per_line,
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.ui.cropped.setGeometry(QRect(50, 40, self.region_size, self.region_size))
        self.ui.cropped.setPixmap(pixmap.scaled(
            self.ui.cropped.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        # Center the window
        frame_geom = self.frameGeometry()
        screen = QtWidgets.QApplication.primaryScreen()
        center_point = screen.availableGeometry().center()
        frame_geom.moveCenter(center_point)
        self.move(frame_geom.topLeft())


class ThresholdSelectionDialog(QtWidgets.QMainWindow):
    """Dialog for selecting tissue area threshold value."""
    
    def __init__(self, initial_threshold: int, mode: ThresholdMode, 
                 region_size: int = 600, parent=None):
        super().__init__(parent)
        
        # Import here to avoid circular imports
        from ..components.dialogs import Ui_choose_TA
        
        self.ui = Ui_choose_TA()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Select an appropriate intensity threshold for the binary mask")
        
        self.threshold = initial_threshold
        self.mode = mode
        self.region_size = region_size
        self.stop = True
        self.cropped_image = None
        
        self._setup_layout()
        self._connect_signals()
        self._update_mode_display()
        
        # Set initial slider value
        self.ui.TA_selection.setValue(initial_threshold)
    
    def _setup_layout(self):
        """Setup the dialog layout."""
        central_widget = self.ui.centralwidget
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Text
        self.ui.text.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        main_layout.addWidget(self.ui.text)
        
        # Images section
        images_layout = self._create_images_layout()
        main_layout.addLayout(images_layout)
        
        # Slider section
        slider_section = self._create_slider_section()
        main_layout.addWidget(slider_section)
        
        # Save button
        save_container = self._create_save_section()
        main_layout.addWidget(save_container)
        
        main_layout.addStretch()
    
    def _create_images_layout(self) -> QtWidgets.QHBoxLayout:
        """Create the images display layout."""
        # H&E image section
        he_layout = QtWidgets.QVBoxLayout()
        self.ui.medium_im.setMinimumSize(150, 20)
        self.ui.text_mid.setMinimumSize(150, 20)
        he_layout.addWidget(self.ui.text_mid)
        self.ui.medium_im.setMinimumSize(300, 300)
        self.ui.medium_im.setMaximumSize(300, 300)
        he_layout.addWidget(self.ui.medium_im)
        
        # Threshold image section
        ta_layout = QtWidgets.QVBoxLayout()
        self.ui.text_TA.setMinimumSize(150, 20)
        ta_layout.addWidget(self.ui.text_TA)
        self.ui.TA_im.setMinimumSize(300, 300)
        self.ui.TA_im.setMaximumSize(300, 300)
        ta_layout.addWidget(self.ui.TA_im)
        
        # Mode widget
        mode_widget = self._create_mode_widget()
        
        # Combine layouts
        images_layout = QtWidgets.QHBoxLayout()
        images_layout.setSpacing(4)
        images_layout.addLayout(he_layout)
        images_layout.addLayout(ta_layout)
        images_layout.addWidget(mode_widget)
        
        # Center the images
        images_outer_layout = QtWidgets.QHBoxLayout()
        images_outer_layout.addStretch()
        images_outer_layout.addLayout(images_layout)
        images_outer_layout.addStretch()
        
        return images_outer_layout
    
    def _create_mode_widget(self) -> QtWidgets.QWidget:
        """Create the mode selection widget."""
        mode_widget = QtWidgets.QWidget()
        mode_layout = QtWidgets.QVBoxLayout(mode_widget)
        
        self.ui.change_mode.setMinimumSize(155, 50)
        self.ui.change_mode.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        mode_layout.addWidget(self.ui.change_mode)
        
        self.ui.text_mode.setMinimumSize(155, 30)
        self.ui.text_mode.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        mode_layout.addWidget(self.ui.text_mode)
        
        self.ui.slider_label.setMinimumSize(155, 30)
        self.ui.slider_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        mode_layout.addWidget(self.ui.slider_label)
        
        mode_layout.addStretch()
        mode_layout.setContentsMargins(0, 25, 0, 0)
        
        return mode_widget
    
    def _create_slider_section(self) -> QtWidgets.QWidget:
        """Create the slider section."""
        slider_section = QtWidgets.QWidget()
        slider_layout = QtWidgets.QHBoxLayout(slider_section)
        
        self.ui.decrease_ta.setMinimumSize(120, 30)
        self.ui.decrease_ta.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        slider_layout.addWidget(self.ui.decrease_ta)
        
        slider_layout.addWidget(self.ui.slider_container, stretch=1)
        
        self.ui.raise_ta.setMinimumSize(120, 30)
        self.ui.raise_ta.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        slider_layout.addWidget(self.ui.raise_ta)
        
        return slider_section
    
    def _create_save_section(self) -> QtWidgets.QWidget:
        """Create the save button section."""
        save_container = QtWidgets.QWidget()
        save_layout = QtWidgets.QHBoxLayout(save_container)
        save_layout.addStretch()
        
        self.ui.apply.setMinimumSize(150, 50)
        self.ui.apply.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        save_layout.addWidget(self.ui.apply)
        
        return save_container
    
    def _connect_signals(self):
        """Connect dialog signals."""
        self.ui.change_mode.clicked.connect(self.on_mode)
        self.ui.apply.clicked.connect(self.on_apply)
        self.ui.TA_selection.valueChanged.connect(self.update_slider)
        self.ui.raise_ta.clicked.connect(self.on_raise)
        self.ui.decrease_ta.clicked.connect(self.on_decrease)
    
    def _update_mode_display(self):
        """Update the mode display based on current mode."""
        if self.mode == ThresholdMode.HE:
            self.ui.text_mode.setStyleSheet("background-color: white; color: black;")
            self.ui.text_mode.setText('Current mode: H&E')
            self.ui.decrease_ta.setText('More whitespace')
            self.ui.raise_ta.setText('More tissue')
        else:
            self.ui.text_mode.setStyleSheet("background-color: #333333; color: white;")
            self.ui.text_mode.setText(f'Current mode: {self.mode.value}')
            self.ui.raise_ta.setText('More whitespace')
            self.ui.decrease_ta.setText('More tissue')
    
    def on_raise(self):
        """Handle raise threshold button."""
        self.threshold += 1
        self.ui.TA_selection.setValue(self.threshold)
    
    def on_decrease(self):
        """Handle decrease threshold button."""
        self.threshold -= 1
        self.ui.TA_selection.setValue(self.threshold)
    
    def update_slider(self):
        """Update slider label and threshold image."""
        self.threshold = self.ui.TA_selection.value()
        self.ui.slider_label.setText(f"Current threshold value: {self.threshold}")
        self._update_threshold_image()
    
    def on_apply(self):
        """Handle apply button."""
        self.stop = False
        self.close()
    
    def on_mode(self):
        """Handle mode change."""
        if self.mode == ThresholdMode.HE:
            self.mode = ThresholdMode.GRAYSCALE
            self.threshold = 50
        else:
            self.mode = ThresholdMode.HE
            self.threshold = 205
        
        self.ui.TA_selection.setValue(self.threshold)
        self._update_mode_display()
        self._update_threshold_image()
    
    def _update_threshold_image(self):
        """Update the threshold preview image."""
        if self.cropped_image is None:
            return
        
        # Create tissue mask
        mask = create_tissue_mask(self.cropped_image, self.threshold, self.mode)
        
        # Invert mask display for H&E mode so tissue appears black
        if self.mode == ThresholdMode.HE:
            display_mask = 255 - mask
        else:
            display_mask = mask
        
        # Display mask
        height, width = display_mask.shape[:2]
        qimage = QtGui.QImage(
            display_mask.data, width, height, display_mask.strides[0],
            QtGui.QImage.Format_Grayscale8
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.ui.TA_im.setPixmap(pixmap.scaled(
            self.ui.TA_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self.ui.apply.setText('Save')
    
    def update_images(self, cropped: np.ndarray):
        """Update both the original and threshold images."""
        self.cropped_image = cropped
        
        # Update original image
        image_array = np.ascontiguousarray(cropped)
        height, width = image_array.shape[:2]
        bytes_per_line = 3 * width
        
        qimage = QtGui.QImage(
            image_array.data, width, height, bytes_per_line,
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.ui.medium_im.setPixmap(pixmap.scaled(
            self.ui.medium_im.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        # Update threshold image
        self._update_threshold_image()


class ImageSelectionDialog(QtWidgets.QMainWindow):
    """Dialog for selecting images to re-evaluate."""
    
    def __init__(self, base_path: str, parent=None):
        super().__init__(parent)
        
        # Import here to avoid circular imports
        from ..components.dialogs import Ui_choose_images_reevaluated
        
        self.ui = Ui_choose_images_reevaluated()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Confirm tissue mask evaluation")
        self.base_path = base_path
        self.images = []
        self.apply_all = False
        
        self._setup_layout()
        self._connect_signals()
    
    def _setup_layout(self):
        """Setup the dialog layout."""
        central_widget = self.ui.centralwidget
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Browse section
        browse_layout = QtWidgets.QHBoxLayout()
        self.ui.image_LE.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.ui.image_LE.setMinimumSize(448, 30)
        browse_layout.addWidget(self.ui.image_LE)
        
        self.ui.browse_PB.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        self.ui.browse_PB.setMinimumSize(100, 30)
        browse_layout.addWidget(self.ui.browse_PB)
        
        main_layout.addLayout(browse_layout)
        
        # List widget
        self.ui.image_LW.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.ui.image_LW.setMinimumSize(550, 100)
        main_layout.addWidget(self.ui.image_LW)
        
        # Buttons
        buttons_layout = self._create_buttons_layout()
        main_layout.addLayout(buttons_layout)
        
        main_layout.addStretch()
    
    def _create_buttons_layout(self) -> QtWidgets.QHBoxLayout:
        """Create the buttons layout."""
        buttons_layout = QtWidgets.QVBoxLayout()
        
        self.ui.delete_PB.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.ui.delete_PB.setMinimumSize(200, 30)
        buttons_layout.addWidget(self.ui.delete_PB)
        
        bottom_buttons = QtWidgets.QHBoxLayout()
        for button in [self.ui.apply_all_PB, self.ui.apply_PB]:
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed
            )
            button.setMinimumSize(100, 30)
            bottom_buttons.addWidget(button)
        
        buttons_layout.addLayout(bottom_buttons)
        
        buttons_container = QtWidgets.QHBoxLayout()
        buttons_container.addStretch()
        buttons_container.addLayout(buttons_layout)
        
        return buttons_container
    
    def _connect_signals(self):
        """Connect dialog signals."""
        self.ui.delete_PB.clicked.connect(self.on_delete_image)
        self.ui.browse_PB.clicked.connect(self.on_browse)
        self.ui.apply_PB.clicked.connect(self.on_apply)
        self.ui.apply_all_PB.clicked.connect(self.on_apply_all)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if event.spontaneous():
            self.images = []
            QtWidgets.QMessageBox.information(
                self, 'Info',
                'Keeping previous tissue mask evaluation'
            )
    
    def on_delete_image(self):
        """Handle delete image button."""
        delete_row = self.ui.image_LW.currentRow()
        if delete_row == -1:
            QtWidgets.QMessageBox.warning(
                self, 'Warning',
                'Please select path to delete'
            )
            return
        
        del self.images[delete_row]
        self.ui.image_LW.takeItem(delete_row)
    
    def on_browse(self):
        """Handle browse button."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image to Reevaluate",
            self.base_path, "TIFF Images (*.tif *.tiff)"
        )
        
        if file_path:
            if os.path.isfile(file_path):
                self.ui.image_LE.setText(file_path)
                
                if not (file_path.endswith('.tif') or file_path.endswith('.tiff')):
                    QtWidgets.QMessageBox.warning(
                        self, 'Warning',
                        'Select a .tif image.'
                    )
                elif any(np.isin(self.images, file_path)):
                    QtWidgets.QMessageBox.warning(
                        self, 'Warning',
                        'The selected image is already on the list'
                    )
                else:
                    self.images.append(file_path)
                    self.ui.image_LW.addItem(file_path)
                
                self.ui.image_LE.setText('')
        else:
            QtWidgets.QMessageBox.warning(
                self, 'Warning',
                'Selected image does not exist'
            )
    
    def on_apply(self):
        """Handle apply button."""
        # Convert to basenames
        self.images = [os.path.basename(img) for img in self.images]
        self.close()
    
    def on_apply_all(self):
        """Handle apply all button."""
        self.apply_all = True
        self.close()


class TissueMaskExistsDialog(QtWidgets.QMainWindow):
    """Dialog for asking user about existing tissue mask evaluation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Import here to avoid circular imports
        from ..components.dialogs import Ui_use_current_TA
        
        self.ui = Ui_use_current_TA()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Tissue Mask Already Evaluated")
        self.keep_current = True  # Default to keeping current
        
        self._setup_layout()
        self._connect_signals()
    
    def _setup_layout(self):
        """Setup the dialog layout."""
        central_widget = self.ui.centralwidget
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Add text label
        self.ui.text.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        main_layout.addWidget(self.ui.text, alignment=Qt.AlignCenter)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)
        
        button_height = 50
        for button in [self.ui.keep_ta, self.ui.new_ta]:
            button.setFixedHeight(button_height)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed
            )
        
        button_layout.addWidget(self.ui.keep_ta)
        button_layout.addWidget(self.ui.new_ta)
        main_layout.addLayout(button_layout)
        
        main_layout.addStretch()
        
        # Center the window
        self.setFixedSize(600, 200)
        frame_geom = self.frameGeometry()
        screen = QtWidgets.QApplication.primaryScreen()
        center_point = screen.availableGeometry().center()
        frame_geom.moveCenter(center_point)
        self.move(frame_geom.topLeft())
    
    def _connect_signals(self):
        """Connect dialog signals."""
        self.ui.keep_ta.clicked.connect(self.on_keep_current)
        self.ui.new_ta.clicked.connect(self.on_new_evaluation)
    
    def on_keep_current(self):
        """Handle keep current button click."""
        self.keep_current = True
        self.close()
    
    def on_new_evaluation(self):
        """Handle new evaluation button click."""
        self.keep_current = False
        self.close()