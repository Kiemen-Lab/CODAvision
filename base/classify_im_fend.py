"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: October 22, 2024
"""

from PySide6 import QtWidgets, QtCore
from base.classify_im import Ui_MainWindow
from base.classify_images import classify_images
from base.quantify_objects import quantify_objects
from base.quantify_images import quantify_images
import os
import cv2

import pandas as pd
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush, QImage, QPixmap, QMouseEvent, QFont
from PySide6.QtWidgets import (QColorDialog, QHeaderView, QDialog, QPushButton, QLabel, QVBoxLayout,
                               QProgressBar, QProgressDialog, QGraphicsPixmapItem)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
import pickle

import time
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

class MainWindowClassify(QtWidgets.QMainWindow):
    def __init__(self, train_im_fold, nm, model_type, train_fold = ''):
        super(MainWindowClassify,self).__init__()  # Use super() to initialize the parent clasz
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # Pass the MainWindow instance itself as the parent
        self.setCentralWidget(self.ui.centralwidget)  # Set the central widget
        if len(train_fold)==0:
            for element in train_im_fold.split(os.sep)[:-1]:
                train_fold = os.path.join(train_fold, element)
        self.train_fold = train_fold
        with open(os.path.join(train_fold,nm,'net.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        self.cmap = self.data['cmap'].copy()
        self.classNames = self.data['classNames']
        self.classify = []
        self.addws = self.data['WS'][1][0]
        self.class_addws = self.data['final_df'].at[self.addws-1, 'Color']
        self.change_cmap = []
        self.train_fold = train_fold
        self.pathDL = os.path.join(train_fold, nm)
        self.train_im_fold = train_im_fold
        self.nm = nm
        self.model_type = model_type
        self.path_first_img = ''
        self.name_first_img =''
        print(os.path.join(train_im_fold, 'classification_'+self.nm+'_'+self.model_type))
        if os.path.isdir(os.path.join(train_im_fold, 'classification_'+self.nm+'_'+self.model_type)):
            for filename in os.listdir(os.path.join(train_im_fold, 'classification_'+self.nm+'_'+self.model_type)):
                # Check if the file has a .tiff or .tif extension (case insensitive)
                if filename.lower().endswith(('.tiff', '.tif')):
                    self.path_first_img = os.path.join(train_im_fold, 'classification_'+self.nm+'_'+self.model_type, filename)
                    self.name_first_img = filename
                    continue
        if self.path_first_img == '' or self.name_first_img == '':
            self.image_displayed = False
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The selected model has no classified images yet.')
        else:
            self.image_displayed = True
        self.ui.change_color_PB.clicked.connect(self.change_color)
        self.ui.reset_PB.clicked.connect(self.reset_cmap)
        self.ui.browseCl_PB.clicked.connect(self.select_imagedir)
        self.ui.deleteCl_PB.clicked.connect(self.remove_imagedir)
        self.ui.classify_PB.clicked.connect(self.apply_changes)
        self.ui.zoom_in_PB.clicked.connect(self.zoom_in)
        self.ui.zoom_out_PB.clicked.connect(self.zoom_out)
        self.overlay = []
        self.populate_cmap()
        self.populate_object_analysis()
        # self.ui.overlay.setText('Loading image. Please wait...')
        font = QFont('Times',14,QFont.Bold)
        #font.setPointSize(14)  # Set font size to 16
        self.ui.zoom_in_PB.setFont(font)
        self.ui.zoom_out_PB.setFont(font)
        QTimer.singleShot(80, self.show_image)



    def select_imagedir(self):
        dialog_title = f'Select Classify Image Directory'
        folder_path = os.path.normpath(QtWidgets.QFileDialog.getExistingDirectory(self, dialog_title, os.getcwd()))
        if folder_path:
            if os.path.isdir(folder_path):
                self.ui.classify_LE.setText(folder_path)
                if not os.path.isdir(folder_path):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected folder does not exist.')
                elif not (any([f.endswith(('.tif')) for f in os.listdir(folder_path)]) or any([f.endswith(('.png')) for f in os.listdir(folder_path)]) or any([f.endswith(('.jpg')) for f in os.listdir(folder_path)])):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected folder does not contain valid image files (.tif, .png or .jpg).')
                elif any(np.isin(self.classify, folder_path)):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected folder is already on the list')
                else:
                    self.classify.append(folder_path)
                    self.ui.classify_LW.addItem(folder_path)
                self.ui.classify_LE.setText('')
        else:
            self.ui.path_check.exec_()

    def remove_imagedir(self):
        delete_row = self.ui.classify_LW.currentRow()
        if delete_row == -1:
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Please select path to delete')
            return
        del self.classify[delete_row]
        self.ui.classify_LW.takeItem(delete_row)

    def populate_cmap(self):
        table = self.ui.colormap_TW
        # table.setRowCount(len(df))
        table.setRowCount(0) # Clear the table before populating
        table.setColumnCount(1)  # Adjust column count
        table.setHorizontalHeaderLabels(["Annotation Class"])
        for layer in self.data['classNames'][:-1]:
            row = table.rowCount()
            table.insertRow(row)
            color = self.cmap[row]

            item = QtWidgets.QTableWidgetItem(layer)
            item.setBackground(QColor(*color))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make the item read-only

            # Convert the background color to greyscale
            greyscale = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if greyscale > 128:  # If greyscale is above 50% grey, set text color to black
                item.setForeground(QBrush(QColor(0, 0, 0)))
            else:  # Otherwise, set text color to white
                item.setForeground(QBrush(QColor(255, 255, 255)))

            table.setItem(row, 0, item)


        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Stretch columns to

    def change_color(self):
        table = self.ui.colormap_TW
        selected_items = table.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select an annotation class from the table.')
            return

        selected_row = selected_items[0].row()
        current_color = self.cmap[selected_row]
        initial_color = QColor(*current_color)

        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(initial_color)

        if color_dialog.exec():
            new_color = color_dialog.currentColor()
            new_rgb = (new_color.red(), new_color.green(), new_color.blue())

            # Update the DataFrame
            self.cmap[selected_row] = new_rgb

            # Update only the Annotation Class column in the table
            item = table.item(selected_row, 0)
            if item:
                item.setBackground(new_color)

            layer_name = self.classNames[selected_row]
            print(f"Color changed for {layer_name} to {new_rgb}")

        self.populate_cmap()
        self.show_image()

    def reset_cmap(self):
        self.cmap = self.data['cmap'].copy()
        self.populate_cmap()
        self.show_image()

    def populate_object_analysis(self):
        self.ui.component_TW.setColumnCount(1)
        self.ui.component_TW.setHorizontalHeaderLabels(["Annotation layers"])
        self.ui.component_TW.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)  # Data matches the table width

        for row, layer_name in enumerate(self.classNames[:-1]):
            self.ui.component_TW.insertRow(row)
            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            # Set background color to dark and text color to white
            if item.checkState() == Qt.Checked:
                item.setBackground(QColor(200, 200, 200))  # Light gray
                item.setForeground(QBrush(Qt.black))  # Black text
            else:
                item.setBackground(QColor(45, 45, 45))  # Dark color
                item.setForeground(QBrush(Qt.white))  # White text

            self.ui.component_TW.setItem(row, 0, item)

            # Connect itemChanged signal to slot (value gets gray after being checked)
        self.ui.component_TW.itemChanged.connect(self.on_item_changed)

    def on_item_changed(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor(200, 200, 200))  # Light gray
            item.setForeground(QBrush(Qt.black))  # Black text
        else:
            item.setBackground(QColor(45, 45, 45))# Dark color
            item.setForeground(QBrush(Qt.white))

    def show_image(self):
        path_image = os.path.join(self.train_im_fold,self.name_first_img)
        self.loader = ImageLoader(self.path_first_img, path_image, self.cmap, self.classNames, self.image_displayed)
        self.process_thread = ProcessThread(self.loader)
        self.loader.started.connect(self.on_processing_started)
        self.loader.finished.connect(self.on_processing_finished)
        self.loader.loaded.connect(self.disp_image)
        self.process_thread.start()

    def on_processing_started(self):
        # Show dialog when processing starts
        self.ui.loading_dialog.exec()


    def on_processing_finished(self):
        # Close dialog when processing is complete
        self.ui.loading_dialog.accept()

    def disp_image(self, img):
        pixmap = QPixmap.fromImage(img)
        scale_x = self.ui.overlay.width() / pixmap.width()
        scale_y = self.ui.overlay.height() / pixmap.height()
        self.scale = min(scale_x, scale_y)  # Use the smaller scale factor to maintain aspect ratio
        self.original_scale = self.scale
        # Reset transformations and apply the scale
        self.ui.overlay.resetTransform()
        self.ui.overlay.scale(self.scale, self.scale)
        self.ui.overlay.image_item = QGraphicsPixmapItem(pixmap)
        self.ui.scene.addItem(self.ui.overlay.image_item)
        self.ui.overlay.image_item.setPos((self.ui.overlay.width() - pixmap.width() * self.scale) / 2,
                               (self.ui.overlay.height() - pixmap.height() * self.scale) / 2)
        self.ui.overlay.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.overlay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.zoom_out_PB.setEnabled(False)

    def zoom_in(self):
        self.scale = self.scale*1.1
        self.ui.overlay.resetTransform()
        self.ui.overlay.scale(self.scale, self.scale)
        self.ui.overlay.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.ui.overlay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.ui.zoom_out_PB.setEnabled(True)

    def zoom_out(self):
        self.scale = self.scale/1.1
        self.ui.overlay.resetTransform()
        self.ui.overlay.scale(self.scale, self.scale)
        if self.original_scale >= self.scale:
            self.ui.zoom_out_PB.setEnabled(False)
            self.ui.overlay.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.ui.overlay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def apply_changes(self):
        self.close()
        self.progress_window = ProgressWindow(self.pathDL, self.classNames, self.cmap, self.ui.classify_LW, self.nm,
                                              self.model_type,self.train_fold, self.ui.component_TW)
        self.progress_window.show()


class ProgressWindow(QDialog):
    """Displays a progress bar in a dialog while a function runs."""
    def __init__(self, pthDL, classNames, cmap, list, nm, model_type, train_fold, tissue_list):
        super().__init__()

        self.setWindowTitle("Processing...")
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(800)

        # Layout
        layout = QVBoxLayout()

        # Label for status
        self.label = QLabel("Processing in progress...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, list.count() * 4)  # Total steps
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        layout.addWidget(self.cancel_button)

        # Close button (hidden until processing is complete)
        self.close_button = QPushButton("Close")
        self.close_button.setVisible(False)
        self.close_button.clicked.connect(self.accept)  # Close the dialog
        layout.addWidget(self.close_button)

        self.setLayout(layout)

        # Start the worker thread
        self.worker = WorkerThread(pthDL, classNames, cmap, list, nm, model_type, train_fold, tissue_list)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.message_updated.connect(self.status_label.setText)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, reason):
        """Handles completion by updating UI and delaying close."""
        if reason == 'Completed':
            self.label.setText("Processing Complete!")
            self.progress_bar.setValue(self.progress_bar.maximum())  # Ensure it's full
            self.close_button.setVisible(True)  # Show close button
            self.cancel_button.setVisible(False)
            QTimer.singleShot(3000, self.accept)  # Delay close for 3 seconds
        else:
            self.close()

    def cancel_operation(self):
        """Handles cancel button click."""
        self.worker.stop()  # Stop the worker thread
        self.worker.wait()

    def closeEvent(self, event):
        """Handle X button click (window close)."""
        self.cancel_operation()  # Call the same function as the cancel button
        event.accept()

class ImageLoader(QObject):
    loaded = Signal(QImage)
    started = Signal()
    finished = Signal()

    def __init__(self, pth_im0, pth_im, cmap, classNames, image_displayed):
        super().__init__()
        self.pth_im0 = pth_im0
        self.pth_im = pth_im
        self.cmap = cmap
        self.classNames = classNames
        self.image_displayed = image_displayed

    def load_image(self):
        self.started.emit()
        if self.image_displayed:
            im0 = cv2.imread(self.pth_im0, cv2.IMREAD_GRAYSCALE)  # Mask
            im = cv2.imread(self.pth_im)  # Image
            im = im[:, :, ::-1]
            if im.shape[0] > 3500 or im.shape[1] > 3500:
                im0 = im0[::10, ::10]
                im = im[::10, ::10, :]
            r = np.zeros_like(im0).astype(np.uint8)
            g = np.zeros_like(im0).astype(np.uint8)
            b = np.zeros_like(im0).astype(np.uint8)
            for l in range(1, len(self.classNames)):
                idx = im0 == l
                r[idx] = self.cmap[l - 1, 0]
                g[idx] = self.cmap[l - 1, 1]
                b[idx] = self.cmap[l - 1, 2]
            prediction_cmap = np.stack([r, g, b], axis=2)
            overlay = cv2.addWeighted(im, 0.65, prediction_cmap, 0.35, 0)
            overlay = np.ascontiguousarray(overlay.astype(np.uint8))
            height, width = overlay.shape[:2]
            bytes_per_line = 3 * width
            qimage = QImage(overlay.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            overlay = np.zeros([1,1])
            overlay = np.ascontiguousarray(overlay.astype(np.uint8))
            height, width = overlay.shape
            bytes_per_line = width
            qimage = QImage(overlay.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.loaded.emit(qimage)
        self.finished.emit()


class ProcessThread(QThread):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def run(self):
        self.loader.load_image()


class WorkerThread(QThread):
    """Runs a long task using input from the GUI without freezing the UI."""
    progress_updated = Signal(int)
    message_updated = Signal(str)
    finished_signal = Signal(str)  # Signal to send results back

    def __init__(self, pthDL,classNames, cmap, list, name, model_type, train_fold, tissue_list):
        super().__init__()
        self.classNames = classNames
        self.pthDL = pthDL
        self.cmap = cmap
        self.list = list
        self.nm = name
        self.model_type = model_type
        self.train_fold = train_fold
        self.tissue_list = tissue_list
        self.is_running=True

    def stop(self):
        """Stops the thread."""
        self.is_running = False

    def apply_new_cmap(self, classification_path, image_path, overwrite = False):
        save_path = os.path.join(classification_path, 'check_classification')
        os.makedirs(save_path, exist_ok=True)
        imlist = [im for im in os.listdir(classification_path) if im.lower().endswith(('.tif','.tiff'))]
        count = 1
        for im in imlist:
            if im[-4:] == 'tiff' or im[-4:] == 'jpg2':
                im_jpg = im[:-4]
                im_jpg = im_jpg + 'jpg'
            else:
                im_jpg = im[:-3]
                im_jpg = im_jpg + 'jpg'
            print(f'  Applying colormap to image {count} of {len(imlist)}: {im}')
            if (not os.path.isfile(os.path.join(save_path, im_jpg))) or overwrite:
                im0 = cv2.imread(os.path.join(classification_path, im), cv2.IMREAD_GRAYSCALE)  # Mask
                im1 = cv2.imread(os.path.join(image_path, im))  # Image
                if im1 is None:
                    im1 = cv2.imread(os.path.join(image_path, im[:-3]+'png'))  # Image
                if im1 is None:
                    im1 = cv2.imread(os.path.join(image_path, im[:-3]+'jpg'))  # Image
                im1 = im1[:, :, ::-1]
                r = np.zeros_like(im0).astype(np.uint8)
                g = np.zeros_like(im0).astype(np.uint8)
                b = np.zeros_like(im0).astype(np.uint8)
                for l in range(1, len(self.classNames)):
                    idx = im0 == l
                    r[idx] = self.cmap[l - 1, 0]
                    g[idx] = self.cmap[l - 1, 1]
                    b[idx] = self.cmap[l - 1, 2]
                prediction_cmap = np.stack([r, g, b], axis=2)
                overlay = cv2.addWeighted(im1, 0.65, prediction_cmap, 0.35, 0)
                overlay = overlay[:,:,::-1]
                cv2.imwrite(os.path.join(save_path,im_jpg), overlay)
            else:
                print(f'  Image {im} already classified with this colormap')
            count += 1

    def run(self):
        progress = 0
        for index in range(self.list.count()):
            item = self.list.item(index)
            image_path = item.text()
            classification_path = os.path.join(item.text(), 'classification_' + self.nm + '_' + self.model_type)
            self.message_updated.emit(f'Classifying images from path {index+1}/{self.list.count()}: {image_path}')
            if self.is_running:
                classify_images(os.path.normpath(item.text()), os.path.join(self.train_fold, self.nm), self.model_type,
                            disp=False)
            else:
                self.finished_signal.emit('Cancel')
                return
            progress +=  1
            self.progress_updated.emit(progress)
            self.message_updated.emit(f'Quantifying images from path {index + 1}/{self.list.count()}: {image_path}')

            if self.is_running:
                quantify_images(os.path.join(self.train_fold, self.nm), os.path.normpath(item.text()))
            else:
                self.finished_signal.emit('Cancel')
                return
            progress += 1
            self.progress_updated.emit(progress)
            self.message_updated.emit(f'Changing colormap of images from path {index + 1}/{self.list.count()}: {image_path}')
            datafile = os.path.join(classification_path, 'cmap.pkl')
            try:
                with open(datafile, 'rb') as file:
                    data = pickle.load(file)
                    cmap = data['cmap']
                    if np.array_equal(cmap, self.cmap):
                        overwrite = False
                    else:
                        overwrite = True
            except:
                overwrite = True
            if self.is_running:
                if os.path.isdir(classification_path) and os.path.isdir(
                        os.path.join(classification_path, 'check_classification')) and not overwrite:
                    num_images = len([f for f in os.listdir(classification_path) if
                                      f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))])
                    num_check_images = len(
                        [f for f in os.listdir(os.path.join(classification_path, 'check_classification')) if
                         f.lower().endswith('.jpg')])
                    if num_images != num_check_images:
                        self.apply_new_cmap(classification_path, image_path)
                    else:
                        print('  All images already classified with this colormap')
                else:
                    self.apply_new_cmap(classification_path, image_path, overwrite)
            else:
                self.finished_signal.emit('Cancel')
                return
            progress +=  1
            self.progress_updated.emit(progress)
            self.message_updated.emit(f'Quantifying tissues of images from path {index + 1}/{self.list.count()}: {image_path}')
            quantify_tissues = []
            if self.is_running:
                for row in range(self.tissue_list.rowCount()):
                    item = self.tissue_list.item(row, 0)
                    if item.checkState() == Qt.Checked and (
                    not os.path.isfile(os.path.join(classification_path, item.text() + '_count_analysis.csv'))):
                        quantify_tissues.append(row + 1)
                quantify_tissues = list(set(quantify_tissues))
                if len(quantify_tissues) > 0:
                    for tissue in quantify_tissues:
                        quantify_objects(self.pthDL, classification_path, tissue)
            else:
                self.finished_signal.emit('Cancel')
                return
            progress += 1
            self.progress_updated.emit(progress)
            data = {'cmap': self.cmap.copy()}
            with open(datafile, 'wb') as f:
                pickle.dump(data, f)
        self.message_updated.emit('')
        self.finished_signal.emit('Completed')  # Send result back


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    # Load and apply the dark theme stylesheet
    with open('dark_theme.qss', 'r') as file:
        app.setStyleSheet(file.read())

    window = MainWindowClassify(r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\mouse lung','5x','10_27_2024')
    window.show()
    app.exec()