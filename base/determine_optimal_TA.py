import numpy as np
from glob import glob
import os
import sys
import pickle
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import Qt, QRect
import cv2


import numpy as np
from glob import glob
import os
import sys
import pickle
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import Qt, QRect
import cv2

def determine_optimal_TA(pthim,pthtestim, numims):
    from gui.components.dialogs import (
        Ui_choose_area, Ui_disp_crop, Ui_choose_TA,
        Ui_choose_images_reevaluated, Ui_use_current_TA
    )
    CT0 = 205
    szz = 600
    print('Answer prompt pop-up window regarding tissue masks to proceed')
    class disp_whole_im(QtWidgets.QMainWindow):
        def __init__(self, shape, rsf, parent=None):

            # Inherit from the aforementioned class and set up the gui
            super(disp_whole_im, self).__init__()
            self.ui = Ui_choose_area()
            self.ui.setupUi(self)
            self.setWindowTitle("Click on a location at the edge of tissue and whitespace")
            self.clicked_position = None
            self.ui.whole_im.setCursor(QtCore.Qt.CrossCursor)
            self.setGeometry(30+np.round(1500-shape[1]*rsf)/2, 30+np.round(800-shape[0]*rsf)/2,
                             np.round(shape[1]*rsf), np.round(shape[0]*rsf))
        def update_image(self, im0, rsf):
            image_array = np.ascontiguousarray(im0)
            height, width = image_array.shape[:2]
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(image_array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.whole_im.setGeometry(0, 0, np.round(im0.shape[1]*rsf), np.round(im0.shape[0]*rsf))
            self.ui.whole_im.setPixmap(pixmap.scaled(
                self.ui.whole_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.whole_im.setScaledContents(True)

        def mousePressEvent(self, event):
            if self.ui.whole_im.geometry().contains(event.position().toPoint()):
                self.clicked_position = event.position() - QtCore.QPointF(self.ui.whole_im.geometry().topLeft())
                self.close()
    def display_image(im0, rsf):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = disp_whole_im(im0.shape,rsf)
        window.show()
        window.update_image(im0,rsf)
        app.exec()
        return window.clicked_position

    class disp_crop(QtWidgets.QMainWindow):
        def __init__(self, szz, parent=None):

            # Inherit from the aforementioned class and set up the gui
            super(disp_crop, self).__init__()
            self.ui = Ui_disp_crop()
            self.ui.setupUi(self)
            central_widget = self.ui.centralwidget
            main_layout = QtWidgets.QVBoxLayout()
            self.setCentralWidget(central_widget)
            self.ui.text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            main_layout.addWidget(self.ui.text,alignment=Qt.AlignCenter)
            self.ui.cropped.setFixedSize(szz, szz)
            self.ui.cropped.setAlignment(Qt.AlignCenter)
            self.ui.cropped.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            main_layout.addWidget(self.ui.cropped, alignment=Qt.AlignCenter)
            button_layout = QtWidgets.QHBoxLayout()
            button_layout.setSpacing(5)
            button_height = 90  # or any height you prefer
            self.ui.looks_good.setFixedHeight(button_height)
            self.ui.new_loc.setFixedHeight(button_height)
            self.ui.looks_good.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.ui.new_loc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            button_layout.addWidget(self.ui.looks_good)
            button_layout.addWidget(self.ui.new_loc)
            main_layout.addLayout(button_layout)
            central_widget.setLayout(main_layout)
            self.setWindowTitle("Check selected region")
            self.do_again = 1
            self.ui.looks_good.clicked.connect(self.on_good)
            self.ui.new_loc.clicked.connect(self.on_new)
        def on_good(self):
            self.do_again = 0
            self.close()

        def on_new(self):
            self.close()

        def update_image(self, szz, cropped):
            image_array = np.ascontiguousarray(cropped)
            height, width = image_array.shape[:2]
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(image_array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.cropped.setGeometry(QRect(50, 40, szz, szz))
            self.ui.cropped.setPixmap(pixmap)
            frame_geom = self.frameGeometry()
            screen = QtWidgets.QApplication.primaryScreen()
            center_point = screen.availableGeometry().center()
            frame_geom.moveCenter(center_point)
            self.move(frame_geom.topLeft())

    def check_region(szz, cropped):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = disp_crop(szz)
        window.show()
        window.update_image(szz, cropped)
        app.exec()
        return window.do_again

    class chooseTA(QtWidgets.QMainWindow):
        def __init__(self, szz, CT0, mode, parent=None):
            super(chooseTA, self).__init__()
            self.ui = Ui_choose_TA()
            self.TA = CT0
            self.CT0 = CT0
            self.ui.setupUi(self)
            central_widget = self.ui.centralwidget
            main_layout = QtWidgets.QVBoxLayout()
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            central_widget.setLayout(main_layout)
            self.setCentralWidget(central_widget)
            self.ui.text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            main_layout.addWidget(self.ui.text)
            HE_layout = QtWidgets.QVBoxLayout()
            self.ui.medium_im.setMinimumSize(150, 20)
            self.ui.text_TA.setMinimumSize(150, 20)
            HE_layout.addWidget(self.ui.text_mid)
            TA_layout = QtWidgets.QVBoxLayout()
            TA_layout.addWidget(self.ui.text_TA)
            images_layout = QtWidgets.QHBoxLayout()
            images_layout.setSpacing(4)
            self.ui.medium_im.setMinimumSize(300, 300)
            self.ui.medium_im.setMaximumSize(300, 300)
            self.ui.TA_im.setMinimumSize(300,300)
            self.ui.TA_im.setMaximumSize(300,300)
            mode_widget = QtWidgets.QWidget()
            mode_widget.setMaximumHeight(330)
            mode_layout = QtWidgets.QVBoxLayout(mode_widget)
            self.ui.change_mode.setMinimumSize(155, 50)
            mode_layout.addWidget(self.ui.change_mode)
            self.ui.text_mode.setMinimumSize(155, 30)
            mode_layout.addWidget(self.ui.text_mode)
            self.ui.slider_label.setMinimumSize(155, 30)
            mode_layout.addWidget(self.ui.slider_label)
            mode_layout.addStretch()
            mode_layout.setContentsMargins(0, 30, 0, 0)
            HE_layout.addWidget(self.ui.medium_im)
            images_layout.addLayout(HE_layout)
            TA_layout.addWidget(self.ui.TA_im)
            images_layout.addLayout(TA_layout)
            images_layout.addWidget(mode_widget)
            main_layout.addLayout(images_layout)
            slider_section = QtWidgets.QWidget()
            slider_layout = QtWidgets.QHBoxLayout(slider_section)
            self.ui.decrease_ta.setMinimumSize(120, 30)
            slider_layout.addWidget(self.ui.decrease_ta)
            slider_layout.addWidget(self.ui.slider_container, stretch=1)
            self.ui.raise_ta.setMinimumSize(120, 30)
            slider_layout.addWidget(self.ui.raise_ta)
            main_layout.addWidget(slider_section)
            save_container = QtWidgets.QWidget()
            save_layout = QtWidgets.QHBoxLayout(save_container)
            save_layout.addStretch()
            self.ui.apply.setMinimumSize(150, 50)
            save_layout.addWidget(self.ui.apply)
            main_layout.addWidget(save_container)
            self.stop = True
            self.setWindowTitle("Select an appropriate intensity threshold for the binary mask")
            self.mode = mode
            self.ui.text_mode.setText(f'Current mode: {mode}')
            if mode == 'H&E':
                self.ui.text_mode.setStyleSheet("background-color: white; color: black;")
                self.ui.text_mode.setText(f'Current mode: H&E')
            else:
                self.ui.text_mode.setStyleSheet("background-color: #333333; color: white;")
                self.ui.raise_ta.setText('More whitespace')
                self.ui.decrease_ta.setText('More tissue')
            self.ui.change_mode.clicked.connect(self.on_mode)
            self.ui.apply.clicked.connect(self.on_apply)
            self.ui.TA_selection.valueChanged.connect(self.update_slider)
            self.ui.TA_selection.setValue(self.CT0)
            self.ui.raise_ta.clicked.connect(self.on_raise)
            self.ui.decrease_ta.clicked.connect(self.on_decrease)

        def on_raise(self):
            self.CT0 = self.CT0 + 1
            self.ui.TA_selection.setValue(self.CT0)

        def on_decrease(self):
            self.CT0 = self.CT0 - 1
            self.ui.TA_selection.setValue(self.CT0)

        def update_slider(self):
            self.ui.slider_label.setText("Current threshold value: "+str(self.ui.TA_selection.value()))
            # handle_pos = self.slider_handle_position()
            # self.ui.slider_label.setGeometry(handle_pos-15, 0, 30, 20)
            self.on_change_TA()

        def slider_handle_position(self):
            """ Get the X coordinate of the slider handle. """
            option = QtWidgets.QStyleOptionSlider()
            self.ui.TA_selection.initStyleOption(option)
            handle_rect = self.ui.TA_selection.style().subControlRect(QtWidgets.QStyle.CC_Slider, option,
                                                                      QtWidgets.QStyle.SC_SliderHandle, self.ui.TA_selection)

            slider_x = self.ui.TA_selection.pos().x()  # Slider X position in parent widget
            return slider_x + handle_rect.x() + (handle_rect.width() // 2)


        def on_apply(self):
            self.TA = self.CT0
            self.stop = False
            self.close()

        def on_change_TA(self):
            self.CT0 = self.ui.TA_selection.value()
            self.change_TA('TA')

        def on_mode(self):
            if self.mode == 'H&E':
                self.mode = 'Grayscale'
                self.ui.TA_selection.setValue(50)
                self.CT0 = 50
            else:
                self.mode = 'H&E'
                self.ui.TA_selection.setValue(205)
                self.CT0 = 205
            self.change_TA('mode')

        def change_TA(self, change):
            if change == 'mode':
                if self.mode == 'H&E':
                    self.ui.text_mode.setStyleSheet("background-color: white; color: black;")
                    self.ui.text_mode.setText(f'Current mode: H&E')
                    self.ui.decrease_ta.setText('More whitespace')
                    self.ui.raise_ta.setText('More tissue')
                else:
                    self.ui.raise_ta.setText('More whitespace')
                    self.ui.decrease_ta.setText('More tissue')
                    self.ui.text_mode.setText(f'Current mode: {self.mode}')
                    self.ui.text_mode.setStyleSheet("background-color: #333333; color: white;")
            if self.mode == 'H&E':
                image_array = np.ascontiguousarray(((cropped[:, :, 1] > self.CT0) * 255).astype(np.uint8))
            else:
                image_array = np.ascontiguousarray(((cropped[:, :, 1] < self.CT0) * 255).astype(np.uint8))
            self.ui.apply.setText(f'Save')
            height, width = image_array.shape[:2]
            qimage = QtGui.QImage(image_array.data, width, height, image_array.strides[0],
                                  QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            # self.ui.TA_im.setGeometry(QRect(54 + szz*0.75, 70, szz / 2, szz / 2))
            self.ui.TA_im.setPixmap(pixmap.scaled(
                self.ui.TA_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            # self.ui.TA_im.setScaledContents(True)

        def update_image(self, szz, cropped):
            if self.mode == 'H&E':
                image_array_medium = np.ascontiguousarray(cropped)
                image_array = np.ascontiguousarray(((cropped[:,:,1]>self.CT0)*255).astype(np.uint8))
            else:
                image_array_medium = np.ascontiguousarray(cropped)
                image_array = np.ascontiguousarray(((cropped[:, :, 1] < self.CT0) * 255).astype(np.uint8))

            height, width = image_array_medium.shape[:2]
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(image_array_medium.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            # self.ui.medium_im.setGeometry(QRect(50 + szz / 4, 70, szz / 2, szz / 2))
            self.ui.medium_im.setPixmap(pixmap.scaled(
                self.ui.medium_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            # self.ui.medium_im.setScaledContents(True)
            height, width = image_array.shape[:2]
            qimage = QtGui.QImage(image_array.data, width, height, image_array.strides[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            # self.ui.TA_im.setGeometry(QRect(54+szz*0.75, 70, szz/2, szz/2))
            self.ui.TA_im.setPixmap(pixmap.scaled(
                self.ui.TA_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            # self.ui.TA_im.setScaledContents(True)


    def select_TA(szz, cropped, CT0, mode):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = chooseTA(szz,CT0, mode)
        window.show()
        window.update_image(szz, cropped)
        app.exec()
        return window.stop, window.TA, window.mode


    class confirm_TA_ui(QtWidgets.QMainWindow):
        def __init__(self):
            super(confirm_TA_ui, self).__init__()
            self.ui = Ui_use_current_TA()
            self.ui.setupUi(self)
            self.setGeometry(550, 300,
                             550, 130)
            self.ui.text.setGeometry(QRect(25, 10, 500, 60))
            self.ui.keep_ta.setGeometry(QRect(25, 80, 248, 40))
            self.ui.new_ta.setGeometry(QRect(275 , 80, 248, 40))
            self.setWindowTitle("Confirm tissue mask evaluation")
            self.ui.keep_ta.clicked.connect(self.on_keep_TA)
            self.ui.new_ta.clicked.connect(self.on_new_TA)


        def on_keep_TA(self):
            self.keep_TA = True
            self.close()

        def on_new_TA(self):
            self.keep_TA = False
            self.close()

    def confirm_TA():
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = confirm_TA_ui()
        window.show()
        app.exec()
        return window.keep_TA

    class choose_images_TA_ui(QtWidgets.QMainWindow):
        def __init__(self):
            super(choose_images_TA_ui, self).__init__()
            self.ui = Ui_choose_images_reevaluated()
            self.ui.setupUi(self)
            self.setGeometry(450, 300,
                            600, 250)
            self.ui.image_LE.setGeometry(QRect(25, 10, 448, 30))
            self.ui.image_LW.setGeometry(QRect(25, 45, 550, 100))
            self.ui.browse_PB.setGeometry(QRect(475, 10, 100, 30))
            self.ui.delete_PB.setGeometry(QRect(375, 150, 200, 30))
            self.ui.apply_PB.setGeometry(QRect(477, 185, 100, 30))
            self.ui.apply_all_PB.setGeometry(QRect(375, 185, 100, 30))
            self.setWindowTitle("Confirm tissue mask evaluation")
            self.ui.delete_PB.clicked.connect(self.on_delete_image)
            self.ui.browse_PB.clicked.connect(self.on_browse)
            self.ui.apply_PB.clicked.connect(self.on_apply)
            self.ui.apply_all_PB.clicked.connect(self.on_apply_all)
            self.images = []
            self.apply_all = False

        def closeEvent(self, event):
            if event.spontaneous():
                self.images = []
                QtWidgets.QMessageBox.information(self, 'Info',
                                              'Keeping previous tissue mask evaluation')

        def on_delete_image(self):
            delete_row = self.ui.image_LW.currentRow()
            if delete_row == -1:
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'Please select path to delete')
                return
            del self.images[delete_row]
            self.ui.image_LW.takeItem(delete_row)

        def on_browse(self):
            file_path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image to Reevaluate",
                                                                pthim,"TIFF Images (*.tif *.tiff)")
            if file_path:
                if os.path.isfile(file_path):
                    self.ui.image_LE.setText(file_path)
                    if not (file_path.endswith(('.tif')) or file_path.endswith(('.tiff'))):
                        QtWidgets.QMessageBox.warning(self, 'Warning',
                                                      'Select a .tif image.')
                    elif any(np.isin(self.images, file_path)):
                        QtWidgets.QMessageBox.warning(self, 'Warning',
                                                      'The selected image is already on the list')
                    else:
                        self.images.append(file_path)
                        self.ui.image_LW.addItem(file_path)
                    self.ui.image_LE.setText('')
            else:
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                             'Selected image does not exist')

        def on_apply(self):
            i = 0
            for image in self.images:
                self.images[i] = os.path.basename(image)
                i +=1
            self.close()

        def on_apply_all(self):
            self.apply_all = True
            self.close()

    def choose_images_TA():
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = choose_images_TA_ui()
        window.show()
        app.exec()
        return window.apply_all, window.images

    imlist = sorted(glob(os.path.join(pthim, '*.tif')))
    imtestlist = sorted(glob(os.path.join(pthtestim, '*.tif')))
    if not imlist:
        jpg_files = glob(os.path.join(pthim, "*.jpg"))
        if jpg_files:
            imlist.extend(jpg_files)  # Add full paths of JPGs to list
        jpg_files = glob(os.path.join(pthtestim, "*.jpg"))
        if jpg_files:
            imtestlist.extend(jpg_files)  # Add full paths of JPGs to list
        png_files = glob(os.path.join(pthim, '*.png'))
        if png_files:
            imlist.extend(png_files)
        png_files = glob(os.path.join(pthtestim, '*.png'))
        if png_files:
            imtestlist.extend(png_files)
    if not imlist:
        print(f"No TIFF, PNG or JPG image files found in either {pthim} or {pthtestim}")
    print('   ')
    i = 0
    for image in imlist:
        imlist[i] = image[len(pthim)+1:]
        i += 1
    imlist.extend(imtestlist)

    outpath = os.path.join(pthim, 'TA')
    cts = {}
    mode = 'H&E'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if os.path.isfile(os.path.join(outpath,'TA_cutoff.pkl')):
        if numims>0:
            keep_TA = confirm_TA()
            if keep_TA:
                print('   Optimal cutoff already chosen, skip this step')
                return
            with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'rb') as f:
                data = pickle.load(f)
                mode = data['mode']
        else:
            with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'rb') as f:
                data = pickle.load(f)
                cts = data['cts']
                mode = data['mode']
            done = []
            for index in cts:
                done.append(index)
            imlist_temp = list(set(imlist) - set(done))
            if not imlist_temp:
                keep_TA = confirm_TA()
                if keep_TA:
                    print('   Optimal cutoff already chosen for all images, skip this step')
                    return
                else:
                    apply_all, redo_list = choose_images_TA()
                    if not apply_all:
                        imlist = redo_list
    if mode == 'Grayscale':
        CT0 = 50
    if numims>0:
        numims = min(numims,len(imlist))
        imlist = np.random.choice(imlist, size=numims, replace=False)
        print(f'Evaluating {numims} randomly selected images to choose a good whitespace detection...')
        average_TA = True
    else:
        numims = len(imlist)
        print(f'Evaluating all training images to choose a good whitespace detection...')
        average_TA = False

    count = 0
    for nm in imlist:
        count += 1
        if len(nm)>len(pthtestim) and nm[:len(pthtestim)]==pthtestim:
            print(f'    Loading image {count} of {numims}: {nm[len(pthtestim)+1:]}')
            im0 = cv2.imread(nm)
        else:
            print(f'    Loading image {count} of {numims}: {nm}')
            im0 = cv2.imread(os.path.join(pthim,nm))
        im0 = im0[:,:,::-1]
        print('     Image loaded')
        rsf = min(1500/im0.shape[1], 780/im0.shape[0])
        do_again = 1
        while do_again == 1:
            click = display_image(im0,rsf)
            x = click.x()
            y = click.y()
            x_norm = int(np.round(x/rsf))
            y_norm = int(np.round(y/rsf))
            cropped_temp = im0[:,:,:]
            cropped = np.array([])
            if 2*szz>im0.shape[0] and 2*szz>im0.shape[1]:
                max_dim = np.max([im0.shape[0],im0.shape[1]])
                pad_x = (max_dim - im0.shape[0]) // 2
                pad_y = (max_dim - im0.shape[1]) // 2
                # Ensure even padding (if odd difference, add extra on one side)
                pad_x1, pad_x2 = pad_x, pad_x + (max_dim - im0.shape[0]) % 2
                pad_y1, pad_y2 = pad_y, pad_y + (max_dim - im0.shape[1]) % 2
                cropped = np.pad(im0, ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)), mode='constant')
            elif 2*szz>im0.shape[0]:
                pad_x = (2*szz - im0.shape[0]) // 2
                pad_x1, pad_x2 = pad_x, pad_x + (2*szz - im0.shape[0]) % 2
                padded = np.pad(im0, ((pad_x1, pad_x2), (0, 0), (0, 0)), mode='constant')
                cropped_temp = padded[y_norm-szz:y_norm+szz,x_norm-szz:x_norm+szz, :]
            elif 2*szz>im0.shape[1]:
                pad_y = (2*szz - im0.shape[1]) // 2
                pad_y1, pad_y2 = pad_y, pad_y + (2*szz - im0.shape[1]) % 2
                padded = np.pad(im0, ((0,0), (pad_y1, pad_y2), (0, 0)), mode='constant')
                cropped_temp = padded[y_norm-szz:y_norm+szz,x_norm-szz:x_norm+szz, :]
            if y_norm<szz:
                cropped_temp = cropped_temp[0: 2*szz, :, :]
            elif y_norm+szz>im0.shape[0]:
                cropped_temp = cropped_temp[-2*szz:, :, :]
            else:
                cropped_temp = cropped_temp[y_norm-szz:y_norm+szz,:,:]
            if x_norm<szz:
                cropped_temp = cropped_temp[:,0: 2*szz, :]
            elif x_norm+szz>im0.shape[1]:
                cropped_temp = cropped_temp[:,-2*szz:, :]
            else:
                cropped_temp = cropped_temp[:,x_norm-szz:x_norm+szz,:]
            if cropped.size == 0:
                cropped = cropped_temp
            do_again = check_region(szz, cropped)
        sstop, CT0, mode = select_TA(szz, cropped, CT0, mode)
        if sstop:
            print('Whitespace detection process stopped by the user')
            return
        if nm in cts:
            cts[nm] = CT0
        else:
            cts = {**cts, nm: CT0}
    with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'wb') as f:
        pickle.dump({'cts':cts,'imlist':imlist, 'mode': mode, 'average_TA': average_TA}, f)
    return