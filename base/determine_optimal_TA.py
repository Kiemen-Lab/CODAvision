import numpy as np
from glob import glob
import os
import sys
import pickle
# Import pyqt stuff
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import Qt, QRect, QMetaObject, QCoreApplication
from networkx.classes import is_empty
from numpy.ma.extras import average

from .determine_optimal_TA_UIs import Ui_choose_area, Ui_disp_crop, Ui_choose_TA, Ui_choose_images_reevaluated, Ui_use_current_TA
import cv2

def determine_optimal_TA(pthim,numims):
    CT0 = 205
    szz = 600
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
            self.setGeometry(30+np.round(1500-szz+100)/2, 50,
                             np.round(szz+100), np.round(szz+300))
            self.ui.looks_good.setGeometry(QRect(50, 50+szz, szz/2, 90))
            self.ui.new_loc.setGeometry(QRect(50+szz/2, 50+szz, szz/2, 90))
            self.ui.text.setGeometry(QRect(50, 10, 900, 20))
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
            self.ui.cropped.setPixmap(pixmap.scaled(
                self.ui.cropped.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.cropped.setScaledContents(True)


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
        def __init__(self, szz, CTA, CT0, CTC, mode, parent=None):

            # Inherit from the aforementioned class and set up the gui
            super(chooseTA, self).__init__()
            self.ui = Ui_choose_TA()
            self.TA = CT0
            self.CTA = CTA
            self.CT0 = CT0
            self.CTC = CTC
            self.ui.setupUi(self, CTA, CT0, CTC)
            self.setGeometry(30+np.round(1500-1.5*szz+100)/2, 50,
                             np.round(1.5 * szz+100), np.round(370+szz/2))
            self.ui.raise_ta.setGeometry(QRect(50, 170+szz/2, 6+szz*0.75, 90))
            self.ui.decrease_ta.setGeometry(QRect(56+szz*0.75, 170+szz/2, 6+szz*0.75, 90))
            self.ui.high_ta.setGeometry(QRect(50, 80+szz/2, 6+szz*0.75, 90))
            self.ui.low_ta.setGeometry(QRect(56+szz*0.75, 80 + szz / 2, 6+szz*0.75, 90))
            self.ui.change_mode.setGeometry(QRect(50, 260 + szz / 2, 12+szz*1.5, 90))
            self.ui.text.setGeometry(QRect(50, 10, np.round(8+szz*1.5), 20))
            self.ui.text_high.setGeometry(QRect(50, 40, 3+szz/2, 20))
            self.ui.text_mid.setGeometry(QRect(54+szz/2, 40, 3 + szz / 2, 20))
            self.ui.text_low.setGeometry(QRect(58+szz, 40, 3 + szz / 2, 20))
            self.setWindowTitle("Which one of the images looks good?")
            self.do_again = 1
            self.mode = mode
            self.ui.change_mode.setText(f'Change mode \n Current mode: {mode}')
            if mode == 'H&E':
                self.ui.change_mode.setStyleSheet("background-color: white; color: black;")
                self.ui.change_mode.setText(f'Change mode \n Current mode: H&&E')
            else:
                self.ui.raise_ta.setText('Keep more whitespace')
                self.ui.decrease_ta.setText('Keep more tissue')
                self.ui.change_mode.setStyleSheet("""
                    QPushButton {
                        background-color: black;
                        color: white;
                        font-size: 12px;
                    }
                    QPushButton:hover {
                        background-color: #333333;  /* Dark gray instead of white */
                        color: white;  /* Keep the text readable */
                    }
                """)
            self.ui.change_mode.clicked.connect(self.on_mode)
            self.ui.high_ta.clicked.connect(self.on_high)
            self.ui.low_ta.clicked.connect(self.on_low)
            self.ui.raise_ta.clicked.connect(self.on_raise)
            self.ui.decrease_ta.clicked.connect(self.on_decrease)

        def on_high(self):
            self.do_again = 0
            self.TA = self.CTA
            self.close()

        def on_low(self):
            self.do_again = 0
            self.TA = self.CT0
            self.close()

        def on_raise(self):
            self.TA = self.CT0 + 10
            self.close()

        def on_decrease(self):
            self.TA = self.CT0 - 10
            self.close()


        def on_mode(self):
            if self.mode == 'H&E':
                self.mode = 'Grayscale'
            else:
                self.mode = 'H&E'
            self.close()

        def update_image(self, szz, cropped):
            if self.mode == 'H&E':
                image_array_high = np.ascontiguousarray(((cropped[:,:,1]>self.CTA)*255).astype(np.uint8))
                image_array_medium = np.ascontiguousarray(cropped)
                image_array_low = np.ascontiguousarray(((cropped[:,:,1]>self.CT0)*255).astype(np.uint8))
            else:
                image_array_high = np.ascontiguousarray(((cropped[:, :, 1] < self.CTA) * 255).astype(np.uint8))
                image_array_medium = np.ascontiguousarray(cropped)
                image_array_low = np.ascontiguousarray(((cropped[:, :, 1] < self.CT0) * 255).astype(np.uint8))

            height, width = image_array_medium.shape[:2]
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(image_array_medium.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.medium_im.setGeometry(QRect(54 + szz / 2, 70, szz / 2, szz / 2))
            self.ui.medium_im.setPixmap(pixmap.scaled(
                self.ui.medium_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.medium_im.setScaledContents(True)
            height, width = image_array_high.shape[:2]
            qimage = QtGui.QImage(image_array_high.data, width, height, image_array_high.strides[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.high_im.setGeometry(QRect(50, 70, szz/2, szz/2))
            self.ui.high_im.setPixmap(pixmap.scaled(
                self.ui.high_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.high_im.setScaledContents(True)
            qimage = QtGui.QImage(image_array_low.data, width, height, image_array_low.strides[0],
                                  QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.low_im.setGeometry(QRect(58+szz, 70, szz / 2, szz / 2))
            self.ui.low_im.setPixmap(pixmap.scaled(
                self.ui.low_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.low_im.setScaledContents(True)


    def select_TA(szz, cropped, CT0, mode):
        CTA = CT0 + 5
        CTC = CT0 - 5
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = chooseTA(szz,CTA,CT0,CTC, mode)
        window.show()
        window.update_image(szz, cropped)
        app.exec()
        return window.do_again, window.TA, window.mode


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
                                                                "","TIFF Images (*.tif *.tiff)")
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
                                             'Seleced image does not exist')

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
    if not imlist:
        jpg_files = glob(os.path.join(pthim, "*.jpg"))
        if jpg_files:
            imlist.extend(jpg_files)  # Add full paths of JPGs to list
        png_files = glob(os.path.join(pthim, '*.png'))
        if png_files:
            imlist.extend(png_files)
    if not imlist:
        print("No TIFF, PNG or JPG image files found in", pthim)
    print('   ')
    i = 0
    for image in imlist:
        imlist[i] = image[len(pthim)+1:]
        i += 1

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
            elif x_norm+szz>im0.shape[0]:
                cropped_temp = cropped_temp[:,-2*szz:, :]
            else:
                cropped_temp = cropped_temp[:,x_norm-szz:x_norm+szz,:]
            if cropped.size == 0:
                cropped = cropped_temp
            do_again = check_region(szz, cropped)
        do_again = 1
        while do_again == 1:
            do_again, CT0, mode = select_TA(szz, cropped, CT0, mode)
        if nm in cts:
            cts[nm] = CT0
        else:
            cts = {**cts, nm: CT0}
    with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'wb') as f:
        pickle.dump({'cts':cts,'imlist':imlist, 'mode': mode, 'average_TA': average_TA}, f)
    return