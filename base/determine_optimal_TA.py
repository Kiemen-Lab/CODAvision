import numpy as np
from glob import glob
import os
import sys
import pickle
# Import pyqt stuff
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import Qt, QRect, QMetaObject, QCoreApplication
from .determine_optimal_TA_UIs import Ui_choose_area, Ui_disp_crop, Ui_choose_TA
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
            self.ui.high_ta.setGeometry(QRect(50, 80+szz/2, 3+szz/2, 90))
            self.ui.medium_ta.setGeometry(QRect(54+szz/2, 80 + szz / 2, 4 +  szz / 2, 90))
            self.ui.low_ta.setGeometry(QRect(58+szz, 80 + szz / 2, 4+ szz / 2, 90))
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
            self.ui.medium_ta.clicked.connect(self.on_med)
            self.ui.raise_ta.clicked.connect(self.on_raise)
            self.ui.decrease_ta.clicked.connect(self.on_decrease)

        def on_high(self):
            self.do_again = 0
            self.TA = self.CTA
            self.close()

        def on_med(self):
            self.do_again = 0
            self.TA = self.CT0
            self.close()

        def on_low(self):
            self.do_again = 0
            self.TA = self.CTC
            self.close()

        def on_raise(self):
            self.TA = self.CT0 + 10
            self.close()

        def on_decrease(self):
            self.TA = self.CT0 - 10
            self.close()

        def on_mode(self):
            if self.mode == 'H&E':
                self.mode = 'MRI'
            else:
                self.mode = 'H&E'
            self.close()

        def update_image(self, szz, cropped):
            if self.mode == 'H&E':
                image_array_high = np.ascontiguousarray(((cropped[:,:,1]>self.CTA)*255).astype(np.uint8))
                image_array_medium = np.ascontiguousarray(((cropped[:,:,1]>self.CT0)*255).astype(np.uint8))
                image_array_low = np.ascontiguousarray(((cropped[:,:,1]>self.CTC)*255).astype(np.uint8))
            else:
                image_array_high = np.ascontiguousarray(((cropped[:, :, 1] < self.CTA) * 255).astype(np.uint8))
                image_array_medium = np.ascontiguousarray(((cropped[:, :, 1] < self.CT0) * 255).astype(np.uint8))
                image_array_low = np.ascontiguousarray(((cropped[:, :, 1] < self.CTC) * 255).astype(np.uint8))
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
            qimage = QtGui.QImage(image_array_medium.data, width, height, image_array_medium.strides[0],
                                  QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.ui.medium_im.setGeometry(QRect(54+szz/2, 70, szz / 2, szz / 2))
            self.ui.medium_im.setPixmap(pixmap.scaled(
                self.ui.medium_im.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.ui.medium_im.setScaledContents(True)
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
    numims = min(numims,len(imlist))
    imlist = np.random.choice(imlist, size = numims, replace=False)
    print(f'Evaluating {numims} randomly selected images to choose a good whitespace detection...')

    outpath = os.path.join(pthim,'TA')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if os.path.isfile(os.path.join(outpath,'TA_cutoff.pkl')):
        print('   Optimal cutoff already chosen, skip this step')
        return

    cts = np.zeros([1,numims])
    count = 0
    mode = 'H&E'
    for image in imlist:
        nm = image[len(pthim)+1:]
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
            cropped = im0[y_norm-szz:y_norm+szz, x_norm-szz:x_norm+szz, :]
            do_again = check_region(szz, cropped)
        do_again = 1
        while do_again == 1:
            do_again, CT0, mode = select_TA(szz, cropped, CT0, mode)
        cts[0,count-1] = CT0
    with open(os.path.join(outpath, 'TA_cutoff.pkl'), 'wb') as f:
        pickle.dump({'cts':cts,'imlist':imlist, 'mode': mode}, f)
    return