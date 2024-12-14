# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'classify_im.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt, QRectF)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QWidget, QDialog, QListWidget, QScrollArea, QMessageBox,
    QProgressBar, QVBoxLayout, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView)
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(865, 616)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(10, 190, 230, 340))
        self.colormap_TW = QTableWidget(self.groupBox_4)
        self.colormap_TW.setObjectName(u"colormap_TW")
        self.colormap_TW.horizontalHeader().setSectionsClickable(False)
        self.colormap_TW.verticalHeader().setSectionsClickable(False)
        self.colormap_TW.setGeometry(QRect(10, 30, 200, 300))
        self.change_color_PB = QPushButton(self.centralwidget)
        self.change_color_PB.setObjectName(u"change_color_PB")
        self.change_color_PB.setGeometry(QRect(125, 532, 113, 30))
        self.reset_PB = QPushButton(self.centralwidget)
        self.reset_PB.setObjectName(u"reset_PB")
        self.reset_PB.setGeometry(QRect(10, 532, 113, 30))
        self.classify_PB = QPushButton(self.centralwidget)
        self.classify_PB.setObjectName(u"classify_PB")
        self.classify_PB.setGeometry(QRect(630, 532, 200, 30))
        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(602, 191, 228, 339))
        self.component_TW = QTableWidget(self.groupBox_5)
        self.component_TW.setObjectName(u"component_TW")
        self.component_TW.setGeometry(QRect(11, 27, 206, 301))
        self.component_TW.horizontalHeader().setSectionsClickable(False)
        self.component_TW.verticalHeader().setSectionsClickable(False)
        self.overlay = ImageViewer()
        self.scene = QGraphicsScene()
        self.overlay = QGraphicsView(self.scene, self.centralwidget)
        self.overlay.setRenderHint(QPainter.Antialiasing)
        # self.overlay = QLabel(self.centralwidget)
        # self.overlay.setObjectName(u"overlay")
        self.overlay.setGeometry(QRect(245, 200, 350, 350))
        # self.overlay.setAutoFillBackground(True)
        self.zoom_in_PB = QPushButton(self.centralwidget)
        self.zoom_in_PB.setObjectName(u"zoom_in_PB")
        self.zoom_in_PB.setGeometry(QRect(570, 173, 25, 25))
        self.zoom_out_PB = QPushButton(self.centralwidget)
        self.zoom_out_PB.setObjectName(u"zoom_out_PB")
        self.zoom_out_PB.setGeometry(QRect(543, 173, 25, 25))
        self.classify_LW = QListWidget(self.centralwidget)
        self.classify_LW.setObjectName(u"classify_LW")
        self.classify_LW.setEnabled(True)
        self.classify_LW.setStyleSheet("QListWidget { border: 1px solid white; }")
        self.classify_LW.setGeometry(QRect(10, 60, 821, 90))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 22, 108, 24))
        self.classify_LE = QLineEdit(self.centralwidget)
        self.classify_LE.setObjectName(u"classify_LE")
        self.classify_LE.setGeometry(QRect(95, 22, 620, 24))
        self.classify_LE.setMaximumSize(QSize(16777215, 16777215))
        self.browseCl_PB = QPushButton(self.centralwidget)
        self.browseCl_PB.setObjectName(u"browseCl_PB")
        self.browseCl_PB.setGeometry(QRect(717, 22, 113, 24))
        self.deleteCl_PB = QPushButton(self.centralwidget)
        self.deleteCl_PB.setObjectName(u"deleteCl_PB")
        self.deleteCl_PB.setGeometry(QRect(717, 155, 113, 24))
        self.loading_dialog = LoadingDialog(self.centralwidget)
        self.retranslateUi(MainWindow)


    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Colormap", None))
        self.change_color_PB.setText(QCoreApplication.translate("MainWindow", u"Change color", None))
        self.reset_PB.setText(QCoreApplication.translate("MainWindow", u"Reset colormap", None))
        self.classify_PB.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.zoom_in_PB.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.zoom_out_PB.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Annotation class component analysis:", None))
        # self.overlay.setText("")
        self.classify_LE.setToolTip(QCoreApplication.translate("MainWindow", u"Path to training annotations", None))
        self.classify_LE.setStatusTip("")
        self.classify_LE.setWhatsThis("")
        self.classify_LE.setInputMask("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Path to images", None))
        self.browseCl_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.deleteCl_PB.setText(QCoreApplication.translate("MainWindow", u"Remove path", None))

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading...")
        self.setModal(True)  # Makes dialog modal
        # Create layout
        layout = QVBoxLayout()
        # Add loading label
        self.label = QLabel("Loading, please wait...")
        layout.addWidget(self.label)
        # Add progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Makes it an infinite progress bar
        layout.addWidget(self.progress)
        self.setLayout(layout)
        # Set fixed size
        self.setFixedSize(200, 100)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Create a QGraphicsScene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = None  # Start with no image
        self.scale_factor = 1.0
        #self.setRenderHint(Qt.Antialiasing)
        self.scene.setSceneRect(QRectF(0, 0, 350, 350))
          # Adjust scene rect to fit the new image

    def disp_im(self, pixmap):
        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.image_item.setPos(int(np.round((350 - pixmap.width()) / 2)), int(np.round((350 - pixmap.height()) / 2)))

    def zoom_in(self):
        self.scale(1.2, 1.2)  # Zoom in by 20%

    def zoom_out(self):
        self.scale(1 / 1.2, 1 / 1.2)  # Zoom out by 20%