"""
Dialog Components for CODAvision GUI

This module provides dialog windows and UI components used in the CODAvision application
for tasks such as tissue analysis optimization, image selection, and progress display.

Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Updated: March 2025
"""

from PySide6.QtCore import QCoreApplication, QPoint, QRect, QSize, Qt, QMetaObject
from PySide6.QtGui import QFont, QPixmap, QCursor, QTransform
from PySide6.QtWidgets import (
    QApplication, QFrame, QLabel, QMainWindow, QPushButton, QSizePolicy,
    QWidget, QLineEdit, QListWidget, QDialog, QVBoxLayout, QProgressBar
)


from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSlider, QSizePolicy, QWidget, QLineEdit,QListWidget)

class Ui_choose_area(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        self.whole_im = QLabel(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.whole_im.setText("")

class Ui_disp_crop(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        self.cropped = QLabel(self.centralwidget)
        self.looks_good = QPushButton(self.centralwidget)
        self.new_loc = QPushButton(self.centralwidget)
        self.text = QLabel(self.centralwidget)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text.setFont(font)
        self.text.setLayoutDirection(Qt.LeftToRight)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.cropped.setText("")
        self.looks_good.setText(QCoreApplication.translate("MainWindow", u"Looks good", None))
        self.new_loc.setText(QCoreApplication.translate("MainWindow", u"No, select a new location", None))
        self.text.setText(QCoreApplication.translate("MainWindow", u"Is this a good location to evaluate tissue and whitespace detection?", None))

class Ui_choose_TA(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        self.TA_im = QLabel(self.centralwidget)
        self.TA_im.setFrameShape(QFrame.Box)
        self.apply = QPushButton(self.centralwidget)
        self.text = QLabel(self.centralwidget)
        self.slider_container = QWidget(self.centralwidget)
        self.TA_selection = QSlider(Qt.Horizontal,self.slider_container)
        self.TA_selection.setMinimum(0)  # Min value
        self.TA_selection.setMaximum(255)  # Max value
        self.TA_selection.setValue(205)  # Default value
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.slider_label = QLabel(self.slider_container)
        self.slider_label.setFont(QFont("Arial", 8, QFont.Bold))
        self.slider_label.setStyleSheet("background-color: #333333; border: 1px solid white; padding: 2px;")
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.text_TA = QLabel(self.centralwidget)
        self.text_TA.setFont(font)
        self.text_TA.setLayoutDirection(Qt.LeftToRight)
        self.text_TA.setAlignment(Qt.AlignCenter)
        self.text_mid = QLabel(self.centralwidget)
        self.text_mid.setFont(font)
        self.text_mid.setLayoutDirection(Qt.LeftToRight)
        self.text_mid.setAlignment(Qt.AlignCenter)
        self.text.setFont(font)
        self.text.setLayoutDirection(Qt.LeftToRight)
        self.text.setAlignment(Qt.AlignCenter)
        self.text_mode = QLabel(self.centralwidget)
        self.text_mode.setFont(QFont("Arial", 8, QFont.Bold))
        self.text_mode.setLayoutDirection(Qt.LeftToRight)
        self.text_mode.setAlignment(Qt.AlignCenter)
        self.medium_im = QLabel(self.centralwidget)
        self.medium_im.setFrameShape(QFrame.Box)
        self.change_mode = QPushButton(self.centralwidget)
        self.raise_ta = QPushButton(self.centralwidget)
        self.decrease_ta = QPushButton(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.TA_im.setText("")
        self.text.setText(QCoreApplication.translate("MainWindow", u"Select an intensity threshold so that the tissue in the binary image is marked in black", None))
        self.text_TA.setText(QCoreApplication.translate("MainWindow", u"Binary mask", None))
        self.text_mid.setText(QCoreApplication.translate("MainWindow", u"Original image", None))
        self.text_mode.setText(QCoreApplication.translate("MainWindow", u"Current mode: H&E", None))
        self.medium_im.setText("")
        self.apply.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.change_mode.setText(QCoreApplication.translate("MainWindow", u"Change mode", None))
        self.slider_label.setText(QCoreApplication.translate("MainWindow", u""+str(self.TA_selection.value()),None))
        self.raise_ta.setText(QCoreApplication.translate("MainWindow", u"More tissue", None))
        self.decrease_ta.setText(QCoreApplication.translate("MainWindow", u"More whitespace", None))

class Ui_use_current_TA(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        self.text = QLabel(self.centralwidget)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text.setFont(font)
        self.text.setLayoutDirection(Qt.LeftToRight)
        self.text.setAlignment(Qt.AlignCenter)
        self.keep_ta = QPushButton(self.centralwidget)
        self.new_ta = QPushButton(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.text.setText(QCoreApplication.translate("MainWindow", u"The tissue mask has already been evaluated.\n Do you want to choose a new tissue mask evaluation?", None))
        self.keep_ta.setText(QCoreApplication.translate("MainWindow", u"Keep current tissue mask evaluation", None))
        self.new_ta.setText(QCoreApplication.translate("MainWindow", u"Evaluate tissue mask again", None))

class Ui_choose_images_reevaluated(object):
    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        self.image_LE = QLineEdit(self.centralwidget)
        self.image_LE.setObjectName(u"image_LE")
        self.browse_PB = QPushButton(self.centralwidget)
        self.browse_PB.setObjectName(u"browse_PB")
        self.delete_PB = QPushButton(self.centralwidget)
        self.delete_PB.setObjectName(u"delete_PB")
        self.apply_PB = QPushButton(self.centralwidget)
        self.apply_PB.setObjectName(u"apply_PB")
        self.apply_all_PB = QPushButton(self.centralwidget)
        self.apply_all_PB.setObjectName(u"apply_all_PB")
        self.image_LW = QListWidget(self.centralwidget)
        self.image_LW.setObjectName(u"image_LW")
        self.image_LW.setEnabled(True)
        self.image_LW.setStyleSheet("QListWidget { border: 1px solid white; }")
        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.browse_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.delete_PB.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.apply_PB.setText(QCoreApplication.translate("MainWindow", u"Accept", None))
        self.apply_all_PB.setText(QCoreApplication.translate("MainWindow", u"Redo all images", None))


class LoadingDialog(QDialog):
    """Dialog displayed during loading operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading...")
        self.setModal(True)

        layout = QVBoxLayout()

        self.label = QLabel("Loading, please wait...")
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self.setFixedSize(200, 100)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)