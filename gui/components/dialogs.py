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


class Ui_choose_area:
    """UI definition for the area selection dialog."""

    def setupUi(self, MainWindow):
        self.centralwidget = QWidget(MainWindow)
        self.whole_im = QLabel(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.whole_im.setText("")


class Ui_disp_crop:
    """UI definition for the crop display dialog."""

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
        self.text.setText(QCoreApplication.translate("MainWindow",
                                                     u"Is this a good location to evaluate tissue and whitespace detection?",
                                                     None))


class Ui_choose_TA:
    """UI definition for the tissue analysis selection dialog."""

    def setupUi(self, MainWindow, CTA, CT0, CTC):
        self.centralwidget = QWidget(MainWindow)
        self.high_im = QLabel(self.centralwidget)
        self.high_im.setFrameShape(QFrame.Box)
        self.high_ta = QPushButton(self.centralwidget)
        self.low_ta = QPushButton(self.centralwidget)
        self.text = QLabel(self.centralwidget)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text_high = QLabel(self.centralwidget)
        self.text_high.setFont(font)
        self.text_high.setLayoutDirection(Qt.LeftToRight)
        self.text_high.setAlignment(Qt.AlignCenter)
        self.text_low = QLabel(self.centralwidget)
        self.text_low.setFont(font)
        self.text_low.setLayoutDirection(Qt.LeftToRight)
        self.text_low.setAlignment(Qt.AlignCenter)
        self.text_mid = QLabel(self.centralwidget)
        self.text_mid.setFont(font)
        self.text_mid.setLayoutDirection(Qt.LeftToRight)
        self.text_mid.setAlignment(Qt.AlignCenter)
        self.text.setFont(font)
        self.text.setLayoutDirection(Qt.LeftToRight)
        self.text.setAlignment(Qt.AlignCenter)
        self.medium_im = QLabel(self.centralwidget)
        self.medium_im.setFrameShape(QFrame.Box)
        self.low_im = QLabel(self.centralwidget)
        self.low_im.setFrameShape(QFrame.Box)
        self.medium_ta = QPushButton(self.centralwidget)
        self.raise_ta = QPushButton(self.centralwidget)
        self.decrease_ta = QPushButton(self.centralwidget)
        self.change_mode = QPushButton(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow, CTA, CT0, CTC)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow, CTA, CT0, CTC):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.high_im.setText("")
        self.high_ta.setText(QCoreApplication.translate("MainWindow", u"Image A\nTA = " + str(CTA), None))
        self.low_ta.setText(QCoreApplication.translate("MainWindow", u"Image B\nTA = " + str(CT0), None))
        self.text.setText(
            QCoreApplication.translate("MainWindow", u"Select the image where only the tissue is marked in black",
                                       None))
        self.text_high.setText(QCoreApplication.translate("MainWindow", u"Image A", None))
        self.text_mid.setText(QCoreApplication.translate("MainWindow", u"Original image", None))
        self.text_low.setText(QCoreApplication.translate("MainWindow", u"Image B", None))
        self.medium_im.setText("")
        self.low_im.setText("")
        self.raise_ta.setText(QCoreApplication.translate("MainWindow", u"Keep more tissue", None))
        self.decrease_ta.setText(QCoreApplication.translate("MainWindow", u"Keep more whitespace", None))
        self.change_mode.setText(
            QCoreApplication.translate("MainWindow", u"Change mode \n Current mode: White background", None))


class Ui_use_current_TA:
    """UI definition for the current tissue analysis confirmation dialog."""
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
        self.text.setText(QCoreApplication.translate("MainWindow",
                                                     u"The tissue mask has already been evaluated.\n Do you want to choose a new tissue mask evaluation?",
                                                     None))
        self.keep_ta.setText(QCoreApplication.translate("MainWindow", u"Keep current tissue mask evaluation", None))
        self.new_ta.setText(QCoreApplication.translate("MainWindow", u"Evaluate tissue mask again", None))


class Ui_choose_images_reevaluated:
    """UI definition for the image reevaluation selection dialog."""

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