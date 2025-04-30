"""
UI Definitions for CODAvision GUI

This module contains the UI definitions generated from Qt Designer UI files.
These classes define the layout and properties of the GUI components.

Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Updated: March 2025
"""
import os.path

import numpy as np
from PySide6.QtCore import (
    QCoreApplication, QMetaObject, QRect, QSize, Qt, QRectF, QPoint
)
from PySide6.QtGui import QFont, QPixmap, QPainter, QIcon
from PySide6.QtWidgets import (
    QWidget, QGroupBox, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QTabWidget, QCheckBox,
    QTableWidget, QHeaderView, QStatusBar, QGraphicsScene,
    QGraphicsView, QDialog, QProgressBar, QVBoxLayout,
    QMessageBox, QGraphicsPixmapItem, QListWidget, QGridLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QTableView
)


class Ui_MainWindow:
    """
    UI definition for the main window of CODAvision.
    """
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(706, 453)

        # Set the application icon
        MainWindow.setWindowIcon(QIcon("gui/resources/logoCODAvision.png"))

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_6 = QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(True)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_3 = QGridLayout(self.tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_4 = QLabel(self.tab)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.resolution_CB = QComboBox(self.tab)
        self.resolution_CB.addItem("")
        self.resolution_CB.addItem("")
        self.resolution_CB.addItem("")
        self.resolution_CB.addItem("")
        self.resolution_CB.addItem("")
        self.resolution_CB.setObjectName(u"resolution_CB")

        self.horizontalLayout_3.addWidget(self.resolution_CB)

        self.horizontalSpacer_3 = QSpacerItem(416, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_3, 3, 1, 1, 2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.model_name = QLineEdit(self.tab)
        self.model_name.setObjectName(u"model_name")

        self.horizontalLayout_6.addWidget(self.model_name)

        self.gridLayout_3.addLayout(self.horizontalLayout_6, 3, 0, 1, 1)
        self.gridLayout_3.addLayout(self.horizontalLayout_3, 4, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 187, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer, 5, 0, 1, 1)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer = QSpacerItem(348, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer)

        self.Save_FL_PB = QPushButton(self.tab)
        self.Save_FL_PB.setObjectName(u"Save_FL_PB")

        self.horizontalLayout_9.addWidget(self.Save_FL_PB)

        self.gridLayout_3.addLayout(self.horizontalLayout_9, 6, 0, 1, 3)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.trianing_LE = QLineEdit(self.tab)
        self.trianing_LE.setObjectName(u"trianing_LE")
        self.trianing_LE.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout.addWidget(self.trianing_LE)

        self.trainin_PB = QPushButton(self.tab)
        self.trainin_PB.setObjectName(u"trainin_PB")

        self.horizontalLayout.addWidget(self.trainin_PB)

        self.gridLayout_3.addLayout(self.horizontalLayout, 1, 0, 1, 3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.testing_LE = QLineEdit(self.tab)
        self.testing_LE.setObjectName(u"testing_LE")

        self.horizontalLayout_2.addWidget(self.testing_LE)

        self.testing_PB = QPushButton(self.tab)
        self.testing_PB.setObjectName(u"testing_PB")

        self.horizontalLayout_2.addWidget(self.testing_PB)

        self.gridLayout_3.addLayout(self.horizontalLayout_2, 2, 0, 1, 3)

        self.label_42 = QLabel(self.tab)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setGeometry(QRect(9, 195, 120, 24))
        self.custom_img_LE = QLineEdit(self.tab)
        self.custom_img_LE.setObjectName(u"custom_img_LE")
        self.custom_img_LE.setGeometry(QRect(100, 199, 520, 20))
        self.custom_img_PB = QPushButton(self.tab)
        self.custom_img_PB.setObjectName(u"custom_img_PB")
        self.custom_img_PB.setGeometry(QRect(626, 194, 51, 28))
        self.label_42.setVisible(False)
        self.custom_img_LE.setVisible(False)
        self.custom_img_PB.setVisible(False)

        self.label_44 = QLabel(self.tab)
        self.label_44.setObjectName(u"label_44")
        self.label_44.setGeometry(QRect(9, 229, 120, 24))
        self.custom_test_img_LE = QLineEdit(self.tab)
        self.custom_test_img_LE.setObjectName(u"custom_test_img_LE")
        self.custom_test_img_LE.setGeometry(QRect(100, 233, 520, 20))
        self.custom_test_img_PB = QPushButton(self.tab)
        self.custom_test_img_PB.setObjectName(u"custom_test_img_PB")
        self.custom_test_img_PB.setGeometry(QRect(626, 228, 51, 28))
        self.label_44.setVisible(False)
        self.custom_test_img_LE.setVisible(False)
        self.custom_test_img_PB.setVisible(False)

        self.label_43 = QLabel(self.tab)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setGeometry(QRect(270, 134, 100, 24))
        self.custom_scale_LE = QLineEdit(self.tab)
        self.custom_scale_LE.setObjectName(u"custom_scale_LE")
        self.custom_scale_LE.setGeometry(QRect(350, 138, 120, 20))
        self.label_43.setVisible(False)
        self.custom_scale_LE.setVisible(False)

        self.label_45 = QLabel(self.tab)
        self.label_45.setObjectName(u"label_45")
        self.label_45.setGeometry(QRect(490, 134, 120, 24))
        self.label_45.setVisible(False)
        self.use_anotated_images_CB = QCheckBox(self.tab)
        self.use_anotated_images_CB.setGeometry(QRect(625, 140, 15, 15))
        self.use_anotated_images_CB.setVisible(False)
        self.label_47 = QLabel(self.tab)
        self.label_47.setObjectName(u"label_47")
        self.label_47.setGeometry(QRect(490, 161, 120, 24))
        self.label_47.setVisible(False)
        self.create_downsample_CB = QCheckBox(self.tab)
        self.create_downsample_CB.setGeometry(QRect(625, 167, 15, 15))
        self.create_downsample_CB.setVisible(False)

        self.prerecorded_PB = QPushButton(self.tab)
        self.prerecorded_PB.setObjectName(u"prerecorded_PB")
        self.prerecorded_PB.setEnabled(True)
        self.gridLayout_3.addWidget(self.prerecorded_PB, 0, 2, 1, 1)

        self.classify_PB = QPushButton(self.tab)
        self.classify_PB.setObjectName(u"classify_PB")
        self.classify_PB.setEnabled(False)
        self.classify_PB.setStyleSheet("background-color: green; color: white;")
        self.classify_PB.setGeometry(QRect(253, 9, 205, 28))
        self.classify_PB.setVisible(False)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_7 = QGridLayout(self.tab_2)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.tissue_segmentation_TW = QTableWidget(self.tab_2)
        self.tissue_segmentation_TW.setObjectName(u"tissue_segmentation_TW")
        self.tissue_segmentation_TW.setEnabled(True)
        self.tissue_segmentation_TW.horizontalHeader().setSectionsClickable(False)
        self.tissue_segmentation_TW.verticalHeader().setSectionsClickable(False)

        self.gridLayout_7.addWidget(self.tissue_segmentation_TW, 0, 0, 2, 1)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox = QGroupBox(self.tab_2)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_4 = QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.wsoptions_CB = QComboBox(self.groupBox)
        self.wsoptions_CB.addItem("")
        self.wsoptions_CB.addItem("")
        self.wsoptions_CB.addItem("")
        self.wsoptions_CB.addItem("")
        self.wsoptions_CB.setObjectName(u"wsoptions_CB")

        self.gridLayout_2.addWidget(self.wsoptions_CB, 0, 0, 1, 1)

        self.apply_PB = QPushButton(self.groupBox)
        self.apply_PB.setObjectName(u"apply_PB")

        self.gridLayout_2.addWidget(self.apply_PB, 0, 1, 1, 1)

        self.applyall_PB = QPushButton(self.groupBox)
        self.applyall_PB.setObjectName(u"applyall_PB")

        self.gridLayout_2.addWidget(self.applyall_PB, 0, 2, 1, 1)

        self.gridLayout_4.addLayout(self.gridLayout_2, 0, 0, 1, 1)

        self.verticalLayout_2.addWidget(self.groupBox)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_6 = QLabel(self.tab_2)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.addws_CB = QComboBox(self.tab_2)
        self.addws_CB.addItem("")
        self.addws_CB.setObjectName(u"addws_CB")

        self.horizontalLayout_4.addWidget(self.addws_CB)

        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_7 = QLabel(self.tab_2)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_5.addWidget(self.label_7)

        self.addnonws_CB = QComboBox(self.tab_2)
        self.addnonws_CB.addItem("")
        self.addnonws_CB.setObjectName(u"addnonws_CB")

        self.horizontalLayout_5.addWidget(self.addnonws_CB)

        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.changecolor_PB = QPushButton(self.tab_2)
        self.changecolor_PB.setObjectName(u"changecolor_PB")

        self.gridLayout.addWidget(self.changecolor_PB, 0, 0, 1, 1)

        self.Reset_PB = QPushButton(self.tab_2)
        self.Reset_PB.setObjectName(u"Reset_PB")
        self.Reset_PB.setEnabled(True)

        self.gridLayout.addWidget(self.Reset_PB, 0, 1, 1, 1)

        self.Combine_PB = QPushButton(self.tab_2)
        self.Combine_PB.setObjectName(u"Combine_PB")

        self.gridLayout.addWidget(self.Combine_PB, 1, 0, 1, 1)

        self.delete_PB = QPushButton(self.tab_2)
        self.delete_PB.setObjectName(u"delete_PB")

        self.gridLayout.addWidget(self.delete_PB, 1, 1, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)

        self.verticalSpacer_3 = QSpacerItem(17, 142, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_3)

        self.gridLayout_7.addLayout(self.verticalLayout_2, 0, 1, 1, 1)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_2 = QSpacerItem(128, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_2)

        self.return_ts_PB = QPushButton(self.tab_2)
        self.return_ts_PB.setObjectName(u"return_ts_PB")

        self.horizontalLayout_7.addWidget(self.return_ts_PB)

        self.save_ts_PB = QPushButton(self.tab_2)
        self.save_ts_PB.setObjectName(u"save_ts_PB")

        self.horizontalLayout_7.addWidget(self.save_ts_PB)

        self.gridLayout_7.addLayout(self.horizontalLayout_7, 1, 1, 1, 1)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_5 = QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.nesting_TW = QTableView(self.tab_3)
        self.nesting_TW.setObjectName(u"nesting_TW")

        self.gridLayout_5.addWidget(self.nesting_TW, 0, 0, 2, 1)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.moveup_PB = QPushButton(self.tab_3)
        self.moveup_PB.setObjectName(u"moveup_PB")

        self.verticalLayout_3.addWidget(self.moveup_PB)

        self.Movedown_PB = QPushButton(self.tab_3)
        self.Movedown_PB.setObjectName(u"Movedown_PB")

        self.verticalLayout_3.addWidget(self.Movedown_PB)
        self.hover_label = HoverPreviewLabel(
            os.path.join(os.getcwd(), 'gui', 'components', 'images', "Question_mark_icon.png"),
            os.path.join(os.getcwd(), 'gui', 'components', 'images', "Nesting_example.jpg"),
            self.tab_3)
        self.verticalLayout_3.addWidget(self.hover_label)

        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.gridLayout_5.addLayout(self.verticalLayout_4, 0, 1, 1, 1)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.nesting_checkBox = QCheckBox(self.tab_3)
        self.nesting_checkBox.setObjectName(u"nesting_checkBox")

        self.verticalLayout_5.addWidget(self.nesting_checkBox)
        self.verticalSpacer_4 = QSpacerItem(20, 161, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_4)

        self.gridLayout_5.addLayout(self.verticalLayout_5, 0, 2, 1, 2)

        self.horizontalSpacer_4 = QSpacerItem(438, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_4, 1, 1, 1, 2)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.AS_PB = QPushButton(self.tab_3)
        self.AS_PB.setObjectName(u'AS_PB')

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.return_nesting_PB = QPushButton(self.tab_3)
        self.return_nesting_PB.setObjectName(u"return_nesting_PB")
        self.return_nesting_PB.setGeometry(QRect(200, 400, 140, 30))

        self.horizontalLayout_8.addWidget(self.return_nesting_PB)

        self.save_nesting_PB = QPushButton(self.tab_3)
        self.save_nesting_PB.setObjectName(u"save_nesting_PB")
        self.close_nesting_PB = QPushButton(self.tab_3)
        self.close_nesting_PB.setObjectName(u"close_nesting_PB")
        self.horizontalLayout_8.addWidget(self.close_nesting_PB)
        self.horizontalLayout_8.addWidget(self.save_nesting_PB)
        self.horizontalLayout_8.addWidget(self.AS_PB)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        container = QWidget(self.tab_3)
        container.setLayout(self.verticalLayout_6)
        container.setGeometry(QRect(275, 340, 410, 45))

        # self.gridLayout_5.addLayout(self.verticalLayout_6, 1, 3, 1, 1)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.gridLayout_9 = QGridLayout(self.tab_4)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalSpacer_6 = QSpacerItem(438, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_6)

        self.return_ad_PB = QPushButton(self.tab_4)
        self.return_ad_PB.setObjectName(u"return_ad_PB")

        self.horizontalLayout_15.addWidget(self.return_ad_PB)

        self.save_ad_PB = QPushButton(self.tab_4)
        self.save_ad_PB.setObjectName(u"save_ad_PB")
        self.close_ad_PB = QPushButton(self.tab_4)
        self.close_ad_PB.setObjectName(u"close_ad_PB")

        self.horizontalLayout_15.addWidget(self.close_ad_PB)
        self.horizontalLayout_15.addWidget(self.save_ad_PB)

        self.gridLayout_9.addLayout(self.horizontalLayout_15, 3, 0, 1, 2)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_5 = QLabel(self.tab_4)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_11.addWidget(self.label_5)

        self.tts_CB = QComboBox(self.tab_4)
        self.tts_CB.addItem("")
        self.tts_CB.addItem("")
        self.tts_CB.addItem("")
        self.tts_CB.addItem("")
        self.tts_CB.addItem("")
        self.tts_CB.addItem("")
        self.tts_CB.setObjectName(u"tts_CB")

        self.horizontalLayout_11.addWidget(self.tts_CB)

        self.verticalLayout_10.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_10 = QLabel(self.tab_4)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_12.addWidget(self.label_10)

        self.ttn_SB = QSpinBox(self.tab_4)
        self.ttn_SB.setObjectName(u"ttn_SB")
        self.ttn_SB.setMaximum(100)
        self.ttn_SB.setSingleStep(1)
        self.ttn_SB.setValue(15)
        self.ttn_SB.lineEdit().setReadOnly(True)

        self.horizontalLayout_12.addWidget(self.ttn_SB)

        self.verticalLayout_10.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_11 = QLabel(self.tab_4)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_13.addWidget(self.label_11)

        self.vtn_SB = QSpinBox(self.tab_4)
        self.vtn_SB.setObjectName(u"vtn_SB")
        self.vtn_SB.setMaximum(100)
        self.vtn_SB.setSingleStep(1)
        self.vtn_SB.setValue(3)
        self.vtn_SB.lineEdit().setReadOnly(True)

        self.horizontalLayout_13.addWidget(self.vtn_SB)

        self.verticalLayout_10.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_12 = QLabel(self.tab_4)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_14.addWidget(self.label_12)

        self.TA_SB = QSpinBox(self.tab_4)
        self.TA_SB.setObjectName(u"TA_SB")
        self.TA_SB.setMaximum(10)
        self.TA_SB.setMinimum(1)
        self.TA_SB.setSingleStep(1)
        self.TA_SB.setValue(3)
        self.TA_SB.setDisplayIntegerBase(10)
        self.TA_SB.lineEdit().setReadOnly(True)
        self.TA_CB = QCheckBox(self.tab_4)
        self.TA_CB.setObjectName(u"TA_CB")
        self.label_69 = QLabel(self.tab_4)
        self.label_69.setObjectName(u"label_69")

        self.horizontalLayout_14.addWidget(self.TA_SB)
        self.horizontalLayout_14.addSpacing(40)
        self.horizontalLayout_14.addWidget(self.label_69)
        self.horizontalLayout_14.addWidget(self.TA_CB)
        self.verticalLayout_10.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.model_name_label = QLabel(self.tab_4)
        self.model_name_label.setObjectName(u"model_name_label")
        self.horizontalLayout_15.addWidget(self.model_name_label)
        self.model_type_CB = QComboBox(self.tab_4)
        self.model_type_CB.addItem("DeepLabV3_plus")
        self.model_type_CB.addItem("UNet")
        # self.model_type_CB.addItem("UNet3_plus")
        # self.model_type_CB.addItem("TransUNet")
        # self.model_type_CB.addItem("CASe_UNet")
        self.model_type_CB.setObjectName(u"model_type_CB")
        self.horizontalLayout_15.addWidget(self.model_type_CB)
        self.verticalLayout_10.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.batch_label = QLabel(self.tab_4)
        self.batch_label.setObjectName(u"batch_label")
        self.horizontalLayout_16.addWidget(self.batch_label)
        self.batch_size_SB = QSpinBox(self.tab_4)
        self.batch_size_SB.setObjectName(u"batch_size")
        self.batch_size_SB.setMaximum(10)
        self.batch_size_SB.setSingleStep(1)
        self.batch_size_SB.setValue(3)
        self.batch_size_SB.setDisplayIntegerBase(10)
        self.batch_size_SB.lineEdit().setReadOnly(True)
        self.batch_size_SB.setObjectName(u"batch_size_SB")
        self.horizontalLayout_16.addWidget(self.batch_size_SB)
        self.verticalLayout_10.addLayout(self.horizontalLayout_16)

        self.gridLayout_9.addLayout(self.verticalLayout_10, 0, 0, 1, 1)

        self.groupBox_3 = QGroupBox(self.tab_4)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_8 = QGridLayout(self.groupBox_3)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.component_TW = QTableWidget(self.groupBox_3)
        self.component_TW.setObjectName(u"component_TW")
        self.component_TW.horizontalHeader().setSectionsClickable(False)
        self.component_TW.verticalHeader().setSectionsClickable(False)

        self.verticalLayout_9.addWidget(self.component_TW)

        self.gridLayout_8.addLayout(self.verticalLayout_9, 0, 0, 1, 1)

        self.gridLayout_9.addWidget(self.groupBox_3, 2, 0, 1, 1)

        self.tabWidget.addTab(self.tab_4, "")

        self.gridLayout_6.addWidget(self.tabWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.tts_CB.setCurrentIndex(4)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        # if QT_CONFIG(whatsthis)
        self.tabWidget.setWhatsThis("")
        # endif // QT_CONFIG(whatsthis)
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Resolution", None))
        self.resolution_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"Select", None))
        self.resolution_CB.setItemText(1, QCoreApplication.translate("MainWindow", u"1x", None))
        self.resolution_CB.setItemText(2, QCoreApplication.translate("MainWindow", u"5x", None))
        self.resolution_CB.setItemText(3, QCoreApplication.translate("MainWindow", u"10x", None))
        self.resolution_CB.setItemText(4, QCoreApplication.translate("MainWindow", u"Custom", None))

        # if QT_CONFIG(tooltip)
        self.resolution_CB.setToolTip(
            QCoreApplication.translate("MainWindow", u"Resolution of the images used for training ", None))
        # endif // QT_CONFIG(tooltip)
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Model name", None))
        # if QT_CONFIG(tooltip)
        self.model_name.setToolTip(QCoreApplication.translate("MainWindow", u"Model name", None))
        # endif // QT_CONFIG(tooltip)
        self.Save_FL_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Continue", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Training annotations", None))
        # if QT_CONFIG(tooltip)
        self.trianing_LE.setToolTip(QCoreApplication.translate("MainWindow", u"Path to training annotations", None))
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.trianing_LE.setStatusTip("")
        # endif // QT_CONFIG(statustip)
        # if QT_CONFIG(whatsthis)
        self.trianing_LE.setWhatsThis("")
        # endif // QT_CONFIG(whatsthis)
        self.trianing_LE.setInputMask("")
        self.trainin_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.custom_img_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.custom_test_img_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Testing annotations", None))
        # if QT_CONFIG(tooltip)
        self.testing_LE.setToolTip(QCoreApplication.translate("MainWindow", u"Path to testing annotations", None))
        # endif // QT_CONFIG(tooltip)
        self.testing_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.prerecorded_PB.setText(QCoreApplication.translate("MainWindow", u"Load prerecorded data", None))
        self.classify_PB.setText(QCoreApplication.translate("MainWindow", u"Classify images", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),
                                  QCoreApplication.translate("MainWindow", u"File Location", None))
        self.groupBox.setTitle(
            QCoreApplication.translate("MainWindow", u"Annotation class whitespace sesttings", None))
        self.wsoptions_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"Select", None))
        self.wsoptions_CB.setItemText(1, QCoreApplication.translate("MainWindow", u"Remove whitespace", None))
        self.wsoptions_CB.setItemText(2, QCoreApplication.translate("MainWindow", u"Keep only whitespace", None))
        self.wsoptions_CB.setItemText(3,
                                      QCoreApplication.translate("MainWindow", u"Keep tissue and whitespace", None))

        self.apply_PB.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.applyall_PB.setText(QCoreApplication.translate("MainWindow", u"Apply to all", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Add Whitespace to:", None))
        self.addws_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"Select", None))

        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Add Non-whitespace to:", None))
        self.addnonws_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"Select", None))

        self.changecolor_PB.setText(QCoreApplication.translate("MainWindow", u"Change Color", None))
        self.Reset_PB.setText(QCoreApplication.translate("MainWindow", u"Reset list", None))
        self.Combine_PB.setText(QCoreApplication.translate("MainWindow", u"Combine classes", None))
        self.delete_PB.setText(QCoreApplication.translate("MainWindow", u"Delete class", None))
        self.return_ts_PB.setText(QCoreApplication.translate("MainWindow", u"Return", None))
        self.save_ts_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Continue", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2),
                                  QCoreApplication.translate("MainWindow", u"Segmentation Settings", None))
        self.moveup_PB.setText(QCoreApplication.translate("MainWindow", u"Move up", None))
        self.Movedown_PB.setText(QCoreApplication.translate("MainWindow", u"Move Down", None))
        self.nesting_checkBox.setText(QCoreApplication.translate("MainWindow", u"Nest uncombined data", None))
        self.return_nesting_PB.setText(QCoreApplication.translate("MainWindow", u"Return", None))
        self.save_nesting_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Train", None))
        self.close_nesting_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Close", None))
        self.AS_PB.setText(QCoreApplication.translate("MainWindow", u"Go to Advanced Settings", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3),
                                  QCoreApplication.translate("MainWindow", u"Nesting", None))
        self.return_ad_PB.setText(QCoreApplication.translate("MainWindow", u"Return", None))
        self.save_ad_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Train", None))
        self.close_ad_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Close", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Training tile size:", None))
        self.tts_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"64", None))
        self.tts_CB.setItemText(1, QCoreApplication.translate("MainWindow", u"128", None))
        self.tts_CB.setItemText(2, QCoreApplication.translate("MainWindow", u"256", None))
        self.tts_CB.setItemText(3, QCoreApplication.translate("MainWindow", u"512", None))
        self.tts_CB.setItemText(4, QCoreApplication.translate("MainWindow", u"1024", None))
        self.tts_CB.setItemText(5, QCoreApplication.translate("MainWindow", u"2048", None))

        self.tts_CB.setCurrentText(QCoreApplication.translate("MainWindow", u"1024", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Training tiles number:", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Validation tiles number:", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Tissue mask evaluation:", None))
        self.label_69.setText(QCoreApplication.translate("MainWindow", u"Individual tissue mask evaluation:", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Training images", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Testing images", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Scaling factor", None))
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"Scale custom images", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"Use pre-scaled images", None))
        self.model_name_label.setText(QCoreApplication.translate("MainWindow", u"Model architecture", None))
        self.batch_label.setText(QCoreApplication.translate("MainWindow", u"Batch size", None))
        self.groupBox_3.setTitle(
            QCoreApplication.translate("MainWindow", u"Annotation class component analysis:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4),
                                  QCoreApplication.translate("MainWindow", u"Advanced Settings", None))
    # retranslateUi


class Ui_classify_im:
    """
    UI definition for the classification window.
    """
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
        self.overlay.setGeometry(QRect(245, 200, 350, 350))
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
    """
    Custom graphics view for displaying and manipulating images.
    """
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

class HoverPreviewLabel(QLabel):
    def __init__(self, icon_path, preview_image_path, parent=None):
        super().__init__(parent)

        # Set the icon (e.g., question mark)
        self.setPixmap(QPixmap(icon_path))
        self.setFixedSize(24, 24)  # optional: fix size for consistent appearance
        self.setScaledContents(True)

        # Preview image popup
        self.preview_popup = QLabel(parent)
        self.preview_popup.setPixmap(QPixmap(preview_image_path))
        self.preview_popup.adjustSize()
        self.preview_popup.setWindowFlags(Qt.ToolTip)  # behaves like a tooltip
        self.preview_popup.hide()

    def enterEvent(self, event):
        global_pos = self.mapToGlobal(self.rect().bottomRight())
        self.preview_popup.move(global_pos + QPoint(10, 10))
        self.preview_popup.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.preview_popup.hide()
        super().leaveEvent(event)