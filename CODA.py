# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'CODA.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(995, 431)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_3 = QGridLayout(self.tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
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


        self.gridLayout_3.addLayout(self.horizontalLayout, 0, 0, 1, 4)

        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_3.addWidget(self.label_3, 2, 0, 1, 1)

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


        self.gridLayout_3.addLayout(self.horizontalLayout_2, 1, 0, 1, 4)

        self.verticalSpacer = QSpacerItem(20, 187, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.verticalSpacer, 4, 3, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(419, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_3, 2, 2, 1, 1)

        self.model_name = QLineEdit(self.tab)
        self.model_name.setObjectName(u"model_name")

        self.gridLayout_3.addWidget(self.model_name, 2, 1, 1, 1)

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
        self.resolution_CB.setObjectName(u"resolution_CB")

        self.horizontalLayout_3.addWidget(self.resolution_CB)


        self.gridLayout_3.addLayout(self.horizontalLayout_3, 3, 0, 1, 2)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer = QSpacerItem(348, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer)

        self.Save_FL_PB = QPushButton(self.tab)
        self.Save_FL_PB.setObjectName(u"Save_FL_PB")

        self.horizontalLayout_9.addWidget(self.Save_FL_PB)


        self.gridLayout_3.addLayout(self.horizontalLayout_9, 5, 0, 1, 4)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_2 = QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.tableWidget = QTableWidget(self.tab_2)
        self.tableWidget.setObjectName(u"tableWidget")

        self.gridLayout_2.addWidget(self.tableWidget, 0, 0, 1, 2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_5 = QLabel(self.tab_2)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.pushButton_4 = QPushButton(self.tab_2)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.horizontalLayout_4.addWidget(self.pushButton_4)

        self.pushButton_5 = QPushButton(self.tab_2)
        self.pushButton_5.setObjectName(u"pushButton_5")

        self.horizontalLayout_4.addWidget(self.pushButton_5)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.comboBox_4 = QComboBox(self.tab_2)
        self.comboBox_4.setObjectName(u"comboBox_4")

        self.verticalLayout.addWidget(self.comboBox_4)


        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_6 = QLabel(self.tab_2)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_6.addWidget(self.label_6)

        self.comboBox_2 = QComboBox(self.tab_2)
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.horizontalLayout_6.addWidget(self.comboBox_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_7 = QLabel(self.tab_2)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_7.addWidget(self.label_7)

        self.comboBox_3 = QComboBox(self.tab_2)
        self.comboBox_3.setObjectName(u"comboBox_3")

        self.horizontalLayout_7.addWidget(self.comboBox_3)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 1, 1, 1, 1)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalSpacer_2 = QSpacerItem(438, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_2)

        self.pushButton_8 = QPushButton(self.tab_2)
        self.pushButton_8.setObjectName(u"pushButton_8")

        self.horizontalLayout_8.addWidget(self.pushButton_8)

        self.pushButton_6 = QPushButton(self.tab_2)
        self.pushButton_6.setObjectName(u"pushButton_6")

        self.horizontalLayout_8.addWidget(self.pushButton_6)


        self.gridLayout_2.addLayout(self.horizontalLayout_8, 2, 0, 1, 2)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.tableView = QTableView(self.tab_3)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setGeometry(QRect(10, 10, 256, 321))
        self.layoutWidget = QWidget(self.tab_3)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(272, 10, 79, 104))
        self.verticalLayout_4 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.pushButton_7 = QPushButton(self.layoutWidget)
        self.pushButton_7.setObjectName(u"pushButton_7")

        self.verticalLayout_3.addWidget(self.pushButton_7)

        self.pushButton_9 = QPushButton(self.layoutWidget)
        self.pushButton_9.setObjectName(u"pushButton_9")

        self.verticalLayout_3.addWidget(self.pushButton_9)


        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.tabWidget.addTab(self.tab_3, "")

        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(whatsthis)
        self.tabWidget.setWhatsThis(QCoreApplication.translate("MainWindow", u"fsdfsfsdfsf", None))
#endif // QT_CONFIG(whatsthis)
        self.label.setText(QCoreApplication.translate("MainWindow", u"Training annotations", None))
#if QT_CONFIG(tooltip)
        self.trianing_LE.setToolTip(QCoreApplication.translate("MainWindow", u"Path to training annotations", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.trianing_LE.setStatusTip("")
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        self.trianing_LE.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.trianing_LE.setInputMask("")
        self.trainin_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Model name", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Testing annotations", None))
#if QT_CONFIG(tooltip)
        self.testing_LE.setToolTip(QCoreApplication.translate("MainWindow", u"Path to testing annotations", None))
#endif // QT_CONFIG(tooltip)
        self.testing_PB.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
#if QT_CONFIG(tooltip)
        self.model_name.setToolTip(QCoreApplication.translate("MainWindow", u"Model name", None))
#endif // QT_CONFIG(tooltip)
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Resolution", None))
        self.resolution_CB.setItemText(0, QCoreApplication.translate("MainWindow", u"Select", None))
        self.resolution_CB.setItemText(1, QCoreApplication.translate("MainWindow", u"1x", None))
        self.resolution_CB.setItemText(2, QCoreApplication.translate("MainWindow", u"5x", None))
        self.resolution_CB.setItemText(3, QCoreApplication.translate("MainWindow", u"10x", None))

#if QT_CONFIG(tooltip)
        self.resolution_CB.setToolTip(QCoreApplication.translate("MainWindow", u"Resolution of the images used for training ", None))
#endif // QT_CONFIG(tooltip)
        self.Save_FL_PB.setText(QCoreApplication.translate("MainWindow", u"Save and Continue", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"File Location", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Tissue/Whitespace removal options:", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Apply to all", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Add Whitespace to:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Add Non-whitespace to:", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"Return", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"Save and Continue", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Tissue Segmentation", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"Move up", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"Move Down", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Nesting", None))
    # retranslateUi

