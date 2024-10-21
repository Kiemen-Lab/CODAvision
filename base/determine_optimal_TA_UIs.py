from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QWidget)

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
    def setupUi(self, MainWindow, CTA,CT0, CTC):
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
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow, CTA,CT0,CTC)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow, CTA,CT0, CTC):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.high_im.setText("")
        self.high_ta.setText(QCoreApplication.translate("MainWindow", u"Image A\nTA = "+ str(CTA), None))
        self.low_ta.setText(QCoreApplication.translate("MainWindow", u"Image C\nTA = "+ str(CTC), None))
        self.text.setText(QCoreApplication.translate("MainWindow", u"Which one of the images look good?", None))
        self.text_high.setText(QCoreApplication.translate("MainWindow", u"Image A", None))
        self.text_mid.setText(QCoreApplication.translate("MainWindow", u"Image B", None))
        self.text_low.setText(QCoreApplication.translate("MainWindow", u"Image C", None))
        self.medium_im.setText("")
        self.low_im.setText("")
        self.medium_ta.setText(QCoreApplication.translate("MainWindow", u"Image B\nTA = "+ str(CT0), None))
        self.raise_ta.setText(QCoreApplication.translate("MainWindow", u"None, keep more tissue", None))
        self.decrease_ta.setText(QCoreApplication.translate("MainWindow", u"No, keep more whitespace", None))