from PySide2 import QtWidgets, QtCore
from CODA import Ui_MainWindow
import os
from datetime import datetime
import xmltodict
import pandas as pd


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, training_folder, parent=None):
        super().__init__(parent)
        self.training_folder = training_folder
        self.setWindowTitle("Load Data")
        self.layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Would you like to load data from an .xml file or use prerecorded data?")
        self.layout.addWidget(self.label)

        self.xml_button = QtWidgets.QPushButton("Load from .xml")
        self.prerecorded_button = QtWidgets.QPushButton("Use prerecorded data")

        self.layout.addWidget(self.xml_button)
        self.layout.addWidget(self.prerecorded_button)

        self.setLayout(self.layout)

        self.xml_button.clicked.connect(self.load_xml)
        self.prerecorded_button.clicked.connect(self.use_prerecorded_data)

    def load_xml(self):
        xml_file = None
        for file in os.listdir(self.training_folder):
            if file.endswith('.xml'):
                xml_file = os.path.join(self.training_folder, file)
                break
        if xml_file:
            try:
                xml_df = self.parse_xml_to_dataframe(xml_file)
                print(f"Loaded XML file: {xml_file}")
                print(xml_df)
                self.accept()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to parse XML file: {str(e)}')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No XML file found in the training annotations folder.')

    def use_prerecorded_data(self):
        print("Using prerecorded data")
        self.accept()

    def parse_xml_to_dataframe(self, xml_file):
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        xml_dict = xmltodict.parse(xml_content)

        # Adjust the extraction logic based on the actual structure of your XML file
        annotations = xml_dict.get("Annotations", {}).get("Annotation", [])
        data = []
        for layer in annotations:
            layer_name = layer.get('@Name')
            color = layer.get('@LineColor')
            rgb = self.int_to_rgb(color)
            data.append({'Layer Name': layer_name, 'Color': rgb})

        df = pd.DataFrame(data)
        return df

    def int_to_rgb(self, hex_color):
        hex_color = int(hex_color)
        b = (hex_color // 65536) % 256
        g = (hex_color // 256) % 256
        r = hex_color % 256

        return r, g, b


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Save_FL_PB.clicked.connect(self.fill_form_and_continue)
        self.ui.trainin_PB.clicked.connect(lambda: self.select_imagedir('training'))
        self.ui.testing_PB.clicked.connect(lambda: self.select_imagedir('testing'))
        self.set_initial_model_name()

    def set_initial_model_name(self):
        """Set the initial text of the model_name text box to today's date."""
        today = datetime.now()
        date_string = today.strftime("%m_%d_%Y")
        self.ui.model_name.setText(date_string)

    def select_imagedir(self, purpose):
        """Open a dialog to select an image directory for training or testing."""
        dialog_title = f'Select {purpose.capitalize()} Image Directory'
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, dialog_title, os.getcwd())
        if folder_path:
            if purpose == 'training':
                self.ui.trianing_LE.setText(folder_path)
            elif purpose == 'testing':
                self.ui.testing_LE.setText(folder_path)

    def fill_form_and_continue(self):
        """Fill the form, process data, and switch to the next tab if successful."""
        if self.fill_form():
            self.switch_to_next_tab()
            self.show_custom_dialog()

    def switch_to_next_tab(self):
        current_index = self.ui.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.ui.tabWidget.count()
        self.ui.tabWidget.setCurrentIndex(next_index)

        # Force update
        QtCore.QCoreApplication.processEvents()

    def fill_form(self):
        """Process data"""
        pth = self.ui.trianing_LE.text()
        pthtest = self.ui.testing_LE.text()
        model_name = self.ui.model_name.text().replace(' ','_')
        resolution = self.ui.resolution_CB.currentText()

        if not pth:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please enter training annotations path')
            return False

        if not pthtest:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please enter testing annotations path')
            return False

        # Check for .ndpi or .svs files in training path
        if not any(f.endswith(('.ndpi', '.svs')) for f in os.listdir(pth)):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The selected training path does not contain .ndpi or .svs files')
            return False

        # Check for .ndpi or .svs files in testing path
        if not any(f.endswith(('.ndpi', '.svs')) for f in os.listdir(pthtest)):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The selected testing path does not contain .ndpi or .svs files')
            return False

        # Check if resolution is selected
        if resolution == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please chose a resolution from the drop down box')
            return False

        # Check if resolution is selected
        if pth == pthtest:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'The folder selected for the testing annotations must be different from the training annotations folder, please select a different folder.')
            return False

        print(f"Form filled with: \nTraining path: {pth}\nTesting path: {pthtest}\nModel name: {model_name}\nResolution: {resolution}")
        return True

    def show_custom_dialog(self):
        training_folder = self.ui.trianing_LE.text()
        dialog = CustomDialog(training_folder, self)
        dialog.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
