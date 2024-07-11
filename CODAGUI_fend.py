from PySide2 import QtWidgets, QtCore
from CODA import Ui_MainWindow
import os
from datetime import datetime
import xmltodict
import pandas as pd
from PySide2.QtGui import QColor
from PySide2.QtWidgets import QColorDialog


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, training_folder, parent=None):
        super().__init__(parent)
        self.training_folder = training_folder
        self.df = None
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
                self.df = self.parse_xml_to_dataframe(xml_file)
                print(f"Loaded XML file: {xml_file}")
                print(self.df)
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

        annotations = xml_dict.get("Annotations", {}).get("Annotation", [])
        data = []
        for layer in annotations:
            layer_name = layer.get('@Name')
            color = layer.get('@LineColor')
            rgb = self.int_to_rgb(color)
            data.append(
                {'Layer Name': layer_name, 'Color': rgb, 'Whitespace Settings': None})  # Add whitespace settings

        df = pd.DataFrame(data)
        return df

    def int_to_rgb(self, hex_color):
        hex_color = int(hex_color)
        b = (hex_color // 65536) % 256
        g = (hex_color // 256) % 256
        r = hex_color % 256

        return r, g, b

    def get_dataframe(self):
        return self.df if hasattr(self, 'df') else None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Save_FL_PB.clicked.connect(self.fill_form_and_continue)
        self.ui.trainin_PB.clicked.connect(lambda: self.select_imagedir('training'))
        self.ui.testing_PB.clicked.connect(lambda: self.select_imagedir('testing'))
        self.ui.changecolor_PB.clicked.connect(self.change_color)
        self.ui.apply_PB.clicked.connect(self.apply_whitespace_setting)  # Connect apply button
        self.ui.applyall_PB.clicked.connect(self.apply_all_whitespace_setting)  # Connect apply all button
        self.ui.save_ts_PB.clicked.connect(self.save_and_continue_from_tab_2)  # Connect Save and Continue button
        self.ui.return_ts_PB.clicked.connect(self.return_to_first_tab)

        self.set_initial_model_name()
        self.ui.tabWidget.setCurrentIndex(0)  # Initialize the first tab
        for i in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, False)

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
            next_tab_index = self.ui.tabWidget.currentIndex() + 1
            if next_tab_index < self.ui.tabWidget.count():
                self.ui.tabWidget.setTabEnabled(next_tab_index, True)
            self.switch_to_next_tab()
            self.show_custom_dialog()

    def save_and_continue_from_tab_2(self):

        if self.ui.addws_CB.currentText() == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a valid option from the "Add Whitespace '
                                                           'to:" dropdown box.')
            return

        if self.ui.addnonws_CB.currentText() == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a valid option from the "Add Non-whitespace '
                                                           'to:" dropdown box.')
            return

        add_ws_to = self.ui.addws_CB.currentIndex()
        add_nonws_to = self.ui.addnonws_CB.currentIndex()

        #TINA _________________________delete all these prints when final version_________________
        print(f'Add whitespace to: {add_ws_to}')
        print(f'Add whitespace to: {add_nonws_to}')
        print('Dataframe:')
        print(self.df)
        #TINA _________________________delete all these prints when final version_________________


        next_tab_index = self.ui.tabWidget.currentIndex() + 1
        if next_tab_index < self.ui.tabWidget.count():
            self.ui.tabWidget.setTabEnabled(next_tab_index, True)
        self.switch_to_next_tab()

    def return_to_first_tab(self):
        # Enable only the first tab
        self.ui.tabWidget.setTabEnabled(0, True)
        self.ui.tabWidget.setCurrentIndex(0)
        for i in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, False)

        QtCore.QCoreApplication.processEvents()

    def switch_to_next_tab(self):
        current_index = self.ui.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.ui.tabWidget.count()

        # Disable all tabs except the current one
        for i in range(self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, False)
        self.ui.tabWidget.setTabEnabled(next_index, True)

        self.ui.tabWidget.setCurrentIndex(next_index)

        # Force update
        QtCore.QCoreApplication.processEvents()

    def fill_form(self):
        """Process data"""
        pth = self.ui.trianing_LE.text()
        pthtest = self.ui.testing_LE.text()
        model_name = self.ui.model_name.text().replace(' ', '_')
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
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The folder selected for the testing annotations must be different from the '
                                          'training annotations folder, please select a different folder.')
            return False

        print(
            f"Form filled with: \nTraining path: {pth}\nTesting path: {pthtest}\nModel name: {model_name}\nResolution: {resolution}")
        return True

    def show_custom_dialog(self):
        training_folder = self.ui.trianing_LE.text()
        dialog = CustomDialog(training_folder, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.df = dialog.get_dataframe()
            if self.df is not None:
                self.populate_table_widget()

    def populate_table_widget(self):
        if self.df is None:
            return

        table = self.ui.tissue_segmentation_TW
        table.setRowCount(len(self.df))
        table.setColumnCount(2)  # Adjust column count
        table.setHorizontalHeaderLabels(["Annotation Class", "Whitespace Settings"])

        for row, (index, data) in enumerate(self.df.iterrows()):
            layer_name = data['Layer Name']
            color = data['Color']

            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setBackground(QColor(*color))
            table.setItem(row, 0, item)

            ws_item = QtWidgets.QTableWidgetItem("")
            ws_item.setBackground(QColor(255, 255, 255))
            table.setItem(row, 1, ws_item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Populate combo boxes with layer names
        self.populate_combo_boxes()

    def populate_combo_boxes(self):
        layer_names = self.df['Layer Name'].tolist()
        self.ui.addws_CB.addItems(layer_names)
        self.ui.addnonws_CB.addItems(layer_names)

    def apply_whitespace_setting(self):
        ws_option = self.ui.wsoptions_CB.currentText()
        ws_map = {
            'Remove whitespace': 0,
            'Keep only whitespace': 1,
            'Keep tissue and whitespace': 2
        }

        if ws_option == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a whitespace setting')
            return

        table = self.ui.tissue_segmentation_TW
        selected_items = table.selectedItems()

        if not selected_items:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select an annotation class from the table.')
            return

        selected_row = selected_items[0].row()
        ws_value = ws_map[ws_option]

        self.df.at[selected_row, 'Whitespace Settings'] = ws_value

        ws_item = table.item(selected_row, 1)
        if ws_item:
            ws_item.setText(ws_option)

    def apply_all_whitespace_setting(self):
        ws_option = self.ui.wsoptions_CB.currentText()
        ws_map = {
            'Remove whitespace': 0,
            'Keep only whitespace': 1,
            'Keep tissue and whitespace': 2
        }

        if ws_option == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a whitespace option')
            return

        ws_value = ws_map[ws_option]

        table = self.ui.tissue_segmentation_TW

        for row in range(table.rowCount()):
            self.df.at[row, 'Whitespace Settings'] = ws_value

            ws_item = table.item(row, 1)
            if ws_item:
                ws_item.setText(ws_option)

    # Function to change the color of the selected row in the annotation class table and update the dataframe
    def change_color(self):
        table = self.ui.tissue_segmentation_TW
        selected_items = table.selectedItems()

        if not selected_items:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select an annotation class from the table.')
            return

        selected_row = selected_items[0].row()

        current_color = self.df.iloc[selected_row]['Color']
        initial_color = QColor(*current_color)

        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(initial_color)

        if color_dialog.exec_():
            new_color = color_dialog.currentColor()
            new_rgb = (new_color.red(), new_color.green(), new_color.blue())

            # Update the DataFrame
            self.df.at[selected_row, 'Color'] = new_rgb

            # Update only the Annotation Class column in the table
            item = table.item(selected_row, 0)
            if item:
                item.setBackground(new_color)

            layer_name = self.df.iloc[selected_row]['Layer Name']
            print(f"Color changed for {layer_name} to {new_rgb}")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
