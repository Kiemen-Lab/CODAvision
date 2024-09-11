
# from PySide2 import QtWidgets, QtCore
# from CODA import Ui_MainWindow
# import os
# from datetime import datetime
# import xmltodict
# import pandas as pd
# from PySide2.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush
# from PySide2.QtWidgets import QColorDialog, QHeaderView, QTableWidgetItem
# from PySide2.QtCore import Qt

from PySide6 import QtWidgets, QtCore
from CODA import Ui_MainWindow
import os
from datetime import datetime
import xmltodict
import pandas as pd
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush
from PySide6.QtWidgets import QColorDialog, QHeaderView, QTableWidgetItem
from PySide6.QtCore import Qt


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, training_folder, parent=None):
        super().__init__(parent)
        self.training_folder = training_folder
        self.df = None
        self.combined_df = None
        self.add_ws_to = None
        self.add_nonws_to = None
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
        self.ui.apply_PB.clicked.connect(self.apply_whitespace_setting)
        self.ui.applyall_PB.clicked.connect(self.apply_all_whitespace_setting)
        self.ui.save_ts_PB.clicked.connect(self.save_and_continue_from_tab_2)
        self.ui.return_ts_PB.clicked.connect(lambda: self.return_to_previous_tab(return_to_first=True))
        self.ui.moveup_PB.clicked.connect(self.move_row_up)
        self.ui.Movedown_PB.clicked.connect(self.move_row_down)
        self.ui.return_nesting_PB.clicked.connect(lambda: self.return_to_previous_tab(return_to_first=False))
        self.ui.save_nesting_PB.clicked.connect(self.save_nesting_and_continue)
        self.ui.addcombo_PB.clicked.connect(self.add_combo)
        self.ui.removecombo_PB.clicked.connect(self.remove_combo)
        self.ui.save_ad_PB.clicked.connect(self.save_advanced_settings_and_close)
        self.ui.return_ad_PB.clicked.connect(self.return_to_previous_tab)
        self.ui.Combine_TW_1.verticalHeader().setVisible(False)
        self.ui.Combine_TW_2.verticalHeader().setVisible(False)
        self.combo_colors = {}

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

        if any(self.df['Whitespace Settings'].isna()):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Please assign a whitespace settings option to all annotation layers.')
            return

        self.add_ws_to = self.ui.addws_CB.currentIndex()
        self.add_nonws_to = self.ui.addnonws_CB.currentIndex()

        self.initialize_nesting_table()


        next_tab_index = self.ui.tabWidget.currentIndex() + 1
        if next_tab_index < self.ui.tabWidget.count():
            self.ui.tabWidget.setTabEnabled(next_tab_index, True)
        self.switch_to_next_tab()

    def initialize_nesting_table(self):
        model = QStandardItemModel()
        model.setColumnCount(1)
        model.setHorizontalHeaderLabels(["Layer Name"])

        for index, row in self.df.iterrows():
            item = QStandardItem(row['Layer Name'])
            item.setBackground(QColor(*row['Color']))
            item.setEditable(False)
            model.appendRow(item)

        self.ui.nesting_TW.setModel(model)
        self.ui.nesting_TW.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def move_row_up(self):
        model = self.ui.nesting_TW.model()
        current_index = self.ui.nesting_TW.currentIndex()
        if current_index.isValid() and current_index.row() > 0:
            current_item = model.takeItem(current_index.row())
            above_item = model.takeItem(current_index.row() - 1)

            model.setItem(current_index.row() - 1, current_item)
            model.setItem(current_index.row(), above_item)

            self.ui.nesting_TW.setCurrentIndex(model.index(current_index.row() - 1, 0))

    def move_row_down(self):
        model = self.ui.nesting_TW.model()
        current_index = self.ui.nesting_TW.currentIndex()
        if current_index.isValid() and current_index.row() < model.rowCount() - 1:
            current_item = model.takeItem(current_index.row())
            below_item = model.takeItem(current_index.row() + 1)

            model.setItem(current_index.row(), below_item)
            model.setItem(current_index.row() + 1, current_item)

            self.ui.nesting_TW.setCurrentIndex(model.index(current_index.row() + 1, 0))

    def return_to_previous_tab(self, return_to_first=False):
        current_index = self.ui.tabWidget.currentIndex()

        if return_to_first:
            target_index = 0
        else:
            target_index = current_index - 1

        # Enable the target tab
        self.ui.tabWidget.setTabEnabled(target_index, True)
        self.ui.tabWidget.setCurrentIndex(target_index)

        # Disable all tabs after the target tab
        for i in range(target_index + 1, self.ui.tabWidget.count()):
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

    def save_nesting_and_continue(self):
        model = self.ui.nesting_TW.model()
        nesting_order = [model.item(row).text() for row in range(model.rowCount())]

        # Create a mapping of layer names to their original indices
        original_indices = {name: index for index, name in enumerate(self.df['Layer Name'])}

        # Create the Nesting column
        self.df['Nesting'] = [original_indices[name] + 1 for name in nesting_order]

        print("Updated DataFrame:")
        print(self.df)

        current_tab_index = self.ui.tabWidget.currentIndex()
        next_tab_index = current_tab_index + 1

        if next_tab_index < self.ui.tabWidget.count():
            self.ui.tabWidget.setTabEnabled(current_tab_index, False)  # Disable current tab
            self.ui.tabWidget.setTabEnabled(next_tab_index, True)  # Enable next tab
            self.ui.tabWidget.setCurrentIndex(next_tab_index)  # Switch to next tab

        self.initialize_advanced_settings()

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
        if dialog.exec() == QtWidgets.QDialog.Accepted:
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



    def initialize_advanced_settings(self):
        layer_names = self.df['Layer Name'].tolist()

        # Configure component_TW
        self.ui.component_TW.setColumnCount(1)
        self.ui.component_TW.setHorizontalHeaderLabels(["Annotation layers"])
        self.ui.component_TW.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) #data matches the table width

        for row, layer_name in enumerate(layer_names):
            self.ui.component_TW.insertRow(row)
            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.ui.component_TW.setItem(row, 0, item)

        # Connect itemChanged signal to slot (value gets gray after being checked)
        self.ui.component_TW.itemChanged.connect(self.on_delete_item_changed)


        # Configure delete_TW
        self.ui.delete_TW.setColumnCount(1)
        self.ui.delete_TW.setHorizontalHeaderLabels(["Annotation layers"])
        self.ui.delete_TW.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) #data matches the table width


        for row, layer_name in enumerate(layer_names):
            self.ui.delete_TW.insertRow(row)
            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.ui.delete_TW.setItem(row, 0, item)

        # Connect itemChanged signal to the slot
        self.ui.delete_TW.itemChanged.connect(self.on_delete_item_changed)


        # Configure Combine_TW_1
        self.ui.Combine_TW_1.setColumnCount(1)
        self.ui.Combine_TW_1.setHorizontalHeaderLabels(["Layer Name"])
        for name in layer_names:
            row_position = self.ui.Combine_TW_1.rowCount()
            self.ui.Combine_TW_1.insertRow(row_position)
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make item non-editable
            self.ui.Combine_TW_1.setItem(row_position, 0, item)

        # Configure Combine_TW_2
        self.ui.Combine_TW_2.setColumnCount(0)

        # Hide vertical headers and add horizontal header line for both tables
        for table in [self.ui.Combine_TW_1, self.ui.Combine_TW_2]:
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setVisible(True)
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            table.setShowGrid(True)
            table.setStyleSheet("QHeaderView::section { background-color: #f0f0f0; }")

    def on_delete_item_changed(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor(200, 200, 200))  # Light gray
        else:
            item.setBackground(QColor(255, 255, 255))  # White

    def add_combo(self):
        selected_items = self.ui.Combine_TW_1.selectedItems()
        if selected_items:
            while True:
                combo_name, ok = QtWidgets.QInputDialog.getText(self, "Combo Name",
                                                                "Please type combo name (letters only):")
                if not ok:
                    return
                if combo_name and combo_name.isalpha():
                    break
                QtWidgets.QMessageBox.warning(self, 'Invalid Name', 'Please use only letters for the combo name.')

            #show info about color selection
            QtWidgets.QMessageBox.information(self, 'Select Color',
                                              'Please select the color for the created combination of annotation classes.')

            # Display the color picker dialog
            color_dialog = QColorDialog(self)
            color_dialog.setWindowTitle("Select Combo Color")

            if color_dialog.exec_():
                selected_color = color_dialog.currentColor()
                rgb_color = (selected_color.red(), selected_color.green(), selected_color.blue())

                # Save the combo name and RGB color
                self.combo_colors[combo_name] = rgb_color

                new_column = self.ui.Combine_TW_2.columnCount()
                self.ui.Combine_TW_2.insertColumn(new_column)
                self.ui.Combine_TW_2.setHorizontalHeaderItem(new_column, QtWidgets.QTableWidgetItem(combo_name))

                for i, item in enumerate(selected_items):
                    if self.ui.Combine_TW_2.rowCount() <= i:
                        self.ui.Combine_TW_2.insertRow(i)
                    new_item = QtWidgets.QTableWidgetItem(item.text())
                    new_item.setFlags(new_item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make item non-editable
                    self.ui.Combine_TW_2.setItem(i, new_column, new_item)

                    # Find and remove the item from Combine_TW_1
                    for row in range(self.ui.Combine_TW_1.rowCount()):
                        if self.ui.Combine_TW_1.item(row, 0).text() == item.text():
                            self.ui.Combine_TW_1.removeRow(row)
                            break

                # Ensure the horizontal header is visible and styled
            self.ui.Combine_TW_2.horizontalHeader().setVisible(True)
            self.ui.Combine_TW_2.setStyleSheet("QHeaderView::section { background-color: #f0f0f0; }")

    def remove_combo(self):
        selected_columns = set(index.column() for index in self.ui.Combine_TW_2.selectedIndexes())
        for column in sorted(selected_columns, reverse=True):
            for row in range(self.ui.Combine_TW_2.rowCount()):
                item = self.ui.Combine_TW_2.item(row, column)
                if item:
                    self.ui.Combine_TW_1.insertRow(self.ui.Combine_TW_1.rowCount())
                    self.ui.Combine_TW_1.setItem(self.ui.Combine_TW_1.rowCount() - 1, 0,
                                                 QtWidgets.QTableWidgetItem(item.text()))
            self.ui.Combine_TW_2.removeColumn(column)


    # Add or update these methods in the MainWindow class:
    def save_advanced_settings_and_close(self):
        # Delete layers
        delete_layers = {}
        for row in range(self.ui.delete_TW.rowCount()):
            item = self.ui.delete_TW.item(row, 0)
            if item:
                layer_name = item.text()
                is_checked = item.checkState() == Qt.Checked
                delete_layers[layer_name] = not is_checked  # Save True for non-checked items

        self.df['Delete layer'] = self.df['Layer Name'].map(delete_layers)

        # Component analysis
        component_layers = {}
        for row in range(self.ui.component_TW.rowCount()):
            item = self.ui.component_TW.item(row, 0)
            if item:
                layer_name = item.text()
                is_checked = item.checkState() == Qt.Checked
                component_layers[layer_name] = is_checked  # Save True for checked items

        self.df['Component analysis'] = self.df['Layer Name'].map(component_layers)

        # Combined layers
        combo_updates = {}
        original_indices = {name: index for index, name in enumerate(self.df['Layer Name'])}

        for col in range(self.ui.Combine_TW_2.columnCount()):
            combo_name = self.ui.Combine_TW_2.horizontalHeaderItem(col).text()
            layers = [self.ui.Combine_TW_2.item(row, col).text() for row in range(self.ui.Combine_TW_2.rowCount())
                      if self.ui.Combine_TW_2.item(row, col)]
            if layers:
                combo_indexes = [original_indices[layer] + 1 for layer in layers]  # Add 1 to make it 1-indexed
                combo_updates[combo_name] = combo_indexes

        # Function to generate the combined layers based on the updates
        def combine_layers(layers_indexes, layer_names, combo_updates):
            new_indexes = layers_indexes.copy()
            new_names = layer_names.copy()

            sorted_updates = sorted(combo_updates.items(), key=lambda x: max(x[1]), reverse=True)

            for combo_name, indexes in sorted_updates:
                min_index = min(indexes)
                max_index = max(indexes)

                for i in range(len(new_indexes)):
                    if new_indexes[i] == max_index:
                        new_indexes[i] = min_index
                    elif new_indexes[i] > max_index:
                        new_indexes[i] -= 1

                new_names[min_index - 1] = combo_name
                new_names.pop(max_index - 1)
                # new_names.insert(max_index - 2, new_names.pop(min_index - 1))

            return new_indexes, new_names

        # Get the original layer indexes and names
        original_indexes = list(range(1, len(self.df) + 1))
        original_names = self.df['Layer Name'].tolist()

        combined_idxes, combined_names = combine_layers(original_indexes, original_names, combo_updates)

        # Create the Combined layers column
        self.df['Combined layers'] = combined_idxes

        # Create new dataframe for combined layers info
        self.combined_df = pd.DataFrame({
            'Combined names': combined_names,
        })

        # Initialize Combined colors with colors from self.combo_colors
        self.combined_df['Combined colors'] = [self.combo_colors.get(name, '') for name in combined_names]

        # Map colors from self.df to combined_df based on 'Combined names' if self.combo_colors does not have a color
        self.combined_df['Combined colors'] = self.combined_df.apply(
            lambda row: self.df.set_index('Layer Name')['Color'].get(row['Combined names'], row['Combined colors']),
            axis=1
        )

        self.tile_size = int(self.ui.tts_CB.currentText())
        self.ntrain = self.ui.ttn_SB.value()
        self.nval = self.ui.vtn_SB.value()
        self.TA = self.ui.TA_SB.value()

        df = self.df
        combined_df = self.combined_df
        print("Original DataFrame:")
        print(df)
        print("\nCombined Layers DataFrame:")
        print(combined_df)

        self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    # Load and apply the dark theme stylesheet
    with open('dark_theme.qss', 'r') as file:
        app.setStyleSheet(file.read())

    window = MainWindow()
    window.show()
    app.exec()
