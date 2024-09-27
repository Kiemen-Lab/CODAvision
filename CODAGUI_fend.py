
from PySide6 import QtWidgets, QtCore
from CODA import Ui_MainWindow
import os
from datetime import datetime
import xmltodict
import pandas as pd
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush
from PySide6.QtWidgets import QColorDialog, QHeaderView
from PySide6.QtCore import Qt


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, training_folder, parent=None):
        super().__init__(parent)
        self.original_df = None
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
                self.original_df = self.df.copy()  # Initialize original_df after loading data
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
        self.original_df = df.copy()  # Save the original dataframe for resetting
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
        super().__init__()  # Use super() to initialize the parent class


        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # Pass the MainWindow instance itself as the parent
        self.setCentralWidget(self.ui.centralwidget)  # Set the central widget
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
        self.ui.Combine_PB.clicked.connect(self.add_combo)
        self.ui.Reset_PB.clicked.connect(self.reset_combo)
        self.ui.save_ad_PB.clicked.connect(self.save_advanced_settings_and_close)
        self.ui.return_ad_PB.clicked.connect(self.return_to_previous_tab)
        self.ui.delete_PB.clicked.connect(self.delete_annotation_class)
        self.ui.nesting_checkBox.stateChanged.connect(self.on_nesting_checkbox_state_changed)
        self.combo_colors = {}
        self.original_df = None  # Initialize original_df
        self.df = None
        self.combined_df = None  # Initialize combined_df


        self.set_initial_model_name()
        self.ui.tabWidget.setCurrentIndex(0)  # Initialize the first tab
        for i in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, False)

        self.setWindowTitle("ANACODA")

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

    def reset_combo(self):
        if self.original_df is not None:
            self.df = self.original_df.copy()  # Reset df to the original data
            self.combined_df = None  # Reset combined_df
            self.populate_table_widget()
            # Reset original_indices
            self.original_indices = {name: index + 1 for index, name in enumerate(self.original_df['Layer Name'])}
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Original data not loaded.')

    def save_and_continue_from_tab_2(self):
        if self.ui.addws_CB.currentText() == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Please select a valid option from the "Add Whitespace to:" dropdown box.')
            return

        if self.ui.addnonws_CB.currentText() == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Please select a valid option from the "Add Non-whitespace to:" dropdown box.')
            return

        if self.combined_df is not None:
            if any(self.combined_df['Whitespace Settings'].isna()):
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'Please assign a whitespace settings option to all annotation layers.')
                return
        else:
            if any(self.df['Whitespace Settings'].isna()):
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'Please assign a whitespace settings option to all annotation layers.')
                return

        self.add_ws_to = self.ui.addws_CB.currentIndex()
        self.add_nonws_to = self.ui.addnonws_CB.currentIndex()

        # Create a mapping of original indices to layer names
        original_indices = {index + 1: name for index, name in enumerate(self.original_df['Layer Name'])}

        # Initialize combined_df if it is None
        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1

        # Ensure 'Layer idx' column exists
        if 'Layer idx' not in self.combined_df.columns:
            self.combined_df['Layer idx'] = self.combined_df.index + 1

        # Update self.df based on the whitespace settings in self.combined_df
        for idx, row in self.combined_df.iterrows():
            layer_indices = row['Layer idx']
            whitespace_setting = row['Whitespace Settings']

            if isinstance(layer_indices, list):  # when you combine layers you get a list of values to update
                for original_idx in layer_indices:
                    layer_name = original_indices[original_idx]
                    df_idx = self.df[self.df['Layer Name'] == layer_name].index
                    if not df_idx.empty:
                        self.df.at[df_idx[0], 'Whitespace Settings'] = whitespace_setting
            else:
                layer_name = original_indices[layer_indices]
                df_idx = self.df[self.df['Layer Name'] == layer_name].index
                if not df_idx.empty:
                    self.df.at[df_idx[0], 'Whitespace Settings'] = whitespace_setting


        # Combine layers in main dataframe
        if self.combined_df is not None:
            combined_layers = [None] * len(self.df)
            for idx, row in self.combined_df.iterrows():
                if isinstance(row['Layer idx'], list):
                    for original_idx in row['Layer idx']:
                        if 0 <= original_idx - 1 < len(combined_layers):
                            combined_layers[original_idx - 1] = idx + 1
                else:
                    if 0 <= row['Layer idx'] - 1 < len(combined_layers):
                        combined_layers[row['Layer idx'] - 1] = idx + 1
            self.df['Combined layers'] = combined_layers
            self.df['Combined layers'] = self.df['Combined layers'].apply(lambda x: int(x) if x is not None else x)
        else:
            self.df['Combined layers'] = (self.df.index + 1).astype(int)

        print("Updated Raw DataFrame from tab 2:")
        print(self.df)

        self.initialize_nesting_table()

        next_tab_index = self.ui.tabWidget.currentIndex() + 1
        if next_tab_index < self.ui.tabWidget.count():
            self.ui.tabWidget.setTabEnabled(next_tab_index, True)
        self.switch_to_next_tab()

    def initialize_nesting_table(self):
        model = QStandardItemModel()
        model.setColumnCount(1)
        model.setHorizontalHeaderLabels(["Layer Name"])

        # Use combined classes or uncombined classes depending on the checkbox state
        if self.ui.nesting_checkBox.isChecked():
            source_df = self.df
        else:
            source_df = self.combined_df
            source_df = source_df[source_df['Deleted'] != True]

        for index, row in source_df.iterrows():
            item = QStandardItem(row['Layer Name'])
            item.setBackground(QColor(*row['Color']))
            item.setEditable(False)

            # Convert the background color to greyscale
            greyscale = 0.299 * row['Color'][0] + 0.587 * row['Color'][1] + 0.114 * row['Color'][2]
            if greyscale > 128:  # If greyscale is above 50% grey, set text color to black
                item.setForeground(QBrush(Qt.black))
            else:  # Otherwise, set text color to white
                item.setForeground(QBrush(Qt.white))

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

    def on_nesting_checkbox_state_changed(self, state):
        self.initialize_nesting_table()

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
        reverse_original_indices = {index + 1: name for index, name in enumerate(self.df['Layer Name'])}

        # Create the Nesting column in self.df
        if self.ui.nesting_checkBox.isChecked():
            # Update the Nesting column for uncombined classes
            self.df['Nesting'] = [original_indices[name] + 1 for name in nesting_order]
        else:
            # Update the Nesting column for combined classes
            nesting_order_combined = []
            for name in nesting_order:
                layer_idx = self.combined_df.loc[self.combined_df['Layer Name'] == name, 'Layer idx'].values[0]
                if isinstance(layer_idx, list):
                    nesting_order_combined.extend(layer_idx)
                else:
                    nesting_order_combined.append(layer_idx)

            # Add deleted layers to the end of the nesting list
            deleted_layers = self.df[self.df['Delete layer'] == True].index + 1
            nesting_order_combined.extend(deleted_layers)

            # Get the names for the combined nesting order, handling missing keys
            nesting_order_combined_names = [reverse_original_indices[i] for i in nesting_order_combined if
                                            i in reverse_original_indices]

            # Update the Nesting column for combined classes
            self.df['Nesting'] = [original_indices[name] + 1 for name in nesting_order_combined_names]

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
                self.original_df = self.df.copy()  # Set original_df in MainWindow
                self.populate_table_widget()
            if self.df is not None:
                self.populate_table_widget()

    def populate_table_widget(self, df=None):
        if df is None:
            df = self.df  # Populate the table with the original dataframe if no dataframe is passed

        if df is None:
            return

        table = self.ui.tissue_segmentation_TW
        # table.setRowCount(len(df))
        table.setRowCount(0) # Clear the table before populating
        table.setColumnCount(2)  # Adjust column count
        table.setHorizontalHeaderLabels(["Annotation Class", "Whitespace Settings"])

        ws_map = {
            0: 'Remove whitespace',
            1: 'Keep only whitespace',
            2: 'Keep tissue and whitespace'
        }

        for index, data in df.iterrows():
            if data.get('Deleted', False):
                print(f"Skipping row {data['Layer Name']} marked as deleted")
                continue  # Skip rows marked as deleted

            row = table.rowCount()
            table.insertRow(row)

            layer_name = data['Layer Name']
            color = data['Color']
            whitespace_setting = data['Whitespace Settings']

            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setBackground(QColor(*color))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make the item read-only

            # Convert the background color to greyscale
            greyscale = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if greyscale > 128:  # If greyscale is above 50% grey, set text color to black
                item.setForeground(QBrush(QColor(0, 0, 0)))
            else:  # Otherwise, set text color to white
                item.setForeground(QBrush(QColor(255, 255, 255)))

            table.setItem(row, 0, item)

            ws_text = ws_map.get(whitespace_setting, "")
            ws_item = QtWidgets.QTableWidgetItem(ws_text)
            ws_item.setBackground(QColor(0, 0, 0))  # Set background color to black
            ws_item.setForeground(QBrush(QColor(255, 255, 255)))  # Set text color to white
            ws_item.setFlags(ws_item.flags() & ~Qt.ItemIsEditable)  # Make the item read-only

            table.setItem(row, 1, ws_item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Stretch columns to fit the table width

        # Populate combo boxes with layer names
        self.populate_combo_boxes()

    def populate_combo_boxes(self):
        if 'Deleted' in self.df.columns:
            layer_names = self.df[self.df['Deleted'] == False]['Layer Name'].tolist()
        else:
            layer_names = self.df['Layer Name'].tolist()

        self.ui.addws_CB.clear()
        self.ui.addnonws_CB.clear()

        # Add "Select" as the first item
        self.ui.addws_CB.addItem("Select")
        self.ui.addnonws_CB.addItem("Select")

        # Add the layer names
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

        if self.combined_df is  None:
            self.df.at[selected_row, 'Whitespace Settings'] = ws_value
            print('changing ws df')        ###delete
            print(self.df) ###delete
        else:
            self.combined_df.at[selected_row, 'Whitespace Settings'] = ws_value
            print('changing ws combined df')        ###delete
            print(self.combined_df) ###delete


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
            if self.combined_df is not None:
                self.combined_df.at[row, 'Whitespace Settings'] = ws_value

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

        if color_dialog.exec():
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
        #Cear table before populating
        self.ui.component_TW.setRowCount(0)
        self.ui.component_TW.setColumnCount(0)

        # Get the layer names from the combined DataFrame
        layer_names = self.combined_df['Layer Name'].tolist()

        # Configure component_TW
        self.ui.component_TW.setColumnCount(1)
        self.ui.component_TW.setHorizontalHeaderLabels(["Annotation layers"])
        self.ui.component_TW.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) #data matches the table width

        for row, layer_name in enumerate(layer_names):
            self.ui.component_TW.insertRow(row)
            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)

            # Set background color to dark and text color to white
            item.setBackground(QColor(45, 45, 45))  # Dark color
            item.setForeground(QBrush(Qt.white))  # White text

            self.ui.component_TW.setItem(row, 0, item)

        # Connect itemChanged signal to slot (value gets gray after being checked)
        self.ui.component_TW.itemChanged.connect(self.on_item_changed)

    def on_item_changed(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor(200, 200, 200))  # Light gray
            item.setForeground(QBrush(Qt.black))  # Black text
        else:
            item.setBackground(QColor(45, 45, 45))  # Dark color

    def add_combo(self):

        # Create the combined DataFrame
        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Deleted'] = False
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1


        table = self.ui.tissue_segmentation_TW
        selected_items = table.selectedItems()

        selected_rows = list(set(item.row() for item in selected_items))
        if len(selected_rows) < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient Selection",
                                          "Please select at least two classes to combine.")
            return

        combo_name, ok = QtWidgets.QInputDialog.getText(self, "Combo Name", "Enter a name for the combined class:")
        if not ok or not combo_name:
            return

        color_dialog = QColorDialog(self)
        if color_dialog.exec():
            selected_color = color_dialog.selectedColor().getRgb()[:3]
        else:
            return

        # Create a mapping of layer names to their original indices
        original_indices = {name: index + 1 for index, name in enumerate(self.original_df['Layer Name'])}

        # Get the layer names from the selected rows
        if self.combined_df is None:
            selected_layer_names = [self.df.iloc[idx]['Layer Name'] for idx in selected_rows]
        else:
            selected_layer_names = [self.combined_df.iloc[idx]['Layer Name'] for idx in selected_rows]


        # Create the combined class with original indices
        layer_indices = sorted([original_indices[name] for name in selected_layer_names])

        combined_class = {
            "Layer Name": combo_name,
            "Color": selected_color,
            "Layer idx": layer_indices,
            "Whitespace Settings": None,
            "Deleted": False
        }

        # Find the position to insert the combined class (minor row number)
        insert_position = min(selected_rows)

        # Remove the selected rows
        self.combined_df = self.combined_df.drop(selected_rows).reset_index(drop=True)

        #Set whtiespace settings to None for the combined class
        combined_class['Whitespace Settings'] = None

        # Insert the new combined class at the position of the minor row number
        self.combined_df = pd.concat([self.combined_df.iloc[:insert_position], pd.DataFrame([combined_class]),
                                      self.combined_df.iloc[insert_position:]]).reset_index(drop=True)

        # Restore the whitespace settings for the remaining rows
        for i, row in enumerate(self.combined_df.index):
            if row not in selected_rows and 'Whitespace Settings' in self.combined_df.columns:
                self.combined_df.at[row, 'Whitespace Settings'] = self.df.at[row, 'Whitespace Settings']

        print("Combined DataFrame:")
        print(self.combined_df)
        self.populate_table_widget(self.combined_df)  # Populate the table with the updated DataFrame

    def delete_annotation_class(self):

        # Initialize combined_df if not already initialized
        if self.combined_df is None:
            columns_to_copy = [col for col in self.df.columns if col != 'Deleted']
            self.combined_df = self.df[columns_to_copy].copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1
            self.combined_df['Delete layer'] = False #Add delete layer column to combined_df if not already present

        table = self.ui.tissue_segmentation_TW
        selected_items = table.selectedItems()

        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Insufficient Selection", "Please select at least one class to delete.")
            return

        selected_rows = list(set(item.row() for item in selected_items))

        for idx in selected_rows:
            if pd.isna(self.combined_df.at[idx, 'Whitespace Settings']):
                QtWidgets.QMessageBox.warning(self, "Unable to Delete",
                                              "You can only delete a class if it has an assigned whitespace value.")
                return

        for idx in selected_rows:
            if isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                QtWidgets.QMessageBox.warning(self, "Invalid Selection",
                                              "Cannot delete a combined class. Please reset the list first.")
                return

        # Add 'Deleted' column to combined_df if not already present
        if 'Deleted' not in self.combined_df.columns:
            self.combined_df['Deleted'] = False

        # Mark the selected rows as deleted
        for idx in selected_rows:
            self.combined_df.at[idx, 'Deleted'] = True

        # Add 'Delete layer' column to self.df
        if 'Delete layer' not in self.df.columns:
            self.df['Delete layer'] = False
            for idx in selected_rows:
                original_idx = self.original_df[self.original_df['Layer Name'] == self.combined_df.at[idx, 'Layer Name']].index[
                    0]
                self.df.at[original_idx, 'Delete layer'] = True
        else:
            for idx in selected_rows:
                original_idx = self.original_df[self.original_df['Layer Name'] == self.combined_df.at[idx, 'Layer Name']].index[
                    0]
                self.df.at[original_idx, 'Delete layer'] = True

        # print("Marked rows as deleted:", selected_rows)
        print("Updated DataFrame with 'Delete layer' column:")
        print(self.df)

        # Populate the table with the updated DataFrame
        # print("Combined DataFrame after deletion:")
        # print(self.combined_df)
        self.populate_table_widget(self.combined_df)

        # Update the add whitespace/non whitespace comboboxes
        self.populate_combo_boxes()

    # Add or update these methods in the MainWindow class:
    def save_advanced_settings_and_close(self):

        # Component analysis
        component_layers = {}


        for row in range(self.ui.component_TW.rowCount()):
            item = self.ui.component_TW.item(row, 0)
            if item:
                layer_name = item.text()
                layer_indices = self.combined_df[self.combined_df['Layer Name'] == layer_name]['Layer idx'].values[0]

                if isinstance(layer_indices, list):  #if the layer is a combined layer
                    for original_idx in layer_indices:
                        is_checked = item.checkState() == Qt.Checked
                        component_layers[self.original_df.loc[original_idx - 1, 'Layer Name']] = is_checked  # Save True for checked items
                else: #Not a combined layer
                    is_checked = item.checkState() == Qt.Checked
                    component_layers[layer_name] = is_checked # Save True for checked items

        self.df['Component analysis'] = self.df['Layer Name'].map(component_layers)

        self.tile_size = int(self.ui.tts_CB.currentText())
        self.ntrain = self.ui.ttn_SB.value()
        self.nval = self.ui.vtn_SB.value()
        self.TA = self.ui.TA_SB.value()

        final_df = self.df
        combined_df = self.combined_df

        print("\nFinal Raw DataFrame with combined indexes:")
        print(final_df)
        print("\nFinal Combined DataFrame:")
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