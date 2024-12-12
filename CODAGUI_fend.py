"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: October 22, 2024
"""

from PySide6 import QtWidgets, QtCore
from CODA import Ui_MainWindow
from classify_im_fend import MainWindowClassify
import os
from datetime import datetime
import xmltodict
import pandas as pd
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem, QBrush,QRegularExpressionValidator
from PySide6.QtWidgets import QColorDialog, QHeaderView
from PySide6.QtCore import Qt,QRegularExpression
import pickle
import numpy as np
from base import save_model_metadata_GUI
pd.set_option('display.max_columns', None)


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
        self.ui.classify_PB.clicked.connect(self.open_classify)
        self.ui.prerecorded_PB.clicked.connect(self.browse_prerecorded_file)
        self.ui.trianing_LE.textChanged.connect(self.check_for_trained_model)
        self.ui.custom_img_LE.textChanged.connect(self.check_for_trained_model)
        self.ui.custom_test_img_LE.textChanged.connect(self.check_for_trained_model)
        self.ui.custom_img_PB.clicked.connect(lambda: self.browse_image_folder('training'))
        self.ui.custom_test_img_PB.clicked.connect(lambda: self.browse_image_folder('testing'))
        self.ui.model_name.textChanged.connect(self.check_for_trained_model)
        self.ui.resolution_CB.currentIndexChanged.connect(self.check_for_trained_model)
        self.ui.use_anotated_images_CB.stateChanged.connect(self.check_for_trained_model)
        self.ui.create_downsample_CB.stateChanged.connect(self.check_for_trained_model)
        self.combo_colors = {}
        self.original_df = None  # Initialize original_df
        self.df = None
        self.prerecorded_data = False
        self.combined_df = None  # Initialize combined_df
        self.delete_count = 0
        self.combo_count = 0
        self.classify = False
        self.img_type = '.ndpi'
        self.test_img_type = '.ndpi'




        self.set_initial_model_name()
        self.ui.tabWidget.setCurrentIndex(0)  # Initialize the first tab
        for i in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, False)

        self.setWindowTitle("CODA Vision")

    def set_initial_model_name(self):
        """Set the initial text of the model_name text box to today's date."""
        today = datetime.now()
        date_string = today.strftime("%m_%d_%Y")
        self.ui.model_name.setText(date_string)

    def select_imagedir(self, purpose):
        dialog_title = f'Select {purpose.capitalize()} Image Directory'
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, dialog_title, os.getcwd())
        if folder_path:
            if os.path.isdir(folder_path):
                if purpose == 'training':
                    self.ui.trianing_LE.setText(folder_path)
                elif purpose == 'testing':
                    self.ui.testing_LE.setText(folder_path)
            else:
                self.ui.path_check.exec_()

    def browse_prerecorded_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Prerecorded Data File", "",
                                                             "Data Files (*.pkl)")
        if file_path:
            self.load_prerecorded_data(file_path)

    def browse_image_folder(self, purpose):
        dialog_title = f'Select Uncompressed {purpose.capitalize()} Image Directory'
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, dialog_title, os.getcwd())
        if folder_path:
            if os.path.isdir(folder_path):
                if purpose == 'training':
                    self.ui.custom_img_LE.setText(folder_path)
                elif purpose == 'testing':
                    self.ui.custom_test_img_LE.setText(folder_path)
            else:
                self.ui.path_check.exec_()

    def load_xml(self):
        xml_file = None
        training_folder = self.ui.trianing_LE.text()
        for file in os.listdir(training_folder):
            if file.endswith('.xml'):
                xml_file = os.path.join(training_folder, file)
                break
        if xml_file:
            try:
                self.df = self.parse_xml_to_dataframe(xml_file)
                self.original_df = self.df.copy()  # Initialize original_df after loading data
                print(f"Loaded XML file: {xml_file}")
                print(self.df)
                self.populate_table_widget()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to parse XML file: {str(e)}')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No XML file found in the training annotations folder.')

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
                {'Layer Name': layer_name.replace(" ", "_") , 'Color': rgb, 'Whitespace Settings': None})  # Add whitespace settings

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

    def load_prerecorded_data(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                self.pth_net = file_path
                self.df = data['final_df']
                self.original_df = self.df.copy()  # Set original_df in MainWindow
                self.combined_df = data['combined_df']
                self.ui.batch_size_SB.setValue(data['batch_size'])
                ws = data['WS']
                nm = data['nm']
                self.nm = nm
                self.prerecorded_data = True
                try:
                    model_type = data['model_type']
                except:
                    model_type = None
                # Populate the training_LE, testing_LE, and resolution_CB fields
                umpix_to_resolution = {1: '10x', 2: '5x', 4: '1x'}
                pthim = data.get('pthim', '')
                self.pthim = os.sep.join(pthim.split(os.sep)[:-1])
                self.ui.trianing_LE.setText(self.pthim)
                self.ui.testing_LE.setText(data.get('pthtest', ''))
                umpix = data.get('umpix','')
                self.resolution = umpix_to_resolution.get(umpix, 'Custom')
                if self.resolution == 'Custom':
                    self.ui.custom_img_LE.setText(data['uncomp_train_pth'])
                    self.ui.custom_test_img_LE.setText(data['uncomp_test_pth'])
                    self.ui.custom_scale_LE.setText(data['scale'])
                self.ui.resolution_CB.setCurrentText(self.resolution)
                if model_type:
                    self.ui.model_type_CB.setCurrentText(model_type)


                self.populate_table_widget(self.combined_df)

                # Initialize combo boxes with chosen annotation classes from ws
                addws_layer_name = self.df.iloc[ws[1][0] - 1]['Layer Name']
                addnonws_layer_name = self.df.iloc[ws[1][1] - 1]['Layer Name']
                count = 0
                for i in self.combined_df['Layer idx']:
                    if isinstance(i, list) and np.isin(ws[1][0], i):
                        addws_layer_name = self.combined_df.iloc[count]['Layer Name']
                    elif isinstance(i, list) and np.isin(ws[1][1], i):
                        addnonws_layer_name = self.combined_df.iloc[count]['Layer Name']
                    count += 1
                self.ui.addws_CB.setCurrentText(addws_layer_name)
                self.ui.addnonws_CB.setCurrentText(addnonws_layer_name)

                # Initialize advanced settings
                self.tile_size = data['sxy']
                self.ntrain = data['ntrain']
                self.nval = data['nvalidate']
                self.TA = data['nTA']
                self.load_saved_values()


                print("Prerecorded data loaded successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load prerecorded data: {str(e)}')

        model_exists = False
        if os.path.isdir(os.sep.join(pthim.split(os.sep)[:-1])+os.sep+nm):
            for file in os.listdir(os.sep.join(pthim.split(os.sep)[:-1])+os.sep+nm):
                if 'best_model' in file and file.endswith('.keras'):
                    model_exists = True

        if model_exists:
            self.ui.classify_PB.setVisible(True)
            self.ui.classify_PB.setEnabled(True)
            self.classification_source = 1
        else:
            self.ui.classify_PB.setVisible(False)
            self.ui.classify_PB.setEnabled(False)


    def check_for_trained_model(self):
        model_exists = False
        if os.path.isdir(os.path.join(self.ui.trianing_LE.text(),self.ui.model_name.text())):
            for file in os.listdir(os.path.join(self.ui.trianing_LE.text(),self.ui.model_name.text())):
                if 'best_model' in file and file.endswith('.keras'):
                    model_exists = True

        if self.ui.resolution_CB.currentText() == 'Custom':
            self.ui.label_43.setVisible(True)
            self.ui.custom_scale_LE.setVisible(True)
            self.ui.label_45.setVisible(True)
            self.ui.use_anotated_images_CB.setVisible(True)
            if not(self.ui.use_anotated_images_CB.isChecked()):
                self.ui.label_42.setVisible(True)
                self.ui.custom_img_LE.setVisible(True)
                self.ui.custom_img_PB.setVisible(True)
                self.ui.label_44.setVisible(True)
                self.ui.custom_test_img_LE.setVisible(True)
                self.ui.custom_test_img_PB.setVisible(True)
                self.ui.label_46.setVisible(True)
                self.ui.label_47.setVisible(True)
                self.ui.create_downsample_CB.setVisible(True)
                if not(self.ui.create_downsample_CB.isChecked()):
                    self.ui.label_48.setVisible(True)
                else:
                    self.ui.label_48.setVisible(False)
            else:
                self.ui.label_42.setVisible(False)
                self.ui.custom_img_LE.setVisible(False)
                self.ui.custom_img_PB.setVisible(False)
                self.ui.label_44.setVisible(False)
                self.ui.custom_test_img_LE.setVisible(False)
                self.ui.custom_test_img_PB.setVisible(False)
                self.ui.label_46.setVisible(False)
                self.ui.label_47.setVisible(False)
                self.ui.label_48.setVisible(False)
                self.ui.create_downsample_CB.setVisible(False)
                self.ui.create_downsample_CB.setChecked(True)



        else:
            self.ui.label_42.setVisible(False)
            self.ui.custom_img_LE.setVisible(False)
            self.ui.custom_img_PB.setVisible(False)
            self.ui.label_44.setVisible(False)
            self.ui.custom_test_img_LE.setVisible(False)
            self.ui.custom_test_img_PB.setVisible(False)
            self.ui.label_43.setVisible(False)
            self.ui.custom_scale_LE.setVisible(False)
            self.ui.label_45.setVisible(False)
            self.ui.label_46.setVisible(False)
            self.ui.use_anotated_images_CB.setVisible(False)
            self.ui.label_47.setVisible(False)
            self.ui.label_48.setVisible(False)
            self.ui.create_downsample_CB.setVisible(False)

            if model_exists and os.path.isdir(os.path.join(self.ui.trianing_LE.text(),self.ui.resolution_CB.currentText())):
                self.ui.classify_PB.setVisible(True)
                self.ui.classify_PB.setEnabled(True)
                self.pthim = self.ui.trianing_LE.text()
                self.resolution = self.ui.resolution_CB.currentText()
                self.nm = self.ui.model_name.text()
                with open(os.path.join(self.pthim,self.nm,'net.pkl'), 'rb') as file:
                    data = pickle.load(file)
                    self.model_type = data['model_type']
                self.classification_source = 2
            else:
                self.ui.classify_PB.setVisible(False)
                self.ui.classify_PB.setEnabled(False)

    def fill_form_and_continue(self):
        """Fill the form, process data, and switch to the next tab if successful."""
        if self.fill_form():
            if not self.prerecorded_data:
                self.load_xml()  # Load and parse the XML file only if not using prerecorded data
            next_tab_index = self.ui.tabWidget.currentIndex() + 1
            if next_tab_index < self.ui.tabWidget.count():
                self.switch_to_next_tab()

    def reset_combo(self):
        if self.original_df is not None:
            self.df = self.original_df.copy()  # Reset df to the original data
            self.df['Delete layer'] = False
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

        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1
            self.combined_df['Deleted'] = False
            self.combined_df['Component analysis'] = np.nan

        self.add_ws_to = self.ui.addws_CB.currentIndex()
        self.add_nonws_to = self.ui.addnonws_CB.currentIndex()
        self.delete_count = 0
        self.combo_count = []
        placehold_ws = self.add_ws_to
        placehold_nonws = self.add_nonws_to
        if self.add_ws_to == self.add_nonws_to:
            placehold_nonws = -3
        iteration = self.combined_df.index
        num_lists = []
        for x in self.combined_df['Layer idx']:
            if isinstance(x, list):
                num_lists.extend(x[1:])
        iteration = pd.RangeIndex(start=iteration.start, stop=iteration.stop + len(num_lists), step=iteration.step)
        for idx in iteration:
            if np.sum([x-1 <=idx for x in self.combo_count]) > 0:
                if idx - self.delete_count - np.sum([x - 1 <= idx for x in self.combo_count]) + 1 == placehold_ws:
                    if idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                        self.add_ws_to = self.combined_df.at[idx, 'Layer idx'][0]-1
                        self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])
                    else:
                        self.add_ws_to = idx
                    placehold_ws = -2
                elif idx - self.delete_count - np.sum([x - 1 <= idx for x in self.combo_count]) + 1 == placehold_nonws:
                    if idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                        self.add_nonws_to = self.combined_df.at[idx, 'Layer idx'][0]-1
                        self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])
                    else:
                        self.add_nonws_to = idx
                    placehold_nonws = -2
                elif idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                    self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])
                if idx < iteration.stop-len(num_lists) and self.combined_df.at[idx, 'Deleted'] == True:
                    self.delete_count += 1
            else:
                if idx < iteration.stop-len(num_lists) and self.combined_df.at[idx, 'Deleted'] == True:
                    self.delete_count += 1
                if idx - self.delete_count - np.sum([x - 1 <= idx for x in self.combo_count]) + 1 == placehold_ws:
                    if idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                        self.add_ws_to = self.combined_df.at[idx, 'Layer idx'][0]-1
                        self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])
                    else:
                        self.add_ws_to = idx
                    placehold_ws = -2
                elif idx - self.delete_count - np.sum([x - 1 <= idx for x in self.combo_count]) + 1 == placehold_nonws:
                    if idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                        self.add_nonws_to = self.combined_df.at[idx, 'Layer idx'][0]-1
                        self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])
                    else:
                        self.add_nonws_to = idx
                    placehold_nonws = -2
                elif idx < iteration.stop-len(num_lists) and isinstance(self.combined_df.at[idx, 'Layer idx'], list):
                    self.combo_count.extend(self.combined_df.at[idx, 'Layer idx'][1:])

        self.add_ws_to += 1
        self.add_nonws_to += 1
        if placehold_nonws == -3:
            self.add_nonws_to = self.add_ws_to

        # Create a mapping of original indices to layer names
        original_indices = {index + 1: name for index, name in enumerate(self.original_df['Layer Name'])}

        # Initialize combined_df if it is None
        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1
            self.combined_df['Component analysis'] = np.nan

        # Ensure 'Layer idx' column exists
        if 'Layer idx' not in self.combined_df.columns:
            self.combined_df['Layer idx'] = self.combined_df.index + 1

        # Ensure 'Delete layer' column exists
        if 'Delete layer' not in self.df.columns:
            self.df['Delete layer'] = False

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

        # Determine the source dataframe based on the checkbox state
        if self.ui.nesting_checkBox.isChecked():
            source_df = self.df.copy()
            for i in source_df.index:
                count = 0
                for j in self.combined_df['Layer idx']:
                    if ((isinstance(j, list) and np.isin(i+1,j)) or ((not isinstance(j, list)) and i+1==j)):
                        source_df.at[i, 'Color'] = self.combined_df.at[count, 'Color']
                    count += 1
        else:
            source_df = self.combined_df.copy()
            # Ensure 'Deleted' column exists
            if 'Deleted' not in source_df.columns:
                source_df['Deleted'] = False
            source_df = source_df[source_df['Deleted'] != True]

        if self.prerecorded_data:
            # Use the Nesting column from final_df to determine the order
            nesting_order = self.df['Nesting'].tolist()

            # Create a mapping of combined indices to their respective names
            combined_indices_to_names = {}
            for idx, row in source_df.iterrows():
                if self.ui.nesting_checkBox.isChecked():
                    layer_indices = idx
                else:
                    layer_indices = row['Layer idx']
                if isinstance(layer_indices, list):
                    if layer_indices:
                        combined_indices_to_names[layer_indices[0]] = row['Layer Name']
                else:
                    combined_indices_to_names[layer_indices] = row['Layer Name']

            # Populate the table with the combined names in the correct order
            for original_idx in nesting_order:
                if self.ui.nesting_checkBox.isChecked():
                    layer_name = combined_indices_to_names.get(original_idx-1)
                else:
                    layer_name = combined_indices_to_names.get(original_idx)
                if layer_name:
                    color = source_df[source_df['Layer Name'] == layer_name]['Color'].values[0]
                    self.add_item_to_model(model, layer_name, color)
        else:
            # Populate the table with the source dataframe
            for index, row in source_df.iterrows():
                self.add_item_to_model(model, row['Layer Name'], row['Color'])

        self.ui.nesting_TW.setModel(model)
        self.ui.nesting_TW.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_item_to_model(self, model, layer_name, color):
        item = QStandardItem(layer_name)
        item.setBackground(QColor(*color))
        item.setEditable(False)

        # Convert the background color to greyscale
        greyscale = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        if greyscale > 128:  # If greyscale is above 50% grey, set text color to black
            item.setForeground(QBrush(Qt.black))
        else:  # Otherwise, set text color to white
            item.setForeground(QBrush(Qt.white))

        model.appendRow(item)


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

        # Check if advanced settings need to be modified
        if self.ui.AS_checkBox.isChecked():
            current_tab_index = self.ui.tabWidget.currentIndex()
            next_tab_index = current_tab_index + 1

            if next_tab_index < self.ui.tabWidget.count():
                self.ui.tabWidget.setTabEnabled(current_tab_index, False)  # Disable current tab
                self.ui.tabWidget.setTabEnabled(next_tab_index, True)  # Enable next tab
                self.ui.tabWidget.setCurrentIndex(next_tab_index)  # Switch to next tab

            self.initialize_advanced_settings()
        else:
            self.initialize_advanced_settings()
            self.save_advanced_settings_and_close()

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

        if not os.path.isdir(pth):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The folder selected for the training annotations does not exist.')
            return False

        if not os.path.isdir(pthtest):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The folder selected for the testing annotations does not exist.')
            return False

        # Check for .xml files in training path
        if not any(f.endswith(('.xml')) for f in os.listdir(pth)):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The selected training path does not contain .xml files')
            return False

        # Check for .xml files in testing path
        if not any(f.endswith(('.xml')) for f in os.listdir(pthtest)):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The selected testing path does not contain .xml files')
            return False

        # Check if resolution is selected
        if resolution == "Select":
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please chose a resolution from the drop down box')
            return False
        elif resolution == "Custom":
            if not(self.ui.use_anotated_images_CB.isChecked()):
                custom_train = self.ui.custom_img_LE.text()
                custom_test = self.ui.custom_test_img_LE.text()
                if not os.path.isdir(custom_train):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The folder selected for the training images does not exist.')
                    return False

                if not os.path.isdir(custom_test):
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The folder selected for the testing images does not exist.')
                    return False
                if any(f.endswith(('.ndpi')) for f in os.listdir(custom_train)):
                    self.img_type = '.ndpi'
                elif any(f.endswith(('.dcm')) for f in os.listdir(custom_train)):
                    self.img_type = '.dcm'
                elif any(f.endswith(('.tif')) for f in os.listdir(custom_train)):
                    self.img_type = '.tif'
                elif any(f.endswith(('.jpg')) for f in os.listdir(custom_train)):
                    self.img_type = '.jpg'
                elif any(f.endswith(('.png')) for f in os.listdir(custom_train)):
                    self.img_type = '.png'
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected uncompressed training images path does not contain'
                                                  ' .ndpi, .dcm, .tif, .png or .jpg files')
                    return False

                if any(f.endswith(('.ndpi')) for f in os.listdir(custom_test)):
                    self.test_img_type = '.ndpi'
                elif any(f.endswith(('.dcm')) for f in os.listdir(custom_test)):
                    self.test_img_type = '.dcm'
                elif any(f.endswith(('.tif')) for f in os.listdir(custom_test)):
                    self.test_img_type = '.tif'
                elif any(f.endswith(('.jpg')) for f in os.listdir(custom_test)):
                    self.img_type = '.jpg'
                elif any(f.endswith(('.png')) for f in os.listdir(custom_test)):
                    self.img_type = '.png'
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected uncompressed testing images path does not contain'
                                                  ' .ndpi, .dcm, .tif, .png or .jpg files')
                    return False
            else:
                train = self.ui.trianing_LE.text()
                test = self.ui.testing_LE.text()
                if any(f.endswith(('.ndpi')) for f in os.listdir(train)):
                    self.img_type = '.ndpi'
                elif any(f.endswith(('.dcm')) for f in os.listdir(train)):
                    self.img_type = '.dcm'
                elif any(f.endswith(('.tif')) for f in os.listdir(train)):
                    self.img_type = '.tif'
                elif any(f.endswith(('.jpg')) for f in os.listdir(train)):
                    self.img_type = '.jpg'
                elif any(f.endswith(('.png')) for f in os.listdir(train)):
                    self.img_type = '.png'
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected training annotation path does not contain'
                                                  ' .ndpi, .dcm, .tif, .png or .jpg files')
                if any(f.endswith(('.ndpi')) for f in os.listdir(test)):
                    self.img_type = '.ndpi'
                elif any(f.endswith(('.dcm')) for f in os.listdir(test)):
                    self.img_type = '.dcm'
                elif any(f.endswith(('.tif')) for f in os.listdir(test)):
                    self.img_type = '.tif'
                elif any(f.endswith(('.jpg')) for f in os.listdir(test)):
                    self.img_type = '.jpg'
                elif any(f.endswith(('.png')) for f in os.listdir(test)):
                    self.img_type = '.png'
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                                  'The selected testing annotation path does not contain'
                                                  ' .ndpi, .dcm, .tif, .png or .jpg files')

            scale = self.ui.custom_scale_LE.text()
            try:
                scale = float(scale)
                if scale < 1:
                    QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'Introduce a valid scaling factor')
                    self.ui.custom_scale_LE.setText('1')
                    return False
            except:
                QtWidgets.QMessageBox.warning(self, 'Warning',
                                              'Introduce a valid scaling factor')
                self.ui.custom_scale_LE.setText('1')
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

    def populate_table_widget(self, df=None, coloring = False):
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
        if not coloring:
            self.populate_combo_boxes()

    def populate_combo_boxes(self):
        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1
            self.combined_df['Deleted'] = False
            self.combined_df['Component analysis'] = np.nan

        if 'Delete layer' in self.df.columns or any(isinstance(idx, list) for idx in self.combined_df['Layer idx']):
            layer_names = self.combined_df[self.combined_df['Deleted'] == False]['Layer Name'].tolist()
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
        updated_selected_row = selected_row

        delete_count = 0
        # Mark the selected rows as deleted
        for idx in self.combined_df.index:
            if self.combined_df.at[idx, 'Deleted'] == True:
                delete_count += 1
            elif idx - delete_count == selected_row:
                updated_selected_row = idx
        ws_value = ws_map[ws_option]

        if self.combined_df is  None:
            self.df.at[updated_selected_row, 'Whitespace Settings'] = ws_value
        else:
            self.combined_df.at[updated_selected_row, 'Whitespace Settings'] = ws_value


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
        self.delete_count = 0
        # Initialize combined_df if it is None
        if self.combined_df is None:
            self.combined_df = self.df.copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1
            self.combined_df['Deleted'] = False
            self.combined_df['Component analysis'] = np.nan


        table = self.ui.tissue_segmentation_TW
        selected_items = table.selectedItems()

        if not selected_items:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select an annotation class from the table.')
            return

        selected_row = selected_items[0].row()

        updated_selected_row = selected_row

        # Mark the selected rows as deleted
        for idx in self.combined_df.index:
            if self.combined_df.at[idx, 'Deleted'] == True:
                self.delete_count += 1
            elif idx-self.delete_count == selected_row:
                updated_selected_row = idx

        current_color = self.combined_df.iloc[updated_selected_row]['Color']
        initial_color = QColor(*current_color)

        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(initial_color)



        if color_dialog.exec():
            new_color = color_dialog.currentColor()
            new_rgb = (new_color.red(), new_color.green(), new_color.blue())

            # Update the DataFrame
            self.combined_df.at[updated_selected_row, 'Color'] = new_rgb

            # Update only the Annotation Class column in the table
            item = table.item(selected_row, 0)
            if item:
                item.setBackground(new_color)

            layer_name = self.combined_df.iloc[updated_selected_row]['Layer Name']
            print(f"Color changed for {layer_name} to {new_rgb}")

        self.populate_table_widget(self.combined_df, coloring = True)

    def initialize_advanced_settings(self):
        # Clear table before populating
        self.ui.component_TW.setRowCount(0)
        self.ui.component_TW.setColumnCount(0)

        # Get the layer names from the combined DataFrame
        layer_names = [self.combined_df['Layer Name'][layer] for layer in self.combined_df.index
                       if self.combined_df['Deleted'][layer] == False]

        # Configure component_TW
        self.ui.component_TW.setColumnCount(1)
        self.ui.component_TW.setHorizontalHeaderLabels(["Annotation layers"])
        self.ui.component_TW.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)  # Data matches the table width

        for row, layer_name in enumerate(layer_names):
            self.ui.component_TW.insertRow(row)
            item = QtWidgets.QTableWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            # Check the item if prerecorded data is loaded and Component analysis is True
            if self.prerecorded_data:
                if self.combined_df is not None and not np.isnan(self.combined_df['Component analysis'][0]):
                    component_analysis_value = self.combined_df.loc[
                        self.combined_df['Layer Name'] == layer_name, 'Component analysis'].values
                    if component_analysis_value:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Unchecked)

            # Set background color to dark and text color to white
            if item.checkState() == Qt.Checked:
                item.setBackground(QColor(200, 200, 200))  # Light gray
                item.setForeground(QBrush(Qt.black))  # Black text
            else:
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
            item.setBackground(QColor(45, 45, 45))# Dark color
            item.setForeground(QBrush(Qt.white))

    def add_combo(self):
        self.delete_count = 0
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
        if not ok or not combo_name or not all(char.isalnum() or char in ' _' for char in combo_name):
            QtWidgets.QMessageBox.warning(self, "Invalid Combo Name",
                                          "Please introduce a combo name that does not contain any especial characters.")
            return

        color_dialog = QColorDialog(self)
        if color_dialog.exec():
            selected_color = color_dialog.selectedColor().getRgb()[:3]
        else:
            return

        # Create a mapping of layer names to their original indices
        original_indices = {name: index + 1 for index, name in enumerate(self.original_df['Layer Name'])}
        updated_selected_rows = np.zeros(len(selected_rows)).astype(int)
        position = updated_selected_rows.copy()
        position[:] = -1

        # Get the layer names from the selected rows
        if self.combined_df is None:
            selected_layer_names = [self.df.iloc[idx]['Layer Name'] for idx in selected_rows]
        else:
            if 'Deleted' in self.combined_df.columns:
                for idx in self.combined_df.index:
                    if self.combined_df.at[idx, 'Deleted'] == True:
                        self.delete_count += 1
                    elif idx - self.delete_count in selected_rows:
                        upd_idx = np.where(position == -1)[0][0]
                        position[upd_idx] = 0
                        updated_selected_rows[upd_idx] = idx
            selected_layer_index = [self.combined_df.iloc[idx]['Layer idx'] for idx in updated_selected_rows]
            selected_layer_index = [item for sublist in selected_layer_index for item in
                         (sublist if isinstance(sublist, list) else [sublist])]
            selected_layer_names = [self.df.iloc[idx-1]['Layer Name'] for idx in selected_layer_index]

        # Create the combined class with original indices
        layer_indices = sorted([original_indices[name] for name in selected_layer_names])

        combined_class = {
            "Layer Name": combo_name.replace(" ", "_"),
            "Color": selected_color,
            "Layer idx": layer_indices,
            "Whitespace Settings": None,
            "Deleted": False
        }

        # Find the position to insert the combined class (minor row number)
        insert_position = min(selected_rows)

        # Remove the selected rows
        self.combined_df = self.combined_df.drop(updated_selected_rows).reset_index(drop=True)

        #Set whtiespace settings to None for the combined class
        combined_class['Whitespace Settings'] = None

        # Insert the new combined class at the position of the minor row number
        self.combined_df = pd.concat([self.combined_df.iloc[:insert_position], pd.DataFrame([combined_class]),
                                      self.combined_df.iloc[insert_position:]]).reset_index(drop=True)

        # Restore the whitespace settings for the remaining rows
        # for i, row in enumerate(self.combined_df.index):
        #     if row not in selected_rows and 'Whitespace Settings' in self.combined_df.columns:
        #         self.combined_df.at[row, 'Whitespace Settings'] = self.df.at[row, 'Whitespace Settings']

        print("Combined DataFrame:")
        print(self.combined_df)
        self.populate_table_widget(self.combined_df)  # Populate the table with the updated DataFrame

        # Populate whitespace/non-whitespace combo boxes
        self.populate_combo_boxes()
        self.combined_df['Component analysis'] = np.nan

    def delete_annotation_class(self):
        self.delete_count = 0
        # Initialize combined_df if not already initialized
        if self.combined_df is None:
            columns_to_copy = [col for col in self.df.columns if col != 'Deleted']
            self.combined_df = self.df[columns_to_copy].copy()
            self.combined_df['Layer idx'] = self.combined_df.index + 1  # Store the original row numbers +1
            self.combined_df['Component analysis'] = np.nan

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

        updated_selected_rows = selected_rows.copy()
        position = np.zeros(len(selected_rows)).astype(int)
        position[:] = -1


        # Mark the selected rows as deleted
        for idx in self.combined_df.index:
            if self.combined_df.at[idx, 'Deleted'] == True:
                self.delete_count += 1
            elif idx-self.delete_count in selected_rows:
                self.combined_df.at[idx, 'Deleted'] = True
                upd_idx = np.where(position == -1)[0][0]
                position[upd_idx] = 0
                updated_selected_rows[upd_idx] = idx

        # Add 'Delete layer' column to self.df
        if 'Delete layer' not in self.df.columns:
            self.df['Delete layer'] = False
            for idx in updated_selected_rows:
                original_idx = self.original_df[self.original_df['Layer Name'] == self.combined_df.at[idx, 'Layer Name']].index[
                    0]
                self.df.at[original_idx, 'Delete layer'] = True
        else:
            for idx in updated_selected_rows:
                original_idx = self.original_df[self.original_df['Layer Name'] == self.combined_df.at[idx, 'Layer Name']].index[
                    0]
                self.df.at[original_idx, 'Delete layer'] = True

        # print("Marked rows as deleted:", selected_rows)
        print("Updated DataFrame with 'Delete layer' column:")
        print(self.df)

        # Populate the table with the updated DataFrame
        self.populate_table_widget(self.combined_df)

        # Update the add whitespace/non whitespace comboboxes
        self.populate_combo_boxes()
        self.combined_df['Component analysis'] = np.nan

    # Add or update these methods in the MainWindow class:
    def save_advanced_settings_and_close(self):

        # Component analysis
        component_layers = {}
        combined_component = {}


        for row in range(self.ui.component_TW.rowCount()):
            item = self.ui.component_TW.item(row, 0)
            if item:
                layer_name = item.text()
                layer_indices = self.combined_df[self.combined_df['Layer Name'] == layer_name]['Layer idx'].values[0]
                layer_indices_combined = self.combined_df[self.combined_df['Layer Name'] == layer_name].index

                if isinstance(layer_indices, list):  #if the layer is a combined layer
                    for original_idx in layer_indices:
                        is_checked = item.checkState() == Qt.Checked
                        component_layers[self.original_df.loc[original_idx - 1, 'Layer Name']] = is_checked  # Save True for checked items
                else: #Not a combined layer
                    is_checked = item.checkState() == Qt.Checked
                    component_layers[layer_name] = is_checked # Save True for checked items
                if self.combined_df is not None:
                    for idx in layer_indices_combined:
                        combined_component[self.combined_df.loc[idx, 'Layer Name']] = is_checked

        self.df['Component analysis'] = self.df['Layer Name'].map(component_layers)
        self.combined_df['Component analysis'] = self.combined_df['Layer Name'].map(combined_component)
        for idx in self.df.index:
            if np.isnan(self.df['Component analysis'][idx]):
                self.df.at[idx, 'Component analysis'] = False
        for idx in self.combined_df.index:
            if np.isnan(self.combined_df['Component analysis'][idx]):
                self.combined_df.at[idx, 'Component analysis'] = False

        self.tile_size = int(self.ui.tts_CB.currentText())
        self.ntrain = self.ui.ttn_SB.value()
        self.nval = self.ui.vtn_SB.value()
        self.TA = self.ui.TA_SB.value()

        #Save model metadata onto pickle file
        self.variable_parametrization_to_WS()
        self.close()



    def variable_parametrization_to_WS(self):

        final_df = self.df
        combined_df = self.combined_df

        # Paths
        pth = self.ui.trianing_LE.text()
        pthtest = self.ui.testing_LE.text()
        model_name = self.ui.model_name.text()
        resolution = self.ui.resolution_CB.currentText()
        pthim = os.path.join(pth, f'{resolution}')
        pthDL = os.path.join(pth, model_name)

        # Tif resolution
        resolution_to_umpix = {"10x": 1, "5x": 2, "1x": 4}
        if resolution == 'Custom':
            umpix = 'TBD'
        else:
            umpix = resolution_to_umpix.get(resolution, 2)  # Default to 2 if resolution not found
        self.umpix = umpix

        # Get the dataframe with annotation information
        classNames = combined_df['Layer Name'].tolist()
        colormap = combined_df['Color'].tolist()

        # Training tile size
        tile_size = self.tile_size
        # Number of training tiles
        ntrain = self.ntrain

        # Number of validations tiles
        nvalidate = self.nval
        # Number of TA images to evaluate (coming soon)
        nTA = self.TA
        #Type of model
        model_type = self.ui.model_type_CB.currentText()
        self.model_type = model_type
        self.resolution = self.ui.resolution_CB.currentText()
        # Batch size
        batch_size = self.ui.batch_size_SB.value()
        # Create WS

        layers_to_delete = final_df.index[final_df['Delete layer']==True].tolist()
        layers_to_delete = [i + 1 for i in layers_to_delete]  # get row index starting from 1
        nesting_list = final_df['Nesting'].tolist()
        nesting_list.reverse()
        WS = [final_df['Whitespace Settings'].tolist(),
              [self.add_ws_to, self.add_nonws_to],
              final_df['Combined layers'].tolist(),
              nesting_list,
              layers_to_delete
              ]

        # numclass = max(WS[2])
        # nblack = numclass + 1
        # nwhite = WS[1][0]

        colormap = np.array(colormap)

        print("\nFinal Raw DataFrame with combined indexes:")
        print(final_df)
        print("\nFinal Combined DataFrame:")
        print(combined_df)

        # Final Parameters
        print('Classnames: ', classNames)
        print('Colormap: ', colormap)
        print('WS', WS)

        self.create_down = self.ui.create_downsample_CB.isChecked()
        self.downsamp_annotated_images = self.ui.use_anotated_images_CB.isChecked()

        # Save model metadata onto pickle file
        if self.resolution == 'Custom':
            self.uncomp_train_pth = self.ui.custom_img_LE.text()
            self.uncomp_test_pth = self.ui.custom_test_img_LE.text()
            self.scale = self.ui.custom_scale_LE.text()
            save_model_metadata_GUI.save_model_metadata_GUI(pthDL, pthim, pthtest, WS, model_name, umpix, colormap,
                                                            tile_size, classNames, ntrain, nvalidate, nTA, final_df,
                                                            combined_df, model_type, batch_size,
                                                            uncomp_train_pth = self.uncomp_train_pth,
                                                            uncomp_test_pth = self.uncomp_test_pth, scale = self.scale,
                                                            create_down = self.create_down,
                                                            downsamp_annotated = self.downsamp_annotated_images)
        else:
            save_model_metadata_GUI.save_model_metadata_GUI(pthDL, pthim, pthtest, WS, model_name, umpix, colormap,
                                                            tile_size, classNames, ntrain, nvalidate, nTA, final_df,
                                                            combined_df, model_type, batch_size)


    def load_saved_values(self):
        if self.prerecorded_data:
            self.ui.tts_CB.setCurrentText(str(self.tile_size))
            self.ui.ttn_SB.setValue(self.ntrain)
            self.ui.vtn_SB.setValue(self.nval)
            self.ui.TA_SB.setValue(self.TA)

    # Load paths
    def get_pthDL(self):
        pth = self.ui.trianing_LE.text()
        model_name = self.ui.model_name.text()
        return os.path.join(pth, model_name)

    def get_pthim(self):
        pth = self.ui.trianing_LE.text()
        resolution = self.ui.resolution_CB.currentText()
        print(self.ui.custom_scale_LE.text())
        if resolution == 'Custom':
            return os.path.join(pth, 'Custom_Scale_'+str(float(self.ui.custom_scale_LE.text())))
        else:
            return os.path.join(pth, f'{resolution}')

    def get_pthtestim(self):
        pthtest = self.ui.testing_LE.text()
        resolution = self.ui.resolution_CB.currentText()
        if resolution == 'Custom':
            print('fix when determine relation between scale and resolution at line 1299')
            return os.path.join(pthtest, 'Custom_Scale_'+str(float(self.ui.custom_scale_LE.text())))
        else:
            return os.path.join(pthtest, f'{resolution}')

    def open_classify(self):
        self.classify = True
        print('Closing model creation GUI. Initializing classification GUI')
        self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    # Load and apply the dark theme stylesheet
    with open('dark_theme.qss', 'r') as file:
        app.setStyleSheet(file.read())

    window = MainWindow()
    window.show()
    app.exec()
    if window.classify:
        window2 = MainWindowClassify(window.pthim, window.resolution, window.nm)
        window2.show()
        app.exec()