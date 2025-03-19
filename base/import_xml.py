"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 15, 2024
"""
from .load_annotations import load_annotations
import os
import pickle
import time
import numpy as np


def read_xml_with_encoding(xml_path):
    """Helper function to read XML files with proper encoding detection."""
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252']

    # First try reading in binary mode to detect encoding
    with open(xml_path, 'rb') as binary_file:
        raw_data = binary_file.read()

        # Try each encoding
        for encoding in encodings_to_try:
            try:
                decoded_text = raw_data.decode(encoding)
                # If successful, return the text and encoding
                return decoded_text, encoding
            except UnicodeDecodeError:
                continue

    # If all encodings fail, use latin-1 with error replacement
    return raw_data.decode('latin-1', errors='replace'), 'latin-1'


def import_xml(annotations_file, xmlfile, dm=None, ra=None):
    """
   Reads an XML file and imports annotation data, saving it to a pickle file.
   'load_annotations' function to extract the annotation coordinates from an XML file, is required

   Parameters:
   annotations_file (str): The file path for the output pickle file.
   xmlfile (str): The file path for the input XML file.
   dm (str): String indicating date and time the xml file was modified.
   ra (float, optional): The reduced annotations value. Defaults to 0.

   Returns:
   xyout_df (pandas.DataFrame): A DataFrame containing the annotation labels and coordinates.
   reduced_annotations (float): The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise None.
   """
    if ra is None:
        ra = 0
    if dm is None:
        dm = []

    print(' 1. of 4. Importing annotation data from xml file')
    load_start = time.time()

    try:
        # Use our helper function to detect encoding
        xml_content, detected_encoding = read_xml_with_encoding(xmlfile)
        print(f'  Detected XML encoding: {detected_encoding}')

        # Call load_annotations which now has encoding handling too
        reduced_annotations, xyout_df = load_annotations(xmlfile)
    except Exception as e:
        print(f'  Error reading XML file: {str(e)}')
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), 0

    elapsed_time = time.time() - load_start
    print(
        f'Loading annotation took {np.floor(elapsed_time / 60)} minutes and {elapsed_time - 60 * np.floor(elapsed_time / 60)} seconds')
    reduced_annotations = float(reduced_annotations)
    if not xyout_df.empty:

        if ra == 1:
            reduced_annotations = ra
        xyout_df.iloc[:, 2:4] = xyout_df.iloc[:, 2:4] * reduced_annotations

        annotations_dir = os.path.dirname(annotations_file)
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)

        if os.path.exists(annotations_file):
            print('File already exists, updating data...')
            with open(annotations_file, 'rb') as f:
                try:
                    existing_data = pickle.load(f)
                except EOFError:
                    existing_data = {}

            existing_data.update({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': dm})

            with open(annotations_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            print(' Creating file...')
            with open(annotations_file, 'wb') as f:
                pickle.dump({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': dm}, f)
    return xyout_df, reduced_annotations