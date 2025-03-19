"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 05, 2024
"""

import xmltodict
import pandas as pd
import io
import codecs


def load_annotations(xml_file):
    """
       Load annotation coordinates from an XML file into a DataFrame. Handles different file encodings by trying
       multiple encodings if the default fails.

       Parameters:
       - xml_file (str): The path to the XML file containing annotations.

       Returns:
       - reduced_annotations (str or None): The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise None.
       - xyout_df (pandas.DataFrame): DataFrame containing the annotation labels and coordinates, organized as:
           'Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'.
       """
    # Try to detect encoding first
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252']
    detected_encoding = None

    # First try reading in binary mode to detect encoding
    with open(xml_file, 'rb') as binary_file:
        raw_data = binary_file.read()

        # Try each encoding
        for encoding in encodings_to_try:
            try:
                decoded_text = raw_data.decode(encoding)
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

    if detected_encoding is None:
        # If we can't detect the encoding, try a more desperate approach with error handling
        detected_encoding = 'latin-1'  # This will read almost anything, but might produce wrong characters

    # Now read with the detected encoding
    try:
        with open(xml_file, 'r', encoding=detected_encoding, errors='replace') as file:
            my_xml = file.read()
            my_dict = xmltodict.parse(my_xml)
    except Exception as e:
        raise ValueError(f"Failed to parse XML file {xml_file} with encoding {detected_encoding}: {str(e)}")

    xyout = []

    try:
        reduced_annotations = my_dict['Annotations'].get('@MicronsPerPixel',1)
        reduced_annotations = float(reduced_annotations)
    except:
        reduced_annotations = 1

    annotations = my_dict.get("Annotations", {}).get("Annotation", [])

    for layer in annotations:
        if 'Region' in layer.get("Regions", {}):  # checks weather there are annotations in the layer
            layer_id = int(layer.get('@Id'))
            regions = layer["Regions"]["Region"]
            if type(regions) == list:
                for annotation in regions:
                    annotation_number = float(annotation["@Id"])
                    vertices = annotation.get("Vertices", {}).get("Vertex", [])
                    for vertex in vertices:
                        x = float(vertex.get('@X'))
                        y = float(vertex.get('@Y'))
                        xyout.append([layer_id, annotation_number, x, y])
            elif type(regions) == dict:
                annotation_number = float(regions.get("@Id"))
                vertices = regions.get("Vertices", {}).get("Vertex", [])
                for vertex in vertices:
                    x = float(vertex.get('@X'))
                    y = float(vertex.get('@Y'))
                    xyout.append([layer_id, annotation_number, x, y])

    xyout_df = pd.DataFrame(xyout, columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])
    return reduced_annotations, xyout_df