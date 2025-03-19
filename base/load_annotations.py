"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 05, 2024
"""

import xmltodict
import pandas as pd
import io
import codecs
import os


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
    # Verify the file exists
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")

    # Read the raw bytes for analysis
    with open(xml_file, 'rb') as binary_file:
        raw_data = binary_file.read()

    # Debug info
    print(f"Reading XML file: {xml_file}")
    print(f"File size: {len(raw_data)} bytes")
    print(f"First 20 bytes: {' '.join(f'{b:02x}' for b in raw_data[:20])}")

    # Check if the file might be binary or non-XML
    text_chars = bytearray(b' \t\n\r')
    text_chars.extend(range(32, 127))
    is_binary = bool(raw_data.translate(None, text_chars))
    if is_binary:
        print("Warning: The file appears to be binary (non-text)")

    # Check for known BOMs
    bom_encodings = {
        codecs.BOM_UTF8: 'utf-8',
        codecs.BOM_UTF16_LE: 'utf-16-le',
        codecs.BOM_UTF16_BE: 'utf-16-be',
        codecs.BOM_UTF32_LE: 'utf-32-le',
        codecs.BOM_UTF32_BE: 'utf-32-be',
    }

    detected_encoding = None
    decoded_text = None

    # First try to handle BOM-marked files
    for bom, encoding in bom_encodings.items():
        if raw_data.startswith(bom):
            print(f"Found BOM for encoding: {encoding}")
            # Remove the BOM
            raw_data_without_bom = raw_data[len(bom):]
            try:
                decoded_text = raw_data_without_bom.decode(encoding)
                detected_encoding = encoding
                print(f"Successfully decoded BOM-marked file with {encoding}")
                break
            except UnicodeDecodeError:
                print(f"Failed to decode BOM-marked file with {encoding}")

    # If no BOM or BOM handling failed, try various encodings
    if not decoded_text:
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252']
        for encoding in encodings_to_try:
            try:
                decoded_text = raw_data.decode(encoding)
                # Check if the decoded text looks like XML
                if decoded_text.strip().startswith('<?xml') or decoded_text.strip().startswith('<Annotations'):
                    detected_encoding = encoding
                    print(f"Successfully decoded with {encoding}, content looks like XML")
                    break
                else:
                    print(f"Decoded with {encoding}, but content doesn't look like XML")
                    # Still keep this decoding as a fallback
                    if not detected_encoding:
                        detected_encoding = encoding
                        decoded_text_fallback = decoded_text
            except UnicodeDecodeError:
                print(f"Failed to decode with {encoding}")

    # If all methods fail, use latin-1 with replacement as last resort
    if not decoded_text:
        try:
            decoded_text = raw_data.decode('latin-1', errors='replace')
            detected_encoding = 'latin-1 with replacements'
            print("Using latin-1 with error replacement as last resort")
        except Exception as e:
            raise ValueError(f"Failed to decode the file with any encoding: {str(e)}")

    # Print a preview of the decoded content
    print(f"Decoded content preview: {decoded_text[:100].strip()}")

    # Try to parse the XML
    try:
        my_dict = xmltodict.parse(decoded_text)
    except Exception as xml_parse_error:
        print(f"XML parsing error: {str(xml_parse_error)}")
        # Try a more detailed diagnostic
        import xml.sax
        try:
            xml.sax.parseString(decoded_text, xml.sax.ContentHandler())
            print("SAX parser could parse the XML, but xmltodict failed.")
        except xml.sax.SAXParseException as sax_error:
            print(f"SAX parser also failed: {str(sax_error)}")
            print(f"Error at line {sax_error.getLineNumber()}, column {sax_error.getColumnNumber()}")
            # Print problematic lines
            lines = decoded_text.split('\n')
            if sax_error.getLineNumber() <= len(lines):
                error_line = lines[sax_error.getLineNumber() - 1]
                print(f"Error line content: {error_line}")

        # If the file appears to be a different format, try recovery measures
        if decoded_text.strip().startswith('{'):
            print("File appears to be JSON, not XML. Attempting to recover...")
            import json
            try:
                json_data = json.loads(decoded_text)
                print("Successfully parsed as JSON. Converting to XML-like structure...")
                # Try to construct a minimal XML-like structure
                my_dict = {"Annotations": {"Annotation": []}}
                # Extract annotation-like data if possible
                # This is highly speculative and depends on your JSON structure
            except:
                print("JSON parsing also failed.")

        raise ValueError(f"Failed to parse XML: {str(xml_parse_error)}")

    # Process the parsed XML
    xyout = []

    try:
        reduced_annotations = my_dict['Annotations'].get('@MicronsPerPixel', 1)
        reduced_annotations = float(reduced_annotations)
    except:
        reduced_annotations = 1
        print("Warning: Could not find or parse MicronsPerPixel, using default value 1")

    annotations = my_dict.get("Annotations", {}).get("Annotation", [])

    if not annotations:
        print("Warning: No annotations found in the XML")
        return reduced_annotations, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

    # Handle both list and single annotation cases
    if not isinstance(annotations, list):
        annotations = [annotations]

    for layer in annotations:
        if 'Region' in layer.get("Regions", {}):
            try:
                layer_id = int(layer.get('@Id'))
                regions = layer["Regions"]["Region"]

                # Handle both list and single region cases
                if isinstance(regions, list):
                    for annotation in regions:
                        try:
                            annotation_number = float(annotation["@Id"])
                            vertices = annotation.get("Vertices", {}).get("Vertex", [])

                            # Handle both list and single vertex cases
                            if not isinstance(vertices, list):
                                vertices = [vertices]

                            for vertex in vertices:
                                try:
                                    x = float(vertex.get('@X'))
                                    y = float(vertex.get('@Y'))
                                    xyout.append([layer_id, annotation_number, x, y])
                                except Exception as vertex_error:
                                    print(f"Error processing vertex: {str(vertex_error)}")
                        except Exception as annotation_error:
                            print(f"Error processing annotation: {str(annotation_error)}")
                elif isinstance(regions, dict):
                    try:
                        annotation_number = float(regions.get("@Id"))
                        vertices = regions.get("Vertices", {}).get("Vertex", [])

                        # Handle both list and single vertex cases
                        if not isinstance(vertices, list):
                            vertices = [vertices]

                        for vertex in vertices:
                            try:
                                x = float(vertex.get('@X'))
                                y = float(vertex.get('@Y'))
                                xyout.append([layer_id, annotation_number, x, y])
                            except Exception as vertex_error:
                                print(f"Error processing vertex: {str(vertex_error)}")
                    except Exception as region_error:
                        print(f"Error processing region: {str(region_error)}")
            except Exception as layer_error:
                print(f"Error processing layer: {str(layer_error)}")

    if not xyout:
        print("Warning: No valid annotation coordinates were extracted")
    else:
        print(f"Successfully extracted {len(xyout)} annotation coordinates")

    xyout_df = pd.DataFrame(xyout, columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])
    return reduced_annotations, xyout_df