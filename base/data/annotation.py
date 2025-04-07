"""
Annotation Handling Utilities for CODAvision

This module provides functions for loading, processing, and handling annotations
from various formats (primarily XML). It includes utilities for:
- Loading and parsing annotation data from XML files
- Creating annotation masks for segmentation tasks
- Extracting bounding boxes from annotations
- Working with annotation coordinates and layers

Authors:
    Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
    Tyler Newton (JHU - DSAI)
    Jaime Gomez (Johns Hopkins - Wirtz/Kiemen Lab)

Updated: March 2025
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import codecs
import xml.sax
import xml.etree.ElementTree as ET
from collections import defaultdict


# XML Parsing and Loading Utilities
# ---------------------------------

def load_annotation_data(model_path: str, annotation_path: str, image_path: str, class_check: int = 0,
                         test: bool = False) -> Tuple:
    """
    Loads annotation data from XML files, creates tissue masks and bounding boxes.

    Args:
        model_path: Path to the directory for saving model data
        annotation_path: Path to the directory containing XML annotation files
        image_path: Path to the directory containing image files
        test (bool, optional): Used in calculate_tissue_mask.

    Returns:
        Tuple of (bounding box file list, annotation counts array, create_new_tiles flag)
    """
    print('\nImporting annotation data...')

    # Load model metadata
    with open(os.path.join(model_path, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        WS = data['WS']
        umpix = data['umpix']
        cmap = data['cmap']
        nm = data['nm']
        nwhite = data['nwhite']
        scale = None
        if umpix == 'TBD':
            scale = float(data['scale'])

    # Prepare color map and get class count
    cmap2 = np.vstack(([0, 0, 0], cmap)) / 255
    numclass = np.max(WS[2])

    # Find XML files
    imlist = [f for f in os.listdir(annotation_path) if f.endswith('.xml')]
    if not imlist:
        raise ValueError(
            'No annotation files (.xml) found in the specified directory. '
            'Please ensure that annotation files exist in the directory: ' + annotation_path
        )

    # Initialize containers for results
    numann0 = []
    ctlist0 = {'tile_name': [], 'tile_pth': []}
    outim = os.path.join(annotation_path, 'check_annotations')
    os.makedirs(outim, exist_ok=True)

    # Check if all XML files have corresponding image files
    for imnm in imlist:
        base_name = imnm[:-4]
        tif_file = os.path.join(image_path, f'{base_name}.tif')
        jpg_file = os.path.join(image_path, f'{base_name}.jpg')
        png_file = os.path.join(annotation_path, f'{base_name}.png')
        if not os.path.isfile(tif_file) and not os.path.isfile(jpg_file) and not os.path.isfile(png_file):
            raise FileNotFoundError(f'Cannot find a tif, png or jpg file for xml file: {imnm}')

    create_new_tiles = False

    # Process each XML file
    for idx, imnm in enumerate(imlist, start=1):
        base_name = imnm[:-4]
        print(f'Image {idx} of {len(imlist)}: {base_name}')
        outpth = os.path.join(annotation_path, 'data py', base_name)
        annotations_file = os.path.join(outpth, 'annotations.pkl')

        # Check if model parameters have changed
        reload_xml = check_if_model_parameters_changed(annotations_file, WS, umpix, nwhite, image_path, base_name)

        # Check if annotations were already processed
        if os.path.isfile(annotations_file):
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                dm, bb = data.get('dm', ''), data.get('bb', 0)
        else:
            dm, bb = '', 0

        modification_time = os.path.getmtime(os.path.join(annotation_path, imnm))
        date_modified = modification_time

        # Skip if already processed and parameters haven't changed
        if str(dm)== str(date_modified) and bb == 1 and reload_xml == 0:
            print(' annotation data previously loaded')
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                numann, ctlist = data.get('numann', []), data.get('ctlist', [])
            numann0.extend(numann)
            ctlist0['tile_name'].extend(ctlist['tile_name'])
            ctlist0['tile_pth'].extend(ctlist['tile_pth'])
            # print(numann0)
            # print(ctlist0)
            continue

        create_new_tiles = True

        # Create output directory
        if os.path.isdir(outpth):
            import shutil
            shutil.rmtree(outpth)
        os.makedirs(outpth)

        # Import XML annotations
        import_xml(annotations_file, os.path.join(annotation_path, imnm), date_modified)

        if os.path.exists(annotations_file):
            # Update model parameters
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                data['WS'] = WS
                data['umpix'] = umpix
                data['nwhite'] = nwhite
                data['pthim'] = image_path
            with open(annotations_file, 'wb') as f:
                pickle.dump(data, f)

            # Load tissue mask
            I0, TA, _ = calculate_tissue_mask(image_path, base_name, test)

            # Save annotation mask
            if scale:
                J0 = save_annotation_mask(I0, outpth, WS, umpix, TA, 1, scale)
            else:
                J0 = save_annotation_mask(I0, outpth, WS, umpix, TA, 1)

            # Save visualization of annotations
            from skimage import io
            io.imsave(os.path.join(outpth, 'view_annotations.png'), J0.astype(np.uint8))

            # Create visualization with color overlay
            I = I0[::2, ::2, :].astype(np.float64) / 255
            J = J0[::2, ::2].astype(int)
            J1 = cmap2[J, 0]
            J1 = J1.reshape(J.shape)
            J2 = cmap2[J, 1]
            J2 = J2.reshape(J.shape)
            J3 = cmap2[J, 2]
            J3 = J3.reshape(J.shape)
            mask = np.dstack((J1, J2, J3))
            I = (I * 0.5) + (mask * 0.5)
            if os.path.isfile(os.path.join(outim, f'{base_name}.png')):
                os.remove(os.path.join(outim, f'{base_name}.png'))
            io.imsave(os.path.join(outim, f'{base_name}.png'), (I * 255).astype(np.uint8))

            # Save bounding boxes
            numann, ctlist = save_bounding_boxes(I0, outpth, nm, numclass)
            numann0.extend(numann)
            ctlist0['tile_name'].extend(ctlist['tile_name'])
            ctlist0['tile_pth'].extend(ctlist['tile_pth'])
            # print(numann0)
            # print(ctlist0)
            # print(" ")

    # Check if any annotations were found
    if not numann0:
        raise ValueError(
            'No valid annotations were found in the XML files. '
            'Please check that your annotation files contain valid annotations.'
        )

    return ctlist0, numann0, create_new_tiles


def import_xml(annotations_file: str, xml_file: str, date_modified: str = None, reduced_annotations: float = None) -> Tuple:
    """
    Reads an XML file and imports annotation data, saving it to a pickle file.

    Args:
        annotations_file: The file path for the output pickle file.
        xml_file: The file path for the input XML file.
        date_modified: String indicating date and time the xml file was modified.
        reduced_annotations: The reduced annotations value (default: None).

    Returns:
        Tuple containing the annotation DataFrame and reduced_annotations value.
    """
    if reduced_annotations is None:
        reduced_annotations = 0
    if date_modified is None:
        date_modified = []

    print(' 1. of 4. Importing annotation data from xml file')
    
    try:
        reduced_annotations, xyout_df = load_annotations(xml_file)
    except Exception as e:
        print(f'  Error reading XML file: {str(e)}')
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), 0

    reduced_annotations = float(reduced_annotations)
    if not xyout_df.empty:
        if reduced_annotations == 1:
            reduced_annotations = reduced_annotations
        xyout_df.iloc[:, 2:4] = xyout_df.iloc[:, 2:4] * reduced_annotations

        # Create directory if needed
        annotations_dir = os.path.dirname(annotations_file)
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)

        # Update or create pickle file
        if os.path.exists(annotations_file):
            print('File already exists, updating data...')
            with open(annotations_file, 'rb') as f:
                try:
                    existing_data = pickle.load(f)
                except EOFError:
                    existing_data = {}

            existing_data.update({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': date_modified})
            with open(annotations_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            print(' Creating file...')
            with open(annotations_file, 'wb') as f:
                pickle.dump({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': date_modified}, f)
    else:
        # Delete empty subfolder
        outpth = os.path.dirname(annotations_file)
        print(f' WARNING: No annotations found in {xml_file}, deleting the subfolder {outpth}')
        if os.path.isdir(outpth):
            import shutil
            shutil.rmtree(outpth)

    return xyout_df, reduced_annotations


def load_annotations(xml_file: str) -> Tuple[float, pd.DataFrame]:
    """
    Load annotation coordinates from an XML file into a DataFrame.

    Args:
        xml_file: The path to the XML file containing annotations.

    Returns:
        Tuple of:
        - reduced_annotations: The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise 1.0.
        - xyout_df: DataFrame with annotation labels and coordinates in format:
          'Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'.
    """
    # Use XML handler to load annotation data
    return load_xml_annotations(xml_file)


def extract_annotation_layers(xml_path: str, debug: bool = True) -> pd.DataFrame:
    """
    Extract annotation layers from an XML file for the GUI.

    Args:
        xml_path: Path to the XML file
        debug: Whether to print debug information

    Returns:
        DataFrame with layer names and colors
    """
    file_name = os.path.basename(xml_path)

    # Skip hidden files
    if file_name.startswith('_'):
        print(f"Skipping hidden file: {file_name}")
        return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])
    
    print(f"Extracting annotation layers from: {file_name}")

    # Try to extract layers from original file
    try:
        xml_content, encoding = read_xml_file(xml_path, debug)
        
        # Parse with standard methods or fallback to manual extraction
        xml_dict = parse_xml(xml_content, debug, file_name)
        if not xml_dict:
            xml_dict = manual_extract_annotations(xml_content, debug, file_name)

        if xml_dict:
            annotations = xml_dict.get("Annotations", {}).get("Annotation", [])
            
            if not annotations:
                return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])
                
            if not isinstance(annotations, list):
                annotations = [annotations]

            data = []
            for layer in annotations:
                try:
                    layer_name = layer.get('@Name', '')
                    color = layer.get('@LineColor', '0')
                    rgb = rgb_from_linecolor(color)
                    data.append({
                        'Layer Name': layer_name.replace(" ", "_"),
                        'Color': rgb,
                        'Whitespace Settings': None
                    })
                except Exception as e:
                    if debug:
                        print(f"  {file_name}: Error processing layer: {str(e)}")

            if data:
                return pd.DataFrame(data)

        # Try with regex if regular parsing fails
        annotation_pattern = re.compile(r'<Annotation[^>]*?Id="(\d+)"[^>]*?Name="([^"]*)"[^>]*?LineColor="([^"]*)"')
        annotations = annotation_pattern.findall(xml_content)

        if annotations:
            data = []
            for _, name, color in annotations:
                try:
                    rgb = rgb_from_linecolor(color)
                    data.append({
                        'Layer Name': name.replace(" ", "_"),
                        'Color': rgb,
                        'Whitespace Settings': None
                    })
                except Exception as e:
                    if debug:
                        print(f"  {file_name}: Error processing color value: {str(e)}")

            if data:
                return pd.DataFrame(data)
    except Exception as e:
        if debug:
            print(f"  {file_name}: Error extracting from file: {str(e)}")

    print(f"  {file_name}: All extraction methods failed. No annotation layers found.")
    return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])


def read_xml_file(xml_path: str, debug: bool = True) -> Tuple[str, str]:
    """
    Comprehensive XML file reader that handles different encodings and format issues.

    Args:
        xml_path: Path to the XML file
        debug: Whether to print debug information

    Returns:
        Tuple containing (clean_xml_content, detected_encoding)
    """
    file_name = os.path.basename(xml_path)
    if debug:
        print(f"Reading XML file: {file_name}")

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    with open(xml_path, 'rb') as binary_file:
        raw_data = binary_file.read()

    # Check for BOM markers
    bom_encodings = {
        codecs.BOM_UTF8: 'utf-8',
        codecs.BOM_UTF16_LE: 'utf-16-le',
        codecs.BOM_UTF16_BE: 'utf-16-be',
        codecs.BOM_UTF32_LE: 'utf-32-le',
        codecs.BOM_UTF32_BE: 'utf-32-be',
    }

    for bom, encoding in bom_encodings.items():
        if raw_data.startswith(bom):
            if debug:
                print(f"  {file_name}: Found BOM for encoding: {encoding}")
            raw_data = raw_data[len(bom):]
            try:
                decoded_text = raw_data.decode(encoding)
                if debug:
                    print(f"  {file_name}: Successfully decoded with BOM-detected encoding: {encoding}")
                return clean_xml_content(decoded_text, debug), encoding
            except UnicodeDecodeError:
                if debug:
                    print(f"  {file_name}: Failed to decode with BOM-detected encoding: {encoding}")
                pass

    # Try different encodings
    str_content_latin1 = raw_data.decode('latin-1', errors='replace')

    # Check for Mac OS X metadata
    if "Mac OS X" in str_content_latin1[:100]:
        if debug:
            print(f"  {file_name}: Detected Mac OS X metadata")
        clean_content, encoding = find_xml_after_mac_metadata(str_content_latin1, debug, file_name)
        if clean_content:
            return clean_content, encoding

    # Try common encodings
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252', 'mac-roman']
    for encoding in encodings_to_try:
        try:
            decoded_text = raw_data.decode(encoding)
            clean_content = clean_xml_content(decoded_text, debug, file_name)
            if clean_content.strip().startswith('<?xml') or clean_content.strip().startswith('<Annotations'):
                if debug:
                    print(f"  {file_name}: Successfully decoded with {encoding}")
                return clean_content, encoding
        except UnicodeDecodeError:
            if debug:
                print(f"  {file_name}: Failed to decode with {encoding}")

    # Fallback to latin-1 with error replacement
    decoded_text = raw_data.decode('latin-1', errors='replace')
    if debug:
        print(f"  {file_name}: Falling back to latin-1 with error replacement")

    clean_content = clean_xml_content(decoded_text, debug, file_name)
    return clean_content, 'latin-1'


def find_xml_after_mac_metadata(content: str, debug: bool = True, file_name: str = "unknown") -> Tuple[str, str]:
    """
    Find XML content after Mac OS X metadata.

    Args:
        content: The string content to search
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Tuple of (clean_content, encoding)
    """
    # Look for XML markers
    xml_markers = ['<?xml', '<Annotations', '<annotation', '<ANNOTATIONS']
    start_index = -1

    for marker in xml_markers:
        pos = content.find(marker)
        if pos != -1:
            start_index = pos
            break

    if start_index != -1:
        clean_content = content[start_index:]
        if debug:
            print(f"  {file_name}: Found XML marker at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-after-mac-metadata'

    # Try other common markers
    markers = [
        "[ZoneTransfer]",
        "<?xml",
        "<Annotations"
    ]

    for marker in markers:
        pos = content.find(marker)
        if pos != -1:
            if marker == "[ZoneTransfer]":
                next_line_pos = content.find('\n', pos)
                if next_line_pos != -1:
                    clean_content = content[next_line_pos+1:]
                    if debug:
                        print(f"  {file_name}: Skipped to after [ZoneTransfer]. First 50 chars: {clean_content[:50]}")
                    return clean_content, 'detected-after-zonetransfer'
            else:
                clean_content = content[pos:]
                if debug:
                    print(f"  {file_name}: Found marker {marker} at position {pos}. First 50 chars: {clean_content[:50]}")
                return clean_content, 'detected-after-marker'

    # Try to find any XML-like tag
    xml_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>')
    match = xml_pattern.search(content)
    if match:
        start_index = match.start()
        clean_content = content[start_index:]
        if debug:
            print(f"  {file_name}: Found XML-like pattern at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-xml-pattern'

    # No XML content found
    if debug:
        print(f"  {file_name}: WARNING: Could not identify XML content in the file")
    return content, 'original-unmodified'


def clean_xml_content(content: str, debug: bool = True, file_name: str = "unknown") -> str:
    """
    Clean the XML content by removing non-XML data and ensuring it's well-formed.

    Args:
        content: The string content to clean
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Cleaned XML content
    """
    # Already starts with XML declaration or Annotations tag
    if content.strip().startswith('<?xml') or content.strip().startswith('<Annotations'):
        return content

    # Find XML markers
    xml_markers = ['<?xml', '<Annotations', '<annotation', '<ANNOTATIONS']
    start_index = -1

    for marker in xml_markers:
        pos = content.find(marker)
        if pos != -1:
            start_index = pos
            if debug:
                print(f"  {file_name}: Clean operation - Found XML marker '{marker}' at position {pos}")
            break

    if start_index != -1:
        return content[start_index:]

    # Try to find any XML-like tag
    xml_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>')
    match = xml_pattern.search(content)
    if match:
        if debug:
            print(f"  {file_name}: Clean operation - Found XML-like tag <{match.group(1)}> at position {match.start()}")
        return content[match.start():]

    return content


def parse_xml(xml_content: str, debug: bool = True, file_name: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Parse XML content into a dictionary using multiple parsing strategies.

    Args:
        xml_content: XML content as string
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Parsed XML as dictionary or None if parsing fails
    """
    # Try using xmltodict
    try:
        import xmltodict
        result = xmltodict.parse(xml_content)
        if debug:
            print(f"  {file_name}: Successfully parsed with xmltodict")
        return result
    except Exception as e:
        if debug:
            print(f"  {file_name}: xmltodict parsing error: {str(e)}")

    # Try using ElementTree
    try:
        xml_io = io.StringIO(xml_content)
        tree = ET.parse(xml_io)
        root = tree.getroot()

        if debug:
            print(f"  {file_name}: Successfully parsed with ElementTree")

        # Convert ElementTree to dictionary
        def etree_to_dict(t):
            d = {t.tag: {}}
            children = list(t)
            if children:
                dd = defaultdict(list)
                for dc in map(etree_to_dict, children):
                    for k, v in dc.items():
                        dd[k].append(v)
                d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
            if t.attrib:
                d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
            if t.text:
                text = t.text.strip()
                if children or t.attrib:
                    if text:
                        d[t.tag]['#text'] = text
                else:
                    d[t.tag] = text
            return d

        return etree_to_dict(root)
    except Exception as et_error:
        if debug:
            print(f"  {file_name}: ElementTree parsing error: {str(et_error)}")

    # Try using SAX parser
    try:
        xml.sax.parseString(xml_content, xml.sax.ContentHandler())
        if debug:
            print(f"  {file_name}: SAX parser could parse the XML, but xmltodict and ElementTree failed.")

        # Try manual extraction
        return manual_extract_annotations(xml_content, debug, file_name)
    except xml.sax.SAXParseException as sax_error:
        if debug:
            print(f"  {file_name}: SAX parser failed: {str(sax_error)}")
            print(f"  {file_name}: Error at line {sax_error.getLineNumber()}, column {sax_error.getColumnNumber()}")

            # Show error line
            lines = xml_content.split('\n')
            if sax_error.getLineNumber() <= len(lines):
                error_line = lines[sax_error.getLineNumber() - 1]
                print(f"  {file_name}: Error line content: {error_line}")

                # Try to fix errors
                fixed_content = fix_common_xml_errors(xml_content, sax_error.getLineNumber(), debug, file_name)
                if fixed_content != xml_content:
                    try:
                        import xmltodict
                        result = xmltodict.parse(fixed_content)
                        if debug:
                            print(f"  {file_name}: Successfully parsed with xmltodict after fixing XML errors")
                        return result
                    except Exception as retry_error:
                        if debug:
                            print(f"  {file_name}: Failed to parse fixed XML: {str(retry_error)}")

    # All parsing methods failed
    if debug:
        print(f"  {file_name}: All parsing methods failed")
    return None


def fix_common_xml_errors(xml_content: str, error_line_number: int, debug: bool = True, file_name: str = "unknown") -> str:
    """
    Fix common XML errors in the content, focusing on the problematic line.

    Args:
        xml_content: Original XML content
        error_line_number: Line number where the error was detected
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Fixed XML content or original if no fixes applied
    """
    lines = xml_content.split('\n')
    if error_line_number <= 0 or error_line_number > len(lines):
        return xml_content

    # Get the problematic line
    line = lines[error_line_number - 1]
    original_line = line

    # Fix unclosed tags
    unclosed_tag_match = re.search(r'<([a-zA-Z][a-zA-Z0-9_:-]*)[^>]*$', line)
    if unclosed_tag_match:
        tag_name = unclosed_tag_match.group(1)
        line = line + ">"
        if debug:
            print(f"  {file_name}: Fixed unclosed tag <{tag_name}> on line {error_line_number}")

    # Fix unquoted attributes
    attr_no_quotes = re.findall(r'(\w+)=([^"\'][^\s>]*)', line)
    for attr, value in attr_no_quotes:
        line = line.replace(f"{attr}={value}", f'{attr}="{value}"')
        if debug:
            print(f"  {file_name}: Fixed unquoted attribute {attr}={value} on line {error_line_number}")

    # Fix unescaped entities within text
    for char, entity in [('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'), ('"', '&quot;'), ("'", '&apos;')]:
        in_tag = False
        in_attr = False
        new_line = ""
        i = 0
        while i < len(line):
            if line[i:i+1] == '<':
                in_tag = True
                new_line += '<'
            elif line[i:i+1] == '>':
                in_tag = False
                new_line += '>'
            elif in_tag and line[i:i+1] in ['"', "'"]:
                in_attr = not in_attr
                new_line += line[i:i+1]
            elif (not in_tag or in_attr) and line[i:i+1] == char and char != '<' and char != '>':
                new_line += entity
                if debug:
                    print(f"  {file_name}: Escaped {char} to {entity} on line {error_line_number}")
            else:
                new_line += line[i:i+1]
            i += 1
        line = new_line

    # Replace the line if changes were made
    if line != original_line:
        lines[error_line_number - 1] = line
        return '\n'.join(lines)

    return xml_content


def manual_extract_annotations(xml_content: str, debug: bool = True, file_name: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Manually extract annotation data using regex when XML parsing fails.

    Args:
        xml_content: XML content as string
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Dictionary with extracted annotations or None if extraction fails
    """
    try:
        # Find Annotations element
        annotations_match = re.search(r'<Annotations.*?>(.*?)</Annotations>',
                                     xml_content, re.DOTALL)
        if annotations_match:
            if debug:
                print(f"  {file_name}: Found Annotations element. Attempting to extract data...")

            # Find all Annotation tags
            annotation_pattern = re.compile(r'<Annotation[^>]*?Id="(\d+)"[^>]*?Name="([^"]*)"[^>]*?LineColor="([^"]*)"')
            annotations = annotation_pattern.findall(xml_content)

            if annotations:
                if debug:
                    print(f"  {file_name}: Manually extracted {len(annotations)} annotations")

                # Build result dictionary
                result = {"Annotations": {"Annotation": []}}

                for anno_id, name, color in annotations:
                    anno_dict = {
                        "@Id": anno_id,
                        "@Name": name,
                        "@LineColor": color,
                        "Regions": {"Region": []}
                    }

                    # Find this annotation's content
                    anno_start = xml_content.find(f'<Annotation Id="{anno_id}"')
                    if anno_start != -1:
                        anno_end = xml_content.find('</Annotation>', anno_start)
                        if anno_end != -1:
                            anno_text = xml_content[anno_start:anno_end]

                            # Find all regions
                            region_pattern = re.compile(r'<Region[^>]*?Id="([^"]*)"')
                            for region_match in region_pattern.finditer(anno_text):
                                region_id = region_match.group(1)
                                region_dict = {"@Id": region_id, "Vertices": {"Vertex": []}}

                                # Find this region's content
                                region_start = anno_text.find(f'<Region Id="{region_id}"', region_match.start())
                                if region_start != -1:
                                    region_end = anno_text.find('</Region>', region_start)
                                    if region_end != -1:
                                        region_text = anno_text[region_start:region_end]

                                        # Find all vertices
                                        vertex_pattern = re.compile(r'<Vertex[^>]*?X="([^"]*)"[^>]*?Y="([^"]*)"')
                                        for vertex_match in vertex_pattern.finditer(region_text):
                                            x, y = vertex_match.groups()
                                            region_dict["Vertices"]["Vertex"].append({"@X": x, "@Y": y})

                                if region_dict["Vertices"]["Vertex"]:
                                    anno_dict["Regions"]["Region"].append(region_dict)

                    result["Annotations"]["Annotation"].append(anno_dict)

                return result
    except Exception as e:
        if debug:
            print(f"  {file_name}: Error in manual extraction: {str(e)}")

    return None


def rgb_from_linecolor(color_str: str) -> Tuple[int, int, int]:
    """
    Convert a LineColor attribute value to RGB tuple.

    Args:
        color_str: String containing a color value

    Returns:
        Tuple of (R, G, B) values
    """
    try:
        hex_color = int(color_str)
        b = (hex_color & 0xFF0000) >> 16
        g = (hex_color & 0x00FF00) >> 8
        r = hex_color & 0x0000FF
        return (r, g, b)
    except (ValueError, TypeError):
        # Return default color if conversion fails
        return (0, 0, 0)


def extract_annotation_coordinates(xml_dict: Dict[str, Any], debug: bool = True, file_name: str = "unknown") -> Tuple[float, pd.DataFrame]:
    """
    Extract annotation coordinates from parsed XML dictionary.

    Args:
        xml_dict: Dictionary containing parsed XML
        debug: Whether to print debug information
        file_name: Name of the file being processed

    Returns:
        Tuple of (reduced_annotations, DataFrame with coordinates)
    """
    # Get MicronsPerPixel value
    try:
        reduced_annotations = float(xml_dict['Annotations'].get('@MicronsPerPixel', 1))
        if debug:
            print(f"  {file_name}: Found MicronsPerPixel: {reduced_annotations}")
    except:
        reduced_annotations = 1.0
        if debug:
            print(f"  {file_name}: Warning: Could not find or parse MicronsPerPixel, using default value 1")

    # Get annotations
    annotations = xml_dict.get("Annotations", {}).get("Annotation", [])

    if not annotations:
        if debug:
            print(f"  {file_name}: Warning: No annotations found in the XML")
        return reduced_annotations, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

    # Ensure annotations is a list
    if not isinstance(annotations, list):
        annotations = [annotations]

    xyout = []
    for layer in annotations:
        if 'Region' in layer.get("Regions", {}):
            try:
                layer_id = int(layer.get('@Id'))
                regions = layer["Regions"]["Region"]

                # Process regions (could be list or single region)
                if isinstance(regions, list):
                    for annotation in regions:
                        try:
                            annotation_number = float(annotation["@Id"])
                            vertices = annotation.get("Vertices", {}).get("Vertex", [])

                            # Ensure vertices is a list
                            if not isinstance(vertices, list):
                                vertices = [vertices]

                            for vertex in vertices:
                                try:
                                    x = float(vertex.get('@X'))
                                    y = float(vertex.get('@Y'))
                                    xyout.append([layer_id, annotation_number, x, y])
                                except Exception as vertex_error:
                                    if debug:
                                        print(f"  {file_name}: Error processing vertex: {str(vertex_error)}")
                        except Exception as annotation_error:
                            if debug:
                                print(f"  {file_name}: Error processing annotation: {str(annotation_error)}")
                elif isinstance(regions, dict):
                    try:
                        annotation_number = float(regions.get("@Id"))
                        vertices = regions.get("Vertices", {}).get("Vertex", [])

                        # Ensure vertices is a list
                        if not isinstance(vertices, list):
                            vertices = [vertices]

                        for vertex in vertices:
                            try:
                                x = float(vertex.get('@X'))
                                y = float(vertex.get('@Y'))
                                xyout.append([layer_id, annotation_number, x, y])
                            except Exception as vertex_error:
                                if debug:
                                    print(f"  {file_name}: Error processing vertex: {str(vertex_error)}")
                    except Exception as region_error:
                        if debug:
                            print(f"  {file_name}: Error processing region: {str(region_error)}")
            except Exception as layer_error:
                if debug:
                    print(f"  {file_name}: Error processing layer: {str(layer_error)}")

    if not xyout:
        if debug:
            print(f"  {file_name}: Warning: No valid annotation coordinates were extracted")
    else:
        if debug:
            print(f"  {file_name}: Successfully extracted {len(xyout)} annotation coordinates")

    return reduced_annotations, pd.DataFrame(xyout, columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])


def load_xml_annotations(xml_path: str, debug: bool = True) -> Tuple[float, pd.DataFrame]:
    """
    Complete function to load XML annotations from a file with robust cross-platform handling.

    Args:
        xml_path: Path to the XML file
        debug: Whether to print debug information

    Returns:
        Tuple of (reduced_annotations, DataFrame with coordinates)
    """
    file_name = os.path.basename(xml_path)

    # Skip hidden files
    if file_name.startswith('_'):
        print(f"Skipping hidden file: {file_name}")
        return 1.0, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

    print(f"Loading XML annotations from: {file_name}")

    # Try to load original file
    try:
        if debug:
            print(f"  {file_name}: Attempting to load original file without cleaning")

        # Read XML content
        xml_content, encoding = read_xml_file(xml_path, debug)

        if debug:
            print(f"  {file_name}: Read original XML content with encoding {encoding}")

        xml_dict = parse_xml(xml_content, debug, file_name)

        if xml_dict:
            if debug:
                print(f"  {file_name}: Successfully parsed original XML file")
            return extract_annotation_coordinates(xml_dict, debug, file_name)

        # Try manual extraction
        if debug:
            print(f"  {file_name}: Standard parsing failed. Attempting manual extraction...")

        manual_dict = manual_extract_annotations(xml_content, debug, file_name)
        if manual_dict:
            if debug:
                print(f"  {file_name}: Successfully extracted annotations manually from original file")
            return extract_annotation_coordinates(manual_dict, debug, file_name)

    except Exception as e:
        if debug:
            print(f"  {file_name}: Error loading original XML file: {str(e)}")
            print(f"  {file_name}: Attempting to clean and retry...")

    # Try cleaning the file
    temp_path = xml_path + ".clean.tmp"
    clean_success = clean_xml_file(xml_path, temp_path)

    if clean_success and os.path.exists(temp_path):
        try:
            # Try to load cleaned file
            if debug:
                print(f"  {file_name}: Attempting to load cleaned XML file")

            xml_content, encoding = read_xml_file(temp_path, debug)

            if debug:
                print(f"  {file_name}: Read cleaned XML file with encoding {encoding}")

            xml_dict = parse_xml(xml_content, debug, file_name)

            if xml_dict:
                if debug:
                    print(f"  {file_name}: Successfully parsed cleaned XML file")
                result = extract_annotation_coordinates(xml_dict, debug, file_name)
                # Clean up temp file
                os.remove(temp_path)
                return result

            # Try manual extraction
            if debug:
                print(f"  {file_name}: Standard parsing failed on cleaned file. Attempting manual extraction...")

            manual_dict = manual_extract_annotations(xml_content, debug, file_name)
            if manual_dict:
                if debug:
                    print(f"  {file_name}: Successfully extracted annotations manually from cleaned file")
                result = extract_annotation_coordinates(manual_dict, debug, file_name)
                # Clean up temp file
                os.remove(temp_path)
                return result

            # Clean up temp file
            os.remove(temp_path)
        except Exception as e:
            if debug:
                print(f"  {file_name}: Error processing cleaned XML file: {str(e)}")
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # All parsing methods failed
    if debug:
        print(f"  {file_name}: All parsing methods failed. Returning empty DataFrame.")
    return 1.0, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])


def clean_xml_file(input_path: str, output_path: str) -> bool:
    """
    Clean an XML file to ensure cross-platform compatibility between Mac and PC.

    Args:
        input_path: Path to the input XML file
        output_path: Path to write the cleaned XML file

    Returns:
        bool: True if cleaning succeeded, False otherwise
    """
    file_name = os.path.basename(input_path)

    # Skip hidden files
    if file_name.startswith('_'):
        print(f"Skipping hidden file: {file_name}")
        return False
    
    print(f"Processing XML file: {file_name}")

    try:
        # Read file content
        with open(input_path, 'rb') as f:
            content = f.read()

        # Remove BOM if present
        bom_removed = False
        if content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]
            bom_removed = True
            print(f"  {file_name}: UTF-8 BOM detected and removed")
        elif content.startswith(b'\xfe\xff'):
            try:
                content = content[2:].decode('utf-16-be').encode('utf-8')
                bom_removed = True
                print(f"  {file_name}: UTF-16 BE BOM detected and removed")
            except UnicodeError:
                pass
        elif content.startswith(b'\xff\xfe'):
            try:
                content = content[2:].decode('utf-16-le').encode('utf-8')
                bom_removed = True
                print(f"  {file_name}: UTF-16 LE BOM detected and removed")
            except UnicodeError:
                pass

        # Detect encoding and decode content
        decoded_content = None
        detected_encoding = None
        for encoding in ['utf-8', 'latin-1', 'mac-roman', 'cp1252', 'windows-1252', 'ISO-8859-1']:
            try:
                decoded_content = content.decode(encoding)
                detected_encoding = encoding
                print(f"  {file_name}: Successfully decoded using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        # Fallback to replacement
        if decoded_content is None:
            decoded_content = content.decode('utf-8', errors='replace')
            detected_encoding = 'utf-8-replace'
            print(f"  {file_name}: Falling back to UTF-8 with character replacement")

        # Normalize line endings
        normalized_content = decoded_content.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\r\n')
        print(f"  {file_name}: Line endings normalized to CRLF")

        # Remove control characters
        original_length = len(normalized_content)
        clean_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', normalized_content)
        if len(clean_content) != original_length:
            print(f"  {file_name}: Removed {original_length - len(clean_content)} control characters")

        # Find start of XML content
        xml_start = clean_content.find('<?xml')

        # Try common root elements if XML declaration not found
        if xml_start == -1:
            common_roots = ['<Annotations', '<root', '<document', '<data', '<config',
                           '<svg', '<html', '<feed', '<rss', '<xml', '<project']

            for root in common_roots:
                pos = clean_content.find(root)
                if pos != -1:
                    xml_start = pos
                    print(f"  {file_name}: XML content starts with '{root}' at position {pos}")
                    break

        # Try to find any tag if common roots not found
        if xml_start == -1:
            tag_match = re.search(r'<[a-zA-Z_][a-zA-Z0-9_:.-]*(?:\s+[^>]*)?>', clean_content)
            if tag_match:
                xml_start = tag_match.start()
                print(f"  {file_name}: Found XML tag at position {xml_start}")

        if xml_start >= 0:
            if xml_start > 0:
                print(f"  {file_name}: Skipping {xml_start} bytes of non-XML content at beginning of file")

            xml_content = clean_content[xml_start:]

            # Check for XML declaration
            has_declaration = xml_content.lstrip().startswith('<?xml')

            if has_declaration:
                # Extract declaration and body
                decl_end = xml_content.find('?>') + 2
                xml_decl = xml_content[:decl_end]
                xml_body = xml_content[decl_end:]

                # Update or add encoding in declaration
                if 'encoding=' in xml_decl:
                    xml_decl = re.sub(r'encoding="[^"]*"', 'encoding="utf-8"', xml_decl)
                    xml_decl = re.sub(r"encoding='[^']*'", "encoding='utf-8'", xml_decl)
                    print(f"  {file_name}: Updated XML declaration encoding to UTF-8")
                else:
                    # Add encoding attribute
                    xml_decl = xml_decl.replace('?>', ' encoding="utf-8"?>')
                    print(f"  {file_name}: Added UTF-8 encoding to XML declaration")
            else:
                # Add XML declaration
                xml_decl = '<?xml version="1.0" encoding="utf-8"?>\r\n'
                xml_body = xml_content
                print(f"  {file_name}: Added XML declaration with UTF-8 encoding")

            # Remove duplicate declarations
            if '<?xml' in xml_body and xml_body.strip().startswith('<?xml'):
                second_decl_end = xml_body.find('?>') + 2
                xml_body = xml_body[second_decl_end:]
                print(f"  {file_name}: Removed duplicate XML declaration")

            # Write cleaned XML to output file
            with open(output_path, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write(xml_decl + xml_body)

            print(f"  {file_name}: Cleaned XML saved to {os.path.basename(output_path)}")
            return True

        print(f"  {file_name}: Could not identify XML content in the file")
        return False
    except Exception as e:
        print(f"  {file_name}: Error cleaning XML file: {e}")
        return False


# Annotation Mask Utilities
# --------------------------

def save_annotation_mask(image: np.ndarray, output_path: str, whitespace_settings: list, scale_factor: any, 
                       tissue_mask: np.ndarray, keep_border: int = 0, scale: Optional[float] = None) -> np.ndarray:
    """
    Creates and saves the annotation mask of an image.

    Args:
        image: The image as a numpy array
        output_path: Path where the mask will be saved
        whitespace_settings: List containing whitespace removal options, tissue order, and distribution
        scale_factor: Scaling factor for the image
        tissue_mask: Binary mask indicating tissue regions
        keep_border: Whether to keep border regions (default: 0)
        scale: Optional custom scaling factor

    Returns:
        Annotation mask as numpy array
    """
    print(' 2. of 4. Interpolating annotated regions and saving mask image')

    try:   
        with open(os.path.join(output_path, 'annotations.pkl'), 'rb') as f:
            data = pickle.load(f)
            xyout = data['xyout']
        
        if xyout.size == 0:
            return np.zeros(image.size)
        
        # Scale coordinates
        if scale is not None:
            xyout[:, 2:4] = np.round(xyout[:, 2:4]/scale)
        else:
            xyout[:, 2:4] = np.round(xyout[:, 2:4]/scale_factor)
        
        # Process tissue mask
        if tissue_mask.size > 0:
            tissue_mask = tissue_mask > 0
            tissue_mask = remove_small_objects(tissue_mask.astype(bool), min_size=30, connectivity=2)
            tissue_mask = np.logical_not(tissue_mask)
        else:
            image = image.astype(float)
            TA1 = np.std(image[:, :, [0, 1]], 2, ddof=1)
            TA2 = np.std(image[:, :, [0, 2]], 2, ddof=1)
            tissue_mask = np.max(np.concatenate((TA1, TA2), axis=2), axis=2)
            tissue_mask_binary = tissue_mask < 10
            tissue_mask_white = image[:, :, 1] > 210
            tissue_mask = tissue_mask_binary & tissue_mask_white
            tissue_mask *= 255
            labeled_mask = cv2.connectedComponents(tissue_mask.astype(np.uint8))[1]
            tissue_mask = labeled_mask >= 5
        
        # Get inverted tissue mask for formatting
        tissue_mask_inverted = tissue_mask > 0
        
        # Format annotations
        shape = tissue_mask.shape
        num_classes = len(whitespace_settings[0])
        annotation_masks = np.zeros((shape[0], shape[1], num_classes), dtype=int)

        # Temporary mask for processing
        temp_mask = np.zeros(shape, dtype=int)
        class_masks = np.zeros(shape, dtype=bool)
        
        # Process each annotation class
        for k in np.unique(xyout[:, 0]):
            if k > len(whitespace_settings[0]):
                continue
            
            # Reset temporary masks
            temp_mask.fill(0)
            class_masks.fill(False)
            
            # Extract annotations for this class
            class_annotations = xyout[xyout[:, 0] == k, :]
            region_ids = np.unique(class_annotations[:, 1])
            
            # Process each region
            for region_id in region_ids[region_ids != 0]:
                coords_idx = np.flatnonzero(class_annotations[:, 1] == region_id)
                vertices = np.vstack((
                    class_annotations[coords_idx, 2:4],
                    class_annotations[coords_idx[0], 2:4]
                ))
                
                # Calculate distances between vertices
                dists = np.sqrt(np.sum((vertices[1:, :] - vertices[:-1, :]) ** 2, axis=1))
                non_zero_dists = dists != 0
                vertices = vertices[np.concatenate(([True], non_zero_dists)), :]
                dists = dists[non_zero_dists]
                dists = np.concatenate(([0], dists))
                
                # Calculate cumulative distances
                cum_dists = np.cumsum(dists)
                interp_points = np.arange(1, np.ceil(cum_dists.max()) + 0.49, 0.49)
                
                # Interpolate points
                x_new = np.interp(interp_points, cum_dists, vertices[:, 0]).round().astype(int)
                y_new = np.interp(interp_points, cum_dists, vertices[:, 1]).round().astype(int)
                
                # Add points to mask
                try:
                    indices = np.ravel_multi_index((y_new, x_new), shape)
                    class_masks.flat[indices] = True
                except ValueError:
                    print('  annotation out of bounds')
                    continue

            # Fill holes and add to class mask
            class_masks = binary_fill_holes(class_masks)
            temp_mask[class_masks] = k
            
            # Remove border if specified
            if not keep_border:
                temp_mask[:400, :] = 0
                temp_mask[:, :400] = 0
                temp_mask[-401:, :] = 0
                temp_mask[:, -401:] = 0

            # Add to annotation masks
            class_idx = k - 1
            # print(f"  Processing class k={k}, type={type(k)}, class_idx={class_idx}, num_classes={num_classes}")
            class_idx_int = int(class_idx)  # Convert float to integer
            if 0 <= class_idx_int < num_classes:
                annotation_masks[:, :, class_idx_int] = temp_mask == k
            else:
                print(
                    f"  Warning: class_idx {class_idx_int} (k={k}) out of bounds (num_classes={num_classes}), skipping")

        # Format whitespace
        formatted_mask, indices = format_white(annotation_masks, tissue_mask_inverted, whitespace_settings, shape)
        
        # Save raw annotation mask
        with open(os.path.join(output_path, 'view_annotations_raw.png'), 'wb') as f:
            Image.fromarray(np.uint8(formatted_mask)).save(f)

        return formatted_mask

    except FileNotFoundError:
        return np.zeros(image.size)


def format_white(J0: np.ndarray, Ig: np.ndarray, WS: list, szz: tuple) -> Tuple[np.ndarray, list]:
    """
    Creates the annotation mask of the image from the annotation coordinates and nesting order.

    Args:
        J0: List containing the coordinates of the annotations of each layer
        Ig: The non-zero indexes of tissue mask
        WS: List containing whitespace settings, tissue order, etc.
        szz: Contains the dimensions of the image

    Returns:
        Tuple of:
        - J: The annotation mask of the image
        - ind: Index list
    """
    ws = WS[0]  # Whitespace removal options
    wsa0 = WS[1]  # Whitespace distribution
    wsa = wsa0[0]
    wsfat = wsa0[1] if len(wsa0) > 1 else 0
    wsnew = WS[2]  # Tissue classes
    wsorder = WS[3]  # Nesting order
    wsdelete = WS[4]  # Deleted classes

    # Initialize arrays
    Jws = np.zeros(szz, dtype=int)
    ind = []

    # Process each layer in nesting order
    for k in wsorder:
        if any(np.isin(wsdelete, k)):
            continue
        
        try:
            py_index = k - 1
            ii = J0[:, :, py_index]
        except IndexError:
            continue

        # Split into whitespace and non-whitespace regions
        iiNW = ii * (Ig == 0)
        iiW = ii * Ig
        iiW = np.flatnonzero(iiW)
        iiNW = np.flatnonzero(iiNW)

        # Apply whitespace settings
        if ws[k-1] == 0 and iiNW.size > 0:  # Remove whitespace
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = wsa
        elif ws[k-1] == 1 and iiNW.size > 0:  # Keep only whitespace
            Jws.flat[iiW] = k
            Jws.flat[iiNW] = wsfat
        elif ws[k-1] == 2 and iiNW.size > 0:  # Keep tissue and whitespace
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = k

    # Create final mask
    J = np.zeros(szz, dtype=int)
    unique_k = np.unique(Jws)
    
    for k in unique_k:
        if k == 0:
            continue
        
        tmp = Jws == k
        ii = np.flatnonzero(tmp)
        J[tmp] = wsnew[k - 1]
        
        if ii.size > 0:
            P = np.column_stack((np.full((ii.size, 2), [1, wsnew[k - 1]]), ii))
            ind.extend(P)
            
    return J, ind


def save_bounding_boxes(image: np.ndarray, output_path: str, model_name: str, num_classes: int) -> Tuple[np.ndarray, dict]:
    """
    Creates bounding box tiles of all annotations in an image and saves them as separate image files.

    Args:
        image: Input image as a 3D numpy array
        output_path: Directory to save the bounding box tiles
        model_name: Name of the model to create a subdirectory
        num_classes: Number of annotation classes

    Returns:
        Tuple of:
        - numann: Array containing the number of pixels per class per bounding box
        - ctlist: Dictionary of tile names and paths
    """
    print(' 4. of 4. Creating bounding box tiles of all annotations')

    # Set PIL decompression bomb limit higher for large histology images
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Rest of the function remains the same...
    try:
        imlabel = np.array(Image.open(os.path.join(output_path, 'view_annotations.png')))
    except:
        imlabel = np.array(Image.open(os.path.join(output_path, 'view_annotations_raw.png')))

    # Create output directories
    pthbb = os.path.join(output_path, model_name + '_boundbox')
    pthim = os.path.join(pthbb, 'im')
    pthlabel = os.path.join(pthbb, 'label')

    # Remove existing directories if needed
    if os.path.isdir(pthim):
        os.rmdir(pthim)
    if os.path.isdir(pthlabel):
        os.rmdir(pthlabel)

    os.makedirs(pthim)
    os.makedirs(pthlabel)

    # Create tissue mask
    tmp = imlabel > 0
    tmp = tmp.astype(np.uint8)
    
    # Close small gaps
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel_large)
    
    # Remove small objects
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel_small)
    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, contours, -1, 255, thickness=cv2.FILLED)
    tmp = remove_small_objects(tmp.astype(bool), min_size=300)

    # Label connected components
    L = cv2.connectedComponents(tmp.astype(np.uint8))[1]

    # Initialize array for pixel counts
    numann = np.zeros((np.max(L), num_classes), dtype=np.uint32)

    # Function to process each bounding box
    def create_bounding_box(pk):
        # Get bounding box mask
        tmp = (L == pk)
        a = np.sum(tmp, axis=1)
        b = np.sum(tmp, axis=0)
        rect = [np.nonzero(b)[0][0], np.nonzero(b)[0][-1], np.nonzero(a)[0][0], np.nonzero(a)[0][-1]]

        # Crop image and label
        tmp = tmp[rect[2]:rect[3], rect[0]:rect[1]]
        tmplabel = imlabel[rect[2]:rect[3], rect[0]:rect[1]] * tmp
        tmpim = image[rect[2]:rect[3], rect[0]:rect[1], :]

        nm = str(pk).zfill(5)
        return nm, tmpim, tmplabel

    # Process all bounding boxes in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(create_bounding_box, pk) for pk in range(1, np.max(L) + 1)]

    count = 0
    for future in futures:
        nm, tmpim, tmplabel = future.result()

        # Save image and label
        Image.fromarray(tmpim.astype(np.uint8)).save(os.path.join(pthim, f'{nm}.png'))
        Image.fromarray(tmplabel.astype(np.uint8)).save(os.path.join(pthlabel, f'{nm}.png'))

        # Count pixels per class
        for anns in range(num_classes):
            numann[count, anns] = np.sum(tmplabel == anns + 1)
        count += 1

    # Create list of tile names and paths
    ctlist = {
        'tile_name': sorted([f for f in os.listdir(pthim) if f.endswith('.png')]),
        'tile_pth': [os.path.dirname(os.path.join(pthim, f)) for f in sorted(os.listdir(pthim)) if f.endswith('.png')]
    }

    # Save data to pickle file
    bb = 1  # Flag to indicate bounding boxes are created
    annotations_file = os.path.join(output_path, 'annotations.pkl')

    if os.path.isfile(annotations_file):
        with open(annotations_file, 'rb') as f:
            data = pickle.load(f)
            data['numann'] = numann
            data['ctlist'] = ctlist
            data['bb'] = bb
        with open(annotations_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        data = {'numann': numann, 'ctlist': ctlist, 'bb': bb}
        with open(annotations_file, 'wb') as f:
            pickle.dump(data, f)
            
    return numann, ctlist


# Utility Function
# ---------------

def check_if_model_parameters_changed(datafile: str, WS: list, umpix: any, nwhite: int, pthim: str, image_name: str) -> int:
    try:
        reload_xml = 0

        # Check if file exists
        if not os.path.exists(datafile):
            reload_xml = 0
        else:
            # Load data
            with open(datafile, 'rb') as f:
                data = pickle.load(f)

            # Check if all required keys exist
            if not all(key in data for key in ['WS', 'umpix', 'nwhite', 'pthim']):
                print('WS, umpix, nwhite, or pthim are missing in the pickle file. Reload the XML file to add them.')
                reload_xml = 1
            else:
                # Check if parameters have changed
                if data['WS'] != WS:
                    print('Reload annotation data with updated WS.')
                    reload_xml = 1
                if data['umpix'] != umpix:
                    print('Reload annotation data with updated umpix.')
                    reload_xml = 1
                if data['nwhite'] != nwhite:
                    print('Reload annotation data with updated nwhite.')
                    reload_xml = 1
                if data['pthim'] != pthim:
                    print('Reload annotation data with updated pthim.')
                    reload_xml = 1

        TA_pkl = os.path.join(pthim, 'TA', 'TA_cutoff.pkl')
        if not os.path.exists(TA_pkl):
            print('Tissue mask evaluation has not been performed')
        else:
            file = os.path.join(pthim, 'TA', image_name + '.tif')
            if os.path.exists(file):
                pkl_modification_time = os.path.getmtime(TA_pkl)
                tif_modification_time = os.path.getmtime(file)
                if tif_modification_time < pkl_modification_time:
                    reload_xml = 1
                    os.remove(file)
            else:
                reload_xml = 1
        return reload_xml

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1


def calculate_tissue_mask(path: str, image_name: str, test) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Reads an image and returns it along with a binary mask of tissue areas.

    Args:
        path: Directory path where the image is located
        image_name: Name of the image file (without extension)

    Returns:
        Tuple of:
        - image: The image as a numpy array
        - tissue_mask: Binary mask where tissue areas are True
        - output_path: Path where the tissue mask is saved
    """
    # Create output directory
    output_path = os.path.join(path.rstrip(os.path.sep), 'TA')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Try to load image with different extensions
    try:
        image = cv2.imread(os.path.join(path, f'{image_name}.tif'))
        image = image[:, :, ::-1]  # Convert BGR to RGB
    except:
        try:
            image = cv2.imread(os.path.join(path, f'{image_name}.jpg'))
            image = image[:, :, ::-1]
        except:
            try:
                image = cv2.imread(os.path.join(path, f'{image_name}.jp2'))
                image = image[:, :, ::-1]
            except:
                image = cv2.imread(os.path.join(path, f'{image_name}.png'))
                image = image[:, :, ::-1]

    # Check if mask already exists
    if os.path.isfile(os.path.join(output_path, f'{image_name}.tif')):
        tissue_mask = cv2.imread(os.path.join(output_path, f'{image_name}.tif'), cv2.IMREAD_GRAYSCALE)
        print('  Existing TA loaded')
        return image, tissue_mask, output_path

    # Calculate tissue mask
    print('  Calculating TA image')
    mode = 'H&E'
    # Try to load saved threshold
    if os.path.isfile(os.path.join(output_path, 'TA_cutoff.pkl')):
        with open(os.path.join(output_path, 'TA_cutoff.pkl'), 'rb') as f:
            data = pickle.load(f)
            cutoffs_list = data['cts']
            mode = data['mode']
            average_TA = data['average_TA']
            if test:
                average_TA = True
        if average_TA:
            cutoff = 0
            for value in cutoffs_list.values():
                cutoff += value
            cutoff = cutoff / len(cutoffs_list)
        else:
             imnm = os.path.basename(image_name)
             cutoff = cutoffs_list[imnm+'.tif']
    else:
        # Use default threshold
        cutoff = 205

    if mode == 'H&E':
        tissue_mask = image[:, :, 1] < cutoff  # Threshold the image green values
    else:
        tissue_mask = image[:, :, 1] > cutoff
    kernel_size = 3
    tissue_mask = tissue_mask.astype(np.uint8)
    kernel = morphology.disk(kernel_size)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel.astype(np.uint8))
    tissue_mask = remove_small_objects(tissue_mask.astype(bool), min_size=10)

    # Save mask
    cv2.imwrite(os.path.join(output_path, f'{image_name}.tif'), tissue_mask.astype(np.uint8))

    return image, tissue_mask, output_path