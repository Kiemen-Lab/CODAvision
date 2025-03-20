"""
XML Handler Utility Module

This module provides utilities for handling XML files with various encodings,
including special handling for Mac-to-Windows transfer issues.

Authors:
    Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 20, 2025
"""

import os
import re
import codecs
import xml.sax
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict
import pandas as pd
import io


def clean_xml_file(input_path, output_path):
    """
    Comprehensively clean an XML file to ensure cross-platform compatibility between Mac and PC.

    This function addresses:
    1. Byte Order Mark (BOM) removal
    2. Line ending normalization (to Windows CRLF)
    3. Character encoding conversion to UTF-8
    4. XML declaration correction or addition
    5. Removal of problematic control characters
    6. Removal of non-XML content at the beginning of the file
    7. Detection of common XML root elements

    Args:
        input_path (str): Path to the input XML file
        output_path (str): Path to write the cleaned XML file

    Returns:
        bool: True if cleaning succeeded, False otherwise
    """
    file_name = os.path.basename(input_path)
    print(f"Processing XML file: {file_name}")

    try:
        # First, open the file in binary mode to check for BOM
        with open(input_path, 'rb') as f:
            content = f.read()

        # Check for and remove BOM
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

        # Try multiple encodings to find one that works
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

        # If all failed, fall back to utf-8 with replacement
        if decoded_content is None:
            decoded_content = content.decode('utf-8', errors='replace')
            detected_encoding = 'utf-8-replace'
            print(f"  {file_name}: Falling back to UTF-8 with character replacement")

        # Normalize line endings using a single approach
        # First convert all line endings to LF, then to CRLF
        normalized_content = decoded_content.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\r\n')
        print(f"  {file_name}: Line endings normalized to CRLF")

        # Remove control characters that can cause XML parsing issues
        original_length = len(normalized_content)
        clean_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', normalized_content)
        if len(clean_content) != original_length:
            print(f"  {file_name}: Removed {original_length - len(clean_content)} control characters")

        # Find where the actual XML content starts
        xml_start = clean_content.find('<?xml')

        # If not found, try other common root elements
        if xml_start == -1:
            common_roots = ['<Annotations', '<root', '<document', '<data', '<config',
                           '<svg', '<html', '<feed', '<rss', '<xml', '<project']

            for root in common_roots:
                pos = clean_content.find(root)
                if pos != -1:
                    xml_start = pos
                    print(f"  {file_name}: XML content starts with '{root}' at position {pos}")
                    break

        # As a last resort, try to find any XML-like tag
        if xml_start == -1:
            tag_match = re.search(r'<[a-zA-Z_][a-zA-Z0-9_:.-]*(?:\s+[^>]*)?>', clean_content)
            if tag_match:
                xml_start = tag_match.start()
                print(f"  {file_name}: Found XML tag at position {xml_start}")

        if xml_start >= 0:
            if xml_start > 0:
                print(f"  {file_name}: Skipping {xml_start} bytes of non-XML content at beginning of file")

            xml_content = clean_content[xml_start:]

            # Handle XML declaration
            has_declaration = xml_content.lstrip().startswith('<?xml')

            if has_declaration:
                # Update existing declaration
                decl_end = xml_content.find('?>') + 2
                xml_decl = xml_content[:decl_end]
                xml_body = xml_content[decl_end:]

                # Update or add encoding attribute
                if 'encoding=' in xml_decl:
                    xml_decl = re.sub(r'encoding="[^"]*"', 'encoding="utf-8"', xml_decl)
                    xml_decl = re.sub(r"encoding='[^']*'", "encoding='utf-8'", xml_decl)
                    print(f"  {file_name}: Updated XML declaration encoding to UTF-8")
                else:
                    # Add encoding attribute if missing
                    xml_decl = xml_decl.replace('?>', ' encoding="utf-8"?>')
                    print(f"  {file_name}: Added UTF-8 encoding to XML declaration")
            else:
                # Add a new declaration if missing
                xml_decl = '<?xml version="1.0" encoding="utf-8"?>\r\n'
                xml_body = xml_content
                print(f"  {file_name}: Added XML declaration with UTF-8 encoding")

            # Handle duplicated XML declarations
            if '<?xml' in xml_body and xml_body.strip().startswith('<?xml'):
                second_decl_end = xml_body.find('?>') + 2
                xml_body = xml_body[second_decl_end:]
                print(f"  {file_name}: Removed duplicate XML declaration")

            # Write the cleaned content to the output file
            with open(output_path, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write(xml_decl + xml_body)

            print(f"  {file_name}: Cleaned XML saved to {os.path.basename(output_path)}")
            return True

        print(f"  {file_name}: Could not identify XML content in the file")
        return False
    except Exception as e:
        print(f"  {file_name}: Error cleaning XML file: {e}")
        return False


def clean_xml_folder(folder_path):
    """
    Clean all XML files in a folder.

    Example:
    clean_xml_folder("/Users/tnewton3/Desktop/liver_tissue_data")
    """
    print(f"Cleaning all XML files in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, "cleaned_" + filename)
            if clean_xml_file(input_file, output_file):
                print(f"Successfully cleaned {filename}")
            else:
                print(f"Failed to clean {filename}")


def read_xml_file(xml_path: str, debug: bool = True) -> Tuple[str, str]:
    """
    Comprehensive XML file reader that handles different encodings and Mac-to-Windows transfers.

    This function combines multiple approaches to handle problematic XML files:
    1. Detects and handles BOMs (Byte Order Marks)
    2. Tries multiple encodings to find one that works
    3. Handles Mac OS X metadata prepended to files
    4. Detects and skips any non-XML content at the beginning of files

    Args:
        xml_path: Path to the XML file
        debug: Whether to print debug information

    Returns:
        Tuple containing (clean_xml_content, detected_encoding)
    """
    file_name = os.path.basename(xml_path)
    if debug:
        print(f"Reading XML file: {file_name}")

    # First check if the file exists
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Create a temporary cleaned file
    temp_cleaned_path = xml_path + ".cleaned.tmp"
    success = clean_xml_file(xml_path, temp_cleaned_path)

    if success and os.path.exists(temp_cleaned_path):
        try:
            # Read the cleaned file
            with open(temp_cleaned_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Remove the temporary file
            os.remove(temp_cleaned_path)
            if debug:
                print(f"  {file_name}: Successfully read cleaned XML file")
            return content, 'utf-8-cleaned'
        except Exception as e:
            if debug:
                print(f"  {file_name}: Error reading cleaned file: {e}")
            # Remove the temporary file if it exists
            if os.path.exists(temp_cleaned_path):
                os.remove(temp_cleaned_path)

    # Fall back to the original method if cleaning failed
    with open(xml_path, 'rb') as binary_file:
        raw_data = binary_file.read()

    if debug:
        print(f"  {file_name}: File size: {len(raw_data)} bytes")
        print(f"  {file_name}: First 20 bytes: {' '.join(f'{b:02x}' for b in raw_data[:20])}")

    # Check for BOMs
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
            # Remove the BOM
            raw_data = raw_data[len(bom):]
            # Try to decode with this encoding
            try:
                decoded_text = raw_data.decode(encoding)
                if debug:
                    print(f"  {file_name}: Successfully decoded with BOM-detected encoding: {encoding}")
                return clean_xml_content(decoded_text, debug), encoding
            except UnicodeDecodeError:
                # If it fails, continue to next approach
                if debug:
                    print(f"  {file_name}: Failed to decode with BOM-detected encoding: {encoding}")
                pass

    # If we get here, try a more aggressive approach
    # First convert to latin-1 which always works, to check for Mac metadata
    str_content_latin1 = raw_data.decode('latin-1', errors='replace')

    # Check for Mac OS X metadata
    if "Mac OS X" in str_content_latin1[:100]:
        if debug:
            print(f"  {file_name}: Detected Mac OS X metadata")

        # Try to find the XML content after the metadata
        clean_content, encoding = find_xml_after_mac_metadata(str_content_latin1, debug, file_name)
        if clean_content:
            return clean_content, encoding

    # Try multiple encodings
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252', 'mac-roman']
    for encoding in encodings_to_try:
        try:
            decoded_text = raw_data.decode(encoding)
            # Clean the content
            clean_content = clean_xml_content(decoded_text, debug, file_name)
            if clean_content.strip().startswith('<?xml') or clean_content.strip().startswith('<Annotations'):
                if debug:
                    print(f"  {file_name}: Successfully decoded with {encoding}")
                return clean_content, encoding
        except UnicodeDecodeError:
            if debug:
                print(f"  {file_name}: Failed to decode with {encoding}")

    # If all else fails, fall back to latin-1 with error replacement
    decoded_text = raw_data.decode('latin-1', errors='replace')
    if debug:
        print(f"  {file_name}: Falling back to latin-1 with error replacement")

    # Try to clean it up as best we can
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
    # Look for common XML markers
    xml_markers = ['<?xml', '<Annotations', '<annotation', '<ANNOTATIONS']
    start_index = -1

    for marker in xml_markers:
        pos = content.find(marker)
        if pos != -1:
            start_index = pos
            break

    # If found a marker, return content from that point
    if start_index != -1:
        clean_content = content[start_index:]
        if debug:
            print(f"  {file_name}: Found XML marker at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-after-mac-metadata'

    # Look for other file transfer markers
    # common in Windows/Mac transfers
    markers = [
        "[ZoneTransfer]",
        "<?xml",
        "<Annotations"
    ]

    for marker in markers:
        pos = content.find(marker)
        if pos != -1:
            # Special handling for Zone.Identifier info
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

    # Try to find any XML-like pattern
    xml_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>')
    match = xml_pattern.search(content)
    if match:
        start_index = match.start()
        clean_content = content[start_index:]
        if debug:
            print(f"  {file_name}: Found XML-like pattern at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-xml-pattern'

    # If all else fails
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
    # If it already starts with XML markers, return as is
    if content.strip().startswith('<?xml') or content.strip().startswith('<Annotations'):
        return content

    # Look for common XML markers
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

    # Try to find any XML-like pattern
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
    # First, try to use xmltodict which is our preferred method
    try:
        import xmltodict
        result = xmltodict.parse(xml_content)
        if debug:
            print(f"  {file_name}: Successfully parsed with xmltodict")
        return result
    except Exception as e:
        if debug:
            print(f"  {file_name}: xmltodict parsing error: {str(e)}")

    # If xmltodict fails, try ElementTree with careful error handling
    try:
        # Use a StringIO to avoid potential encoding issues
        xml_io = io.StringIO(xml_content)
        tree = ET.parse(xml_io)
        root = tree.getroot()

        if debug:
            print(f"  {file_name}: Successfully parsed with ElementTree")

        # Convert ET structure to dictionary (simplified version)
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

    # If all else fails, try the SAX parser just for validation
    try:
        xml.sax.parseString(xml_content, xml.sax.ContentHandler())
        if debug:
            print(f"  {file_name}: SAX parser could parse the XML, but xmltodict and ElementTree failed.")

        # If SAX works but others fail, try our manual extraction
        return manual_extract_annotations(xml_content, debug, file_name)
    except xml.sax.SAXParseException as sax_error:
        if debug:
            print(f"  {file_name}: SAX parser failed: {str(sax_error)}")
            print(f"  {file_name}: Error at line {sax_error.getLineNumber()}, column {sax_error.getColumnNumber()}")

            # Show the problematic line
            lines = xml_content.split('\n')
            if sax_error.getLineNumber() <= len(lines):
                error_line = lines[sax_error.getLineNumber() - 1]
                print(f"  {file_name}: Error line content: {error_line}")

                # Try to fix common XML errors in the problematic line
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

    # Return None if all parsing methods fail
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

    # Fix 1: Unclosed tags
    unclosed_tag_match = re.search(r'<([a-zA-Z][a-zA-Z0-9_:-]*)[^>]*$', line)
    if unclosed_tag_match:
        tag_name = unclosed_tag_match.group(1)
        line = line + ">"
        if debug:
            print(f"  {file_name}: Fixed unclosed tag <{tag_name}> on line {error_line_number}")

    # Fix 2: Missing quotes in attributes
    attr_no_quotes = re.findall(r'(\w+)=([^"\'][^\s>]*)', line)
    for attr, value in attr_no_quotes:
        line = line.replace(f"{attr}={value}", f'{attr}="{value}"')
        if debug:
            print(f"  {file_name}: Fixed unquoted attribute {attr}={value} on line {error_line_number}")

    # Fix 3: Unescaped special characters
    for char, entity in [('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'), ('"', '&quot;'), ("'", '&apos;')]:
        # Only replace within attribute values and text content, not in tags
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

    # Only update if changes were made
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
        # Try to find the Annotations element
        annotations_match = re.search(r'<Annotations.*?>(.*?)</Annotations>',
                                     xml_content, re.DOTALL)
        if annotations_match:
            if debug:
                print(f"  {file_name}: Found Annotations element. Attempting to extract data...")

            # Extract all annotation elements
            annotation_pattern = re.compile(r'<Annotation[^>]*?Id="(\d+)"[^>]*?Name="([^"]*)"[^>]*?LineColor="([^"]*)"')
            annotations = annotation_pattern.findall(xml_content)

            if annotations:
                if debug:
                    print(f"  {file_name}: Manually extracted {len(annotations)} annotations")

                # Create a dictionary structure similar to what xmltodict would produce
                result = {"Annotations": {"Annotation": []}}

                for anno_id, name, color in annotations:
                    anno_dict = {
                        "@Id": anno_id,
                        "@Name": name,
                        "@LineColor": color,
                        "Regions": {"Region": []}
                    }

                    # Extract regions for this annotation
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

                                # Extract vertices for this region
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
        # Return a default color if parsing fails
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
    # Try to get the MicronsPerPixel value
    try:
        reduced_annotations = float(xml_dict['Annotations'].get('@MicronsPerPixel', 1))
        if debug:
            print(f"  {file_name}: Found MicronsPerPixel: {reduced_annotations}")
    except:
        reduced_annotations = 1.0
        if debug:
            print(f"  {file_name}: Warning: Could not find or parse MicronsPerPixel, using default value 1")

    # Get the annotations
    annotations = xml_dict.get("Annotations", {}).get("Annotation", [])

    if not annotations:
        if debug:
            print(f"  {file_name}: Warning: No annotations found in the XML")
        return reduced_annotations, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

    # Make sure annotations is a list
    if not isinstance(annotations, list):
        annotations = [annotations]

    xyout = []
    for layer in annotations:
        if 'Region' in layer.get("Regions", {}):
            try:
                layer_id = int(layer.get('@Id'))
                regions = layer["Regions"]["Region"]

                # Handle regions as list or single item
                if isinstance(regions, list):
                    for annotation in regions:
                        try:
                            annotation_number = float(annotation["@Id"])
                            vertices = annotation.get("Vertices", {}).get("Vertex", [])

                            # Handle vertices as list or single item
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

                        # Handle vertices as list or single item
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

    This function handles all the steps:
    1. Read and clean the XML file content
    2. Parse the XML content
    3. Extract annotation coordinates
    4. Fall back to manual extraction if parsing fails

    Args:
        xml_path: Path to the XML file
        debug: Whether to print debug information

    Returns:
        Tuple of (reduced_annotations, DataFrame with coordinates)
    """
    file_name = os.path.basename(xml_path)
    print(f"Loading XML annotations from: {file_name}")

    # First, try to use the clean_xml_file function to create a temporary clean version
    temp_path = xml_path + ".clean.tmp"
    clean_success = clean_xml_file(xml_path, temp_path)

    if clean_success and os.path.exists(temp_path):
        try:
            # Try to process the cleaned file
            original_xml_path = xml_path
            xml_path = temp_path

            # Continue with normal processing but using the cleaned file
            xml_content, encoding = read_xml_file(xml_path, debug)

            if debug:
                print(f"  {file_name}: Using cleaned XML file with encoding {encoding}")

            xml_dict = parse_xml(xml_content, debug, file_name)

            if xml_dict:
                result = extract_annotation_coordinates(xml_dict, debug, file_name)
                # Cleanup temp file
                os.remove(temp_path)
                return result

            # If standard parsing failed, try manual extraction
            if debug:
                print(f"  {file_name}: Standard parsing failed. Attempting manual extraction...")

            manual_dict = manual_extract_annotations(xml_content, debug, file_name)
            if manual_dict:
                result = extract_annotation_coordinates(manual_dict, debug, file_name)
                # Cleanup temp file
                os.remove(temp_path)
                return result

            # Cleanup temp file
            os.remove(temp_path)

        except Exception as e:
            if debug:
                print(f"  {file_name}: Error processing cleaned XML file: {str(e)}")
            # Cleanup temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # If the cleaned approach failed or wasn't possible, try the original approach
    try:
        xml_content, encoding = read_xml_file(xml_path, debug)

        if debug:
            print(f"  {file_name}: Cleaned XML content with encoding {encoding}. Preview: {xml_content[:100]}")

        xml_dict = parse_xml(xml_content, debug, file_name)

        if xml_dict:
            return extract_annotation_coordinates(xml_dict, debug, file_name)

        if debug:
            print(f"  {file_name}: Standard parsing failed. Attempting manual extraction...")

        manual_dict = manual_extract_annotations(xml_content, debug, file_name)
        if manual_dict:
            return extract_annotation_coordinates(manual_dict, debug, file_name)

        if debug:
            print(f"  {file_name}: All parsing methods failed. Returning empty DataFrame.")
        return 1.0, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

    except Exception as e:
        if debug:
            print(f"  {file_name}: Error loading XML annotations: {str(e)}")
            import traceback
            print(traceback.format_exc())

        return 1.0, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])


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
    print(f"Extracting annotation layers from: {file_name}")

    try:
        # First try reading and cleaning the XML content
        xml_content, encoding = read_xml_file(xml_path, debug)

        if debug:
            print(f"  {file_name}: Cleaned XML content with encoding {encoding}. Preview: {xml_content[:100]}")

        # Try parsing with standard methods
        xml_dict = parse_xml(xml_content, debug, file_name)

        # If standard parsing fails, try manual extraction
        if not xml_dict:
            if debug:
                print(f"  {file_name}: Standard parsing failed. Attempting manual extraction...")
            xml_dict = manual_extract_annotations(xml_content, debug, file_name)

        # Extract layer information from the dictionary
        if xml_dict:
            annotations = xml_dict.get("Annotations", {}).get("Annotation", [])

            if not annotations:
                if debug:
                    print(f"  {file_name}: Warning: No annotations found in the XML")
                return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])

            # Make sure annotations is a list
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
                if debug:
                    print(f"  {file_name}: Successfully extracted {len(data)} annotation layers")
                return pd.DataFrame(data)

        # If the structured approach fails, try direct regex matching
        if debug:
            print(f"  {file_name}: Standard extraction failed. Attempting direct regex matching...")

        # Look for annotation patterns directly in the XML
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
                if debug:
                    print(f"  {file_name}: Successfully extracted {len(data)} annotation layers using regex")
                return pd.DataFrame(data)

        # If all methods fail
        print(f"  {file_name}: All extraction methods failed. No annotation layers found.")
        return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])

    except Exception as e:
        if debug:
            print(f"  {file_name}: Error extracting annotation layers: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])