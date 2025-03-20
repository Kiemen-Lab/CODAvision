"""
XML Handler Utility Module

This module provides utilities for handling XML files with various encodings,
including special handling for Mac-to-Windows transfer issues.

Authors:
    Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 19, 2025
"""

import os
import re
import codecs
import xml.sax
from typing import Tuple, Dict, Any, Optional
import pandas as pd


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
    try:
        # Read file as binary
        with open(input_path, 'rb') as f:
            content = f.read()

        # 1. Handle BOMs (Byte Order Marks)
        if content.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
            content = content[3:]
        elif content.startswith(b'\xfe\xff'):  # UTF-16 BE BOM
            try:
                content = content[2:].decode('utf-16-be').encode('utf-8')
            except UnicodeError:
                pass
        elif content.startswith(b'\xff\xfe'):  # UTF-16 LE BOM
            try:
                content = content[2:].decode('utf-16-le').encode('utf-8')
            except UnicodeError:
                pass

        # 2. Try to decode with various encodings
        for encoding in ['utf-8', 'latin-1', 'mac-roman', 'cp1252']:
            try:
                decoded_content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # If all encodings failed, use replacement mode
            decoded_content = content.decode('utf-8', errors='replace')

        # 3. Normalize line endings (convert to Windows CRLF)
        normalized_content = decoded_content.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\r\n')

        # 4. Remove problematic control characters (except tabs, CR, LF)
        clean_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', normalized_content)

        # 5. Find the XML content start
        # First look for XML declaration
        xml_start = clean_content.find('<?xml')

        # If no XML declaration, look for common root elements
        if xml_start == -1:
            # Common XML root elements to check
            common_roots = ['<Annotations', '<root', '<document', '<data', '<config',
                            '<svg', '<html', '<feed', '<rss', '<xml', '<project']

            for root in common_roots:
                pos = clean_content.find(root)
                if pos != -1:
                    xml_start = pos
                    break

        # If still not found, look for any XML tag
        if xml_start == -1:
            tag_match = re.search(r'<[a-zA-Z_][a-zA-Z0-9_:.-]*(?:\s+[^>]*)?>', clean_content)
            if tag_match:
                xml_start = tag_match.start()

        if xml_start >= 0:
            xml_content = clean_content[xml_start:]

            # 6. Fix or add XML declaration
            has_declaration = xml_content.lstrip().startswith('<?xml')

            if has_declaration:
                # Fix existing declaration
                decl_end = xml_content.find('?>') + 2
                xml_decl = xml_content[:decl_end]
                xml_body = xml_content[decl_end:]

                # Ensure UTF-8 encoding in declaration
                if 'encoding=' in xml_decl:
                    xml_decl = re.sub(r'encoding="[^"]*"', 'encoding="utf-8"', xml_decl)
                    xml_decl = re.sub(r"encoding='[^']*'", "encoding='utf-8'", xml_decl)
                else:
                    # Add encoding attribute if missing
                    xml_decl = xml_decl.replace('?>', ' encoding="utf-8"?>')
            else:
                # Add standard XML declaration
                xml_decl = '<?xml version="1.0" encoding="utf-8"?>\r\n'
                xml_body = xml_content

            # 7. Remove duplicate XML declarations (can happen in some cases)
            if '<?xml' in xml_body and xml_body.strip().startswith('<?xml'):
                second_decl_end = xml_body.find('?>') + 2
                xml_body = xml_body[second_decl_end:]

            # 8. Write cleaned XML with UTF-8 encoding and Windows line endings
            with open(output_path, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write(xml_decl + xml_body)

            return True

        return False
    except Exception as e:
        print(f"Error cleaning XML file: {e}")
        return False


def clean_xml_folder(folder_path):
    """
    Clean all XML files in a folder.

    # Usage example
    clean_xml_folder("/Users/tnewton3/Desktop/liver_tissue_data")
    """
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
    # First check if the file exists
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Read the raw bytes
    with open(xml_path, 'rb') as binary_file:
        raw_data = binary_file.read()

    # Debug: Print the first few bytes to examine content
    if debug:
        print(f"File size: {len(raw_data)} bytes")
        print(f"First 20 bytes: {' '.join(f'{b:02x}' for b in raw_data[:20])}")

    # Check for known BOMs and remove them
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
                print(f"Found BOM for encoding: {encoding}")
            # Remove the BOM
            raw_data = raw_data[len(bom):]
            # Try to decode with the detected encoding
            try:
                decoded_text = raw_data.decode(encoding)
                return clean_xml_content(decoded_text, debug), encoding
            except UnicodeDecodeError:
                # If it still fails, continue to try other encodings
                pass

    # Look for Mac OS X metadata
    str_content_latin1 = raw_data.decode('latin-1', errors='replace')

    if "Mac OS X" in str_content_latin1[:100]:
        if debug:
            print("Detected Mac OS X metadata")

        # Try to find XML markers after Mac metadata
        clean_content, encoding = find_xml_after_mac_metadata(str_content_latin1, debug)
        if clean_content:
            return clean_content, encoding

    # Try various encodings if no Mac metadata was found or cleaning failed
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            decoded_text = raw_data.decode(encoding)
            # Check if decoded content looks like XML and clean if needed
            clean_content = clean_xml_content(decoded_text, debug)
            if clean_content.strip().startswith('<?xml') or clean_content.strip().startswith('<Annotations'):
                if debug:
                    print(f"Successfully decoded with {encoding}")
                return clean_content, encoding
        except UnicodeDecodeError:
            if debug:
                print(f"Failed to decode with {encoding}")

    # If all else fails, use latin-1 which can decode any byte stream
    decoded_text = raw_data.decode('latin-1', errors='replace')
    if debug:
        print("Falling back to latin-1 with error replacement")

    # Try to clean the content even with the fallback encoding
    clean_content = clean_xml_content(decoded_text, debug)
    return clean_content, 'latin-1'

def find_xml_after_mac_metadata(content: str, debug: bool = True) -> Tuple[str, str]:
    """
    Find XML content after Mac OS X metadata.
    
    Args:
        content: The string content to search
        debug: Whether to print debug information
        
    Returns:
        Tuple of (clean_content, encoding)
    """
    # Try to find the actual XML start
    xml_markers = ['<?xml', '<Annotations', '<annotation', '<ANNOTATIONS']
    start_index = -1
    
    for marker in xml_markers:
        pos = content.find(marker)
        if pos != -1:
            start_index = pos
            break
    
    # If we found an XML marker, extract from there
    if start_index != -1:
        clean_content = content[start_index:]
        if debug:
            print(f"Found XML marker at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-after-mac-metadata'
    
    # Try to find the end of the metadata section
    # Common patterns that might indicate the end of metadata
    markers = [
        "[ZoneTransfer]",
        "<?xml",
        "<Annotations"
    ]
    
    for marker in markers:
        pos = content.find(marker)
        if pos != -1:
            # If we found [ZoneTransfer], skip to the next line
            if marker == "[ZoneTransfer]":
                next_line_pos = content.find('\n', pos)
                if next_line_pos != -1:
                    clean_content = content[next_line_pos+1:]
                    if debug:
                        print(f"Skipped to after [ZoneTransfer]. First 50 chars: {clean_content[:50]}")
                    return clean_content, 'detected-after-zonetransfer'
            else:
                clean_content = content[pos:]
                if debug:
                    print(f"Found marker {marker} at position {pos}. First 50 chars: {clean_content[:50]}")
                return clean_content, 'detected-after-marker'
    
    # Last resort: try to find anything that looks like XML
    xml_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>')
    match = xml_pattern.search(content)
    if match:
        start_index = match.start()
        clean_content = content[start_index:]
        if debug:
            print(f"Found XML-like pattern at position {start_index}. First 50 chars: {clean_content[:50]}")
        return clean_content, 'detected-xml-pattern'
    
    # If all else fails, return the original content with a warning
    if debug:
        print("WARNING: Could not identify XML content in the file")
    return content, 'original-unmodified'

def clean_xml_content(content: str, debug: bool = True) -> str:
    """
    Clean the XML content by removing non-XML data and ensuring it's well-formed.
    
    Args:
        content: The string content to clean
        debug: Whether to print debug information
        
    Returns:
        Cleaned XML content
    """
    # Check if content already starts with XML declaration or root element
    if content.strip().startswith('<?xml') or content.strip().startswith('<Annotations'):
        return content
    
    # Try to find the first XML marker
    xml_markers = ['<?xml', '<Annotations', '<annotation', '<ANNOTATIONS']
    start_index = -1
    
    for marker in xml_markers:
        pos = content.find(marker)
        if pos != -1:
            start_index = pos
            break
    
    if start_index != -1:
        return content[start_index:]
    
    # If no standard XML markers are found, try to find any tag
    xml_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>')
    match = xml_pattern.search(content)
    if match:
        return content[match.start():]
    
    return content

def parse_xml(xml_content: str, debug: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse XML content into a dictionary.
    
    Args:
        xml_content: XML content as string
        debug: Whether to print debug information
        
    Returns:
        Parsed XML as dictionary or None if parsing fails
    """
    try:
        import xmltodict
        return xmltodict.parse(xml_content)
    except Exception as e:
        if debug:
            print(f"XML parsing error: {str(e)}")
            
            # Try SAX parser for more diagnostic information
            try:
                xml.sax.parseString(xml_content, xml.sax.ContentHandler())
                print("SAX parser could parse the XML, but xmltodict failed.")
            except xml.sax.SAXParseException as sax_error:
                print(f"SAX parser also failed: {str(sax_error)}")
                print(f"Error at line {sax_error.getLineNumber()}, column {sax_error.getColumnNumber()}")
                
                # Print problematic lines
                lines = xml_content.split('\n')
                if sax_error.getLineNumber() <= len(lines):
                    error_line = lines[sax_error.getLineNumber() - 1]
                    print(f"Error line content: {error_line}")
        
        return None

def manual_extract_annotations(xml_content: str, debug: bool = True) -> Optional[Dict[str, Any]]:
    """
    Manually extract annotation data using regex when XML parsing fails.
    
    Args:
        xml_content: XML content as string
        debug: Whether to print debug information
        
    Returns:
        Dictionary with extracted annotations or None if extraction fails
    """
    try:
        # Try to extract annotations manually
        annotations_match = re.search(r'<Annotations.*?>(.*?)</Annotations>', 
                                     xml_content, re.DOTALL)
        if annotations_match:
            if debug:
                print("Found Annotations element. Attempting to extract data...")
            
            # Extract annotation elements
            annotation_pattern = re.compile(r'<Annotation[^>]*?Id="(\d+)"[^>]*?Name="([^"]*)"[^>]*?LineColor="([^"]*)"')
            annotations = annotation_pattern.findall(xml_content)
            
            if annotations:
                if debug:
                    print(f"Manually extracted {len(annotations)} annotations")
                
                # Build a dictionary similar to what xmltodict would produce
                result = {"Annotations": {"Annotation": []}}
                
                for anno_id, name, color in annotations:
                    anno_dict = {
                        "@Id": anno_id,
                        "@Name": name,
                        "@LineColor": color,
                        "Regions": {"Region": []}
                    }
                    
                    # Find regions for this annotation
                    anno_start = xml_content.find(f'<Annotation Id="{anno_id}"')
                    if anno_start != -1:
                        anno_end = xml_content.find('</Annotation>', anno_start)
                        if anno_end != -1:
                            anno_text = xml_content[anno_start:anno_end]
                            
                            # Extract regions
                            region_pattern = re.compile(r'<Region[^>]*?Id="([^"]*)"')
                            for region_match in region_pattern.finditer(anno_text):
                                region_id = region_match.group(1)
                                region_dict = {"@Id": region_id, "Vertices": {"Vertex": []}}
                                
                                # Find vertices for this region
                                region_start = anno_text.find(f'<Region Id="{region_id}"', region_match.start())
                                if region_start != -1:
                                    region_end = anno_text.find('</Region>', region_start)
                                    if region_end != -1:
                                        region_text = anno_text[region_start:region_end]
                                        
                                        # Extract vertices
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
            print(f"Error in manual extraction: {str(e)}")
    
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
        # Default to black if conversion fails
        return (0, 0, 0)

def extract_annotation_coordinates(xml_dict: Dict[str, Any], debug: bool = True) -> Tuple[float, pd.DataFrame]:
    """
    Extract annotation coordinates from parsed XML dictionary.
    
    Args:
        xml_dict: Dictionary containing parsed XML
        debug: Whether to print debug information
        
    Returns:
        Tuple of (reduced_annotations, DataFrame with coordinates)
    """
    # Extract MicronsPerPixel value
    try:
        reduced_annotations = float(xml_dict['Annotations'].get('@MicronsPerPixel', 1))
    except:
        reduced_annotations = 1.0
        if debug:
            print("Warning: Could not find or parse MicronsPerPixel, using default value 1")

    # Extract annotations
    annotations = xml_dict.get("Annotations", {}).get("Annotation", [])
    
    if not annotations:
        if debug:
            print("Warning: No annotations found in the XML")
        return reduced_annotations, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])
    
    # Handle both list and single annotation cases
    if not isinstance(annotations, list):
        annotations = [annotations]
    
    xyout = []
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
                                    if debug:
                                        print(f"Error processing vertex: {str(vertex_error)}")
                        except Exception as annotation_error:
                            if debug:
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
                                if debug:
                                    print(f"Error processing vertex: {str(vertex_error)}")
                    except Exception as region_error:
                        if debug:
                            print(f"Error processing region: {str(region_error)}")
            except Exception as layer_error:
                if debug:
                    print(f"Error processing layer: {str(layer_error)}")

    if not xyout:
        if debug:
            print("Warning: No valid annotation coordinates were extracted")
    else:
        if debug:
            print(f"Successfully extracted {len(xyout)} annotation coordinates")
            
    return reduced_annotations, pd.DataFrame(xyout, columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])

def load_xml_annotations(xml_path: str, debug: bool = True) -> Tuple[float, pd.DataFrame]:
    """
    Complete function to load XML annotations from a file.
    
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
    try:
        # Read and clean the XML file
        xml_content, encoding = read_xml_file(xml_path, debug)
        
        if debug:
            print(f"Cleaned XML content with encoding {encoding}. Preview: {xml_content[:100]}")
        
        # Parse the XML content
        xml_dict = parse_xml(xml_content, debug)
        
        # If parsing succeeded, extract annotation coordinates
        if xml_dict:
            return extract_annotation_coordinates(xml_dict, debug)
        
        # If parsing failed, try manual extraction
        if debug:
            print("Standard parsing failed. Attempting manual extraction...")
        
        manual_dict = manual_extract_annotations(xml_content, debug)
        if manual_dict:
            return extract_annotation_coordinates(manual_dict, debug)
        
        # If all else fails, return empty DataFrame
        if debug:
            print("All parsing methods failed. Returning empty DataFrame.")
        return 1.0, pd.DataFrame(columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])
    
    except Exception as e:
        if debug:
            print(f"Error loading XML annotations: {str(e)}")
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
    try:
        # Read and clean the XML file
        xml_content, encoding = read_xml_file(xml_path, debug)
        
        if debug:
            print(f"Cleaned XML content with encoding {encoding}. Preview: {xml_content[:100]}")
        
        # Parse the XML content
        xml_dict = parse_xml(xml_content, debug)
        
        # If parsing failed, try manual extraction
        if not xml_dict:
            if debug:
                print("Standard parsing failed. Attempting manual extraction...")
            xml_dict = manual_extract_annotations(xml_content, debug)
        
        # If we have valid XML data, extract layers
        if xml_dict:
            annotations = xml_dict.get("Annotations", {}).get("Annotation", [])
            
            if not annotations:
                if debug:
                    print("Warning: No annotations found in the XML")
                return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])
            
            # Handle both list and single annotation cases
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
                        print(f"Error processing layer: {str(e)}")
            
            if data:
                return pd.DataFrame(data)
        
        # If standard extraction failed, try direct regex matching
        if debug:
            print("Standard extraction failed. Attempting direct regex matching...")
        
        # Extract annotation elements directly
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
                        print(f"Error processing color value: {str(e)}")
            
            if data:
                return pd.DataFrame(data)
        
        # If all attempts fail, return empty DataFrame
        return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])
    
    except Exception as e:
        if debug:
            print(f"Error extracting annotation layers: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return pd.DataFrame(columns=['Layer Name', 'Color', 'Whitespace Settings'])