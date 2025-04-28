"""
Author: Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
Updated: March 2025
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import OrderedDict

# Constants for XML generation
MICRONS_PER_PIXEL = "0.460100"

DEFAULT_METADATA = {
    "ReadOnly": "0",
    "NameReadOnly": "0",
    "LineColorReadOnly": "0",
    "Incremental": "0",
    "Type": "4",
    "LineColor": "13434828",
    "Visible": "1",
    "Selected": "0",
    "MarkupImagePath": "",
    "MacroName": "",
}

REGION_METADATA = {
    "Type": "0",
    "Zoom": "1.0",
    "Selected": "0",
    "ImageLocation": "",
    "ImageFocus": "-1",
    "Length": "0.0",
    "Area": "0.0",
    "LengthMicrons": "0.0",
    "AreaMicrons": "0.0",
    "Text": "",
    "NegativeROA": "0",
    "InputRegionId": "0",
    "Analyze": "1",
}

ANNOTATIONS_ATTRIBUTES = [
    {"Name": "Description", "Id": "0", "Value": ""}
]

REGION_ATTRIBUTE_HEADERS = [
    {"Id": "9999", "Name": "Region", "ColumnWidth": "-1"},
    {"Id": "9997", "Name": "Length", "ColumnWidth": "-1"},
    {"Id": "9996", "Name": "Area", "ColumnWidth": "-1"},
    {"Id": "9998", "Name": "Text", "ColumnWidth": "-1"},
    {"Id": "1", "Name": "Description", "ColumnWidth": "-1"},
]


def rgb_to_bgr_hex(rgb):
    """
    Convert RGB to ImageScope's BGR format as a decimal integer.
    """
    r, g, b = rgb
    return str((b << 16) + (g << 8) + r)


def prettify(elem):
    """
    Return a pretty-printed XML string from an ElementTree Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    return minidom.parseString(rough_string).toprettyxml(indent="  ")


def collect_all_labels_from_folder(folder_path):
    """
    Scans all GeoJSON files in the folder to collect a master list of all labels.
    Returns sorted list of unique label names.
    """
    all_labels = set()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.geojson'):
            geojson_path = os.path.join(folder_path, file_name)

            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            # Get features list
            if isinstance(geojson_data, dict):
                features = geojson_data.get("features", [geojson_data])
            elif isinstance(geojson_data, list):
                features = geojson_data
            else:
                continue

            # Collect labels from each feature
            for feature in features:
                props = feature.get("properties", {})
                classification = props.get("classification", {})
                name = classification.get("name")

                if name:
                    all_labels.add(name)

    # Return sorted list for consistent ordering
    return sorted(list(all_labels))


def convert_geojson_to_imagescope_xml(geojson_path, label_order):
    """
    Converts a GeoJSON file into an ImageScope-compatible XML string.
    Uses provided label_order to maintain consistent ordering.
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Get features list
    if isinstance(geojson_data, dict):
        features = geojson_data.get("features", [geojson_data])
    elif isinstance(geojson_data, list):
        features = geojson_data
    else:
        raise ValueError("Unsupported GeoJSON format.")

    # Use OrderedDict to maintain order
    annotations_by_name = OrderedDict()

    # Initialize all possible labels from the master list
    for label in label_order:
        annotations_by_name[label] = {
            "color": None,  # Will be set when we encounter this label
            "regions": []
        }

    # Process features
    for feature in features:
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        name = classification.get("name")
        color = classification.get("color", [0, 255, 0])  # Default: green

        if not name or name not in annotations_by_name:
            continue

        # Set color if not already set
        if annotations_by_name[name]["color"] is None:
            annotations_by_name[name]["color"] = rgb_to_bgr_hex(color)

        # Process geometry
        geometry = feature.get("geometry", {})
        coords_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if coords_type == "Polygon":
            polygon = coords[0]
            if polygon[0] != polygon[-1]:  # Close polygon if needed
                polygon.append(polygon[0])
            annotations_by_name[name]["regions"].append(polygon)

        elif coords_type == "MultiPolygon":
            for poly in coords:
                if poly and poly[0]:
                    if poly[0][0] != poly[0][-1]:
                        poly[0].append(poly[0][0])
                    annotations_by_name[name]["regions"].append(poly[0])

    # Remove empty labels and set default color for unused labels
    for name in list(annotations_by_name.keys()):
        if not annotations_by_name[name]["regions"]:
            del annotations_by_name[name]
        elif annotations_by_name[name]["color"] is None:
            annotations_by_name[name]["color"] = DEFAULT_METADATA["LineColor"]

    # Set up XML structure
    annotations_element = ET.Element("Annotations", {"MicronsPerPixel": MICRONS_PER_PIXEL})

    annotation_id = 1
    region_id = 1
    for name, data in annotations_by_name.items():
        # Create annotation with default metadata
        annotation_metadata = DEFAULT_METADATA.copy()
        annotation_metadata.update({
            "Id": str(annotation_id),
            "Name": name,
            "LineColor": data["color"],
            "Selected": "1",  # Selected for annotations
        })

        annotation = ET.SubElement(annotations_element, "Annotation", annotation_metadata)

        # Add annotation-level attributes
        attributes = ET.SubElement(annotation, "Attributes")
        for attr in ANNOTATIONS_ATTRIBUTES:
            ET.SubElement(attributes, "Attribute", attr)

        # Add region headers
        regions = ET.SubElement(annotation, "Regions")
        headers = ET.SubElement(regions, "RegionAttributeHeaders")
        for attr in REGION_ATTRIBUTE_HEADERS:
            ET.SubElement(headers, "AttributeHeader", attr)

        # Add regions
        for coords in data["regions"]:
            # Create region with default metadata
            region_metadata = REGION_METADATA.copy()
            region_metadata.update({
                "Id": str(region_id),
                "DisplayId": str(region_id)
            })

            region = ET.SubElement(regions, "Region", region_metadata)

            ET.SubElement(region, "Attributes")
            vertices = ET.SubElement(region, "Vertices")
            for x, y in coords:
                ET.SubElement(vertices, "Vertex", {"X": str(x), "Y": str(y), "Z": "0"})

            region_id += 1

        ET.SubElement(annotation, "Plots")
        annotation_id += 1

    # Output as pretty XML string
    return prettify(annotations_element)


# Example usage
if __name__ == "__main__":
    folder_path = r"\\10.99.134.183\kiemen-lab-data\Valentina Matos\qupath project 3"

    # First pass: collect all possible labels from all files
    all_labels = collect_all_labels_from_folder(folder_path)
    print(f"Found {len(all_labels)} unique labels across all files")

    # Second pass: process each file with consistent label order
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.geojson'):
            geojson_file = os.path.join(folder_path, file_name)
            xml_output = convert_geojson_to_imagescope_xml(geojson_file, all_labels)
            xml_file = geojson_file.replace('.geojson', '.xml')

            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_output)

            print(f"✅ XML saved for: {file_name}")