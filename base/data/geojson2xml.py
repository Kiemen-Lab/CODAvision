import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

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

def convert_geojson_to_imagescope_xml(geojson_path, microns_per_pixel="0.504"):
    """
    Converts a GeoJSON file into an ImageScope-compatible XML string.
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

    annotations_by_name = {}

    for feature in features:
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        name = classification.get("name")
        color = classification.get("color", [0, 255, 0])  # Default: green

        if not name:
            continue  # Skip unnamed annotations

        # Initialize if new label
        if name not in annotations_by_name:
            annotations_by_name[name] = {
                "color": rgb_to_bgr_hex(color),
                "regions": []
            }

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

    # Set up XML structure
    annotations_element = ET.Element("Annotations", {"MicronsPerPixel": microns_per_pixel})

    annotation_id = 1
    region_id = 1
    for name, data in annotations_by_name.items():
        annotation = ET.SubElement(annotations_element, "Annotation", {
            "Id": str(annotation_id),
            "Name": name,
            "ReadOnly": "0",
            "NameReadOnly": "0",
            "LineColorReadOnly": "0",
            "Incremental": "0",
            "Type": "4",
            "LineColor": data["color"],
            "Visible": "1",
            "Selected": "1",
            "MarkupImagePath": "",
            "MacroName": ""
        })

        # Add annotation-level attribute
        attributes = ET.SubElement(annotation, "Attributes")
        ET.SubElement(attributes, "Attribute", {"Name": "Description", "Id": "0", "Value": ""})

        # Add region headers
        regions = ET.SubElement(annotation, "Regions")
        headers = ET.SubElement(regions, "RegionAttributeHeaders")
        for attr in [
            {"Id": "9999", "Name": "Region"},
            {"Id": "9997", "Name": "Length"},
            {"Id": "9996", "Name": "Area"},
            {"Id": "9998", "Name": "Text"},
            {"Id": "1", "Name": "Description"}
        ]:
            ET.SubElement(headers, "AttributeHeader", {**attr, "ColumnWidth": "-1"})

        # Add regions
        for coords in data["regions"]:
            region = ET.SubElement(regions, "Region", {
                "Id": str(region_id),
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
                "DisplayId": str(region_id)
            })

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
    geojson_file = r"\\10.99.134.183\kiemen-lab-data\Valentina Matos\qupath project 3\2024-02-26 10.36.39.geojson"
    xml_output = convert_geojson_to_imagescope_xml(geojson_file)
    with open(geojson_file.replace('.geojson', '.xml'), "w", encoding="utf-8") as f:
        f.write(xml_output)
    print("✅ XML saved.")
