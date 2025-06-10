import os
import json
import xmltodict

# Set up logging
import logging
logger = logging.getLogger(__name__)

def rgb2hex(rgb_list):  # rgb_list example: [255,0,3]
    r, g, b = rgb_list
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def geojson2xml(geojson_path):
    # Derive output filename in the same folder as input
    input_dir = os.path.dirname(geojson_path)
    input_base = os.path.basename(geojson_path)
    output_filename = os.path.join(input_dir, input_base.replace('.geojson', '.xml'))

    # Read in geojson file
    with open(geojson_path, 'r') as f:
        geojson_dict = json.load(f)

    # Process geojson data
    geojson_polygons = geojson_dict['features']
    xml_polygons, group_dict, group_list = [], {}, []

    for ind, geojson_polygon in enumerate(geojson_polygons):
        coords = geojson_polygon['geometry']['coordinates'][0]
        if coords[0] == coords[-1]:  # Remove duplicate last coordinate
            coords = coords[:-1]
        try:
            color = geojson_polygon['properties']['color']
        except KeyError:
            try:
                color = geojson_polygon['properties']['classification']['color']
            except KeyError:
                color = [255, 0, 3]
        try:
            label = geojson_polygon['properties']['classification']['name']
        except:
            label = 'tissue'
        group_dict[label] = color

        coords_dict = [{'@Order': str(ind_), '@X': str(coord[0]), '@Y': str(coord[1])} for ind_, coord in enumerate(coords)]

        xml_polygon = {'@Name': 'Annotation ' + str(ind),
                       '@Type': 'Polygon',
                       '@PartOfGroup': label,
                       '@Color': str(rgb2hex(color)).upper(),
                       'Coordinates': {'Coordinate': coords_dict}}
        xml_polygons.append(xml_polygon)

    for group, col in group_dict.items():
        group_list.append({'@Name': group, '@PartOfGroup': None, '@Color': col, 'Attributes': None})

    xml = {'ASAP_Annotations': {'Annotations': {'Annotation': xml_polygons},
                                'AnnotationGroups': {'Group': group_list}}}

    # Convert to XML and save
    xml_string = xmltodict.unparse(xml, pretty=True)
    with open(output_filename, 'w') as f:
        f.write(xml_string)

    logger.info(f'XML file {output_filename} generated from GeoJSON file {geojson_path}')

# Example usage
if __name__ == "__main__":
    geojson_path = r"\\path\kiemen-lab-data\Valentina Matos\qupath project 3\2024-02-26 10.36.39.geojson"
    geojson2xml(geojson_path)
