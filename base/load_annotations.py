import xmltodict
import pandas as pd

"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 05, 2024
"""

def load_annotations(xml_file):
    """
       Load annotation coordinates from an XML file into a DataFrame.

       Parameters:
       - xml_file (str): The path to the XML file containing annotations.

       Returns:
       - reduced_annotations (str or None): The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise None.
       - xyout_df (pandas.DataFrame): DataFrame containing the annotation labels and coordinates, organized as:
           'Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'.
       """

    # Use xmltodict to directly parse the XML file
    with open(xml_file, 'r', encoding='utf-8') as file:
        my_xml = file.read()
        my_dict = xmltodict.parse(my_xml)

    xyout = []

    reduced_annotations = my_dict['Annotations'].get('@MicronsPerPixel', 1)
    reduced_annotations = float(reduced_annotations)

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
