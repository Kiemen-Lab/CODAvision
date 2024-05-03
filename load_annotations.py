import xmltodict
import pandas as pd


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

    reduced_annotations = my_dict['Annotations'].get('@MicronsPerPixel')
    reduced_annotations = float(reduced_annotations)

    annotations = my_dict.get("Annotations", {}).get("Annotation", [])

    for annotation in annotations:

        if 'Region' in annotation.get("Regions", {}):  #checks weather there are annotations in the layer
            annotation_id = float(annotation.get('@Id'))
            regions = annotation["Regions"]["Region"]
            if type(regions) == list:
                for region in regions:
                    annotation_number = float(region.get('@Id'))
                    vertices = region.get("Vertices", {}).get("Vertex", [])
                    for vertex in vertices:
                        x = float(vertex.get('@X'))
                        y = float(vertex.get('@Y'))
                        xyout.append([annotation_id, annotation_number, x, y])
            elif type(regions) == dict:
                annotation_number = float(regions.get('@Id'))
                vertices = regions.get("Vertices", {}).get("Vertex", [])
                for vertex in vertices:
                    x = float(vertex.get('@X'))
                    y = float(vertex.get('@Y'))
                    xyout.append([annotation_id, annotation_number, x, y])

    xyout_df = pd.DataFrame(xyout, columns=['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'])
    return reduced_annotations, xyout_df

# #Example usage
# def main():
#     xml_file = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\SG_014_0016.xml'
#
#     reduced_annotations, annotations_df = load_annotations(xml_file)
#
#     print("Reduced Annotations (Microns Per Pixel):", reduced_annotations)
#     print("\nAnnotations DataFrame:")
#     # print(annotations_df.head())
#     print(annotations_df)
#
# if __name__ == "__main__":
#     main()