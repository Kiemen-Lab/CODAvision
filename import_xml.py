"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 15, 2024
"""
from load_annotations import load_annotations
import os
import pickle


def import_xml(annotations_file, xmlfile, dm=None, ra=None):
    """
   Reads an XML file and imports annotation data, saving it to a pickle file.
   'load_annotations' function to extract the annotation coordinates from an XML file, is required

   Parameters:
   annotations_file (str): The file path for the output pickle file.
   xmlfile (str): The file path for the input XML file.
   dm (str): String indicating date and time the xml file was modified.
   ra (float, optional): The reduced annotations value. Defaults to 0.

   Returns:
   xyout_df (pandas.DataFrame): A DataFrame containing the annotation labels and coordinates.
   reduced_annotations (float): The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise None.
   """

    if ra is None:
        ra = 0
    if dm is None:
        dm = []

    print(' 1. of 4. Importing annotation data from xml file')

    reduced_annotations, xyout_df = load_annotations(xmlfile)
    reduced_annotations = float(reduced_annotations)
    if not xyout_df.empty:

        if ra == 1:
            reduced_annotations = ra
        xyout_df.iloc[:, 2:4] = xyout_df.iloc[:, 2:4] * reduced_annotations

        # Create the necessary directories if they don't exist
        annotations_dir = os.path.dirname(annotations_file)
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)

        if os.path.exists(annotations_file):
            print('File already exists, updating data...')
            with open(annotations_file, 'rb') as f:
                try:
                    existing_data = pickle.load(f)
                except EOFError:
                    existing_data = {}

            existing_data.update({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': dm})

            with open(annotations_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            print('Creating file...')
            with open(annotations_file, 'wb') as f:
                pickle.dump({'xyout': xyout_df.values, 'reduce_annotations': reduced_annotations, 'dm': dm}, f)

    return xyout_df, reduced_annotations


# if __name__ == "__main__":
    # Example usage

    # Inputs
    # annotations_file = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\data\SG_014_0016\annotations.pkl'
    # xmlfile = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\SG_014_0016.xml'
    # #dm = '17-Apr-2024 16:56:26'
    # ra = None
    #
    # xyout_df, reduced_annotations = import_xml(annotations_file, xmlfile)
    #
    # print("Reduced Annotations (Microns Per Pixel):", reduced_annotations)
    # print("\nAnnotations Coordinates DataFrame:")
    # print(xyout_df)