"""
Updated to use xml_handler module

Author:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated March 19, 2025
"""

from base.utils.xml_handler import load_xml_annotations

def load_annotations(xml_file):
    """
    Load annotation coordinates from an XML file into a DataFrame.
    Now uses the xml_handler module for robust parsing across different platforms.

    Parameters:
    - xml_file (str): The path to the XML file containing annotations.

    Returns:
    - reduced_annotations (float): The value of 'MicronsPerPixel' under 'Annotations' if present, otherwise 1.0.
    - xyout_df (pandas.DataFrame): DataFrame containing the annotation labels and coordinates, organized as:
        'Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex'.
    """
    return load_xml_annotations(xml_file)