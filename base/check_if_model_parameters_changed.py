"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 18, 2024
"""
import pickle
import os
import glob
import time


def check_if_model_parameters_changed(datafile, WS, umpix, nwhite, pthim, imnm):
    """
       Check if model parameters have changed compared to the data loaded from a pickle file.

       Parameters:
       - datafile (str): The path to the pickle file containing the model parameters.
       - WS: Current value of the 'WS' parameter.
       - umpix: Current value of the 'umpix' parameter.
       - nwhite: Current value of the 'nwhite' parameter.
       - pthim: Current value of the 'pthim' parameter.

       Returns:
       - reload_xml (int): Indicates if XML reload is needed:
           - 0: No reload needed.
           - 1: Reload is needed due to missing or changed parameters.
           - Nothing: An error occurred during the process.
    """
    try:
        # Initialize the variable to indicate if XML reload is needed
        reload_xml = 0

        # Check if the pickle file exists
        if not os.path.exists(datafile):
            reload_xml = 0
        else:
            # Load data from the pickle file
            with open(datafile, 'rb') as f:
                data = pickle.load(f)

            # Check if variables exist in the loaded data
            if not all(key in data for key in ['WS', 'umpix', 'nwhite', 'pthim']):
                print('WS, umpix, nwhite, or pthim are missing in the pickle file. Reload the XML file to add them.')
                reload_xml = 1
            else:
                # Check if any variables have changed compared to the existing ones
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

        TA_pkl = os.path.join(pthim,'TA','TA_cutoff.pkl')
        if not os.path.exists(TA_pkl):
            print('Tissue mask evaluation has not been performed')
        else:
            file = os.path.join(pthim,'TA',imnm+'.tif')
            if os.path.exists(file):
                pkl_modification_time = os.path.getmtime(TA_pkl)
                tif_modification_time = os.path.getmtime(file)
                if tif_modification_time < pkl_modification_time:
                    reload_xml = 1
                    os.remove(file)
            else:
                reload_xml=1
        return reload_xml

    except Exception as e:
        print(f"An error occurred: {str(e)}")