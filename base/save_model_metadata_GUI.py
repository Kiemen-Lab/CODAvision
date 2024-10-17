"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: September 30, 2024
"""

import os
import matplotlib.pyplot as plt
from .plot_cmap_legend import plot_cmap_legend
import pickle

def save_model_metadata_GUI(pthDL, pthim, pthtest,  WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate, final_df, combined_df):
    """
      Saves model metadata to a pickle file and generates a color map legend plot.

      Parameters:
      - pthDL (str): The path where the model metadata will be saved.
      - pthim (str): The path where the images are located.
      - WS (list): List containing whitespace removal options, tissue order, tissues being deleted,
                   and whitespace distribution.
      - nm (str): The name of the model.
      - umpix (int/ndarray{1,1}): Scaling factor.
      - cmap (ndarray{n,3}): The color map of the model with one column per RGB value.
      - sxy (int/ndarray{1,1}): Training tiles size.
      - classNames (list): List of strings containing class names.
      - ntrain (int/ndarray{1,1}): Number of training tiles.
      - nvalidate (int/ndarray{1,1}): Number of validation tiles.
      - final_df (DataFrame): The final raw DataFrame with the annotation layer settings created thorugh the GUI.
      - combined_df (DataFrame): The final combined DataFrame with the names and colors of the layers used for the
        classification.

      Returns:
      None

      This function saves the provided model metadata to a pickle file located at the specified path (pthDL).
      It also creates a color map legend plot based on the provided color map and class names, and saves
      the plot as 'model_color_legend.png' in the same directory as the model metadata.
      """
    if not os.path.isdir(pthDL):
        os.mkdir(pthDL)

    print('Saving model metadata and classification colormap...')

    if classNames[-1] != "black":
        classNames.append("black")

    if classNames[-1] == "black":
        classNames.pop()

    # fix WS and classNames if there are classes to delete
    ndelete = WS[4].copy()
    if isinstance(ndelete, list):
        ndelete.sort(reverse=True)
        if ndelete:
            for b in ndelete:
                ncombine = WS[2].copy()
                nload = WS[3].copy()
                oldnum = ncombine[b - 1]
                ncombine[b - 1] = 1
                ncombine = [n - 1 if n > oldnum else n for n in ncombine]
                nload = [n for n in nload if n != b]

                if len(classNames) == max(WS[2]):
                    zz = [i for i in range(len(classNames)) if i + 1 not in [b, oldnum]]
                    classNames = [classNames[i] for i in zz]
                    cmap = cmap[zz]

                WS[2] = ncombine

            WS[2] = ncombine
            WS[3] = nload


    nwhite = WS[2]
    nwhite = nwhite[WS[1][0] - 1]
    if max(WS[2]) != len(classNames):
        raise ValueError('The length of classNames does not match the number of classes specified in WS[2].')
    if classNames[-1] != "black":
        classNames.append("black")
    nblack = len(classNames)

    datafile = os.path.join(pthDL.rstrip('\\'), 'net.pkl')

    # Save the data to a pickle file
    if os.path.exists(datafile):
        print('Net file already exists, updating data...')
        with open(datafile, 'rb') as f:
            try:
                existing_data = pickle.load(f)
            except EOFError:
                existing_data = {}

        existing_data.update(
            {"pthim": pthim, "pthDL": pthDL, "pthtest":pthtest, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
             "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "final_df": final_df,
                         "combined_df": combined_df, "nvalidate": nvalidate})

        with open(datafile, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        print('Creating Net metadata file...')
        with open(datafile, 'wb') as f:
            pickle.dump({"pthim": pthim, "pthDL": pthDL, "pthtest": pthtest, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
                         "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "final_df": final_df,
                         "combined_df": combined_df,"nvalidate": nvalidate}, f)

    # plot color legend
    plot_cmap_legend(cmap, classNames)
    plt.savefig(os.path.join(pthDL, 'model_color_legend.jpg'))

#Example usage

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    # Inputs
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\october_test_delete'
    pthim = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
    WS = [[ 2, 0, 0, 1, 0, 0, 2, 0, 2, 2, 2, 0, 0, 0 ],  # remove whitespace if 0, keep only whitespace if 1, keep both if 2
          [7, 6],  # first = add removed whitespace to this class, second = add removed tissue to this class
          [ 1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 10, 8, 11, 12],  # rename classes according to this order
          [ 14, 13, 11, 10, 12, 8, 6, 5, 4, 3, 2, 1, 9, 7 ],  # reverse priority of classes (left = bottom, right = top)
          [14]]  # List of annotations to delete
    nm = '04_456_2024_test_delete'
    umpix = 2
    cmap = np.array([
        [0, 255, 255],
        [0, 0, 255],
        [170, 255, 127],
        [255, 255, 127],
        [170, 0, 127],
        [255, 170, 255],
        [255, 255, 255],
        [85, 0, 0],
        [85, 85, 127],
        [0, 0, 0],
        [170, 170, 127],
        [255, 0, 0]
    ])

    sxy = 1000
    classNames = ['islets', 'normal duct', 'blood vessel', 'fat', 'acini', 'ecm', 'noiscombo', 'panincombo', 'nerve', 'immune', 'PDAC', 'weird']
    ntrain = 15
    nvalidate = 3

    #Final df

    data_fd = {
    'Layer Name': ['islets', 'normal duct', 'blood vessel', 'fat', 'acini', 'ecm', 'whitespace', 'panin', 'noise', 'nerve'],
    'Color': ['(0, 255, 0)', '(255, 255, 0)', '(255, 0, 0)', '(0, 255, 255)', '(255, 0, 255)', '(255, 128, 64)', '(0, 0, 255)', '(255, 0, 128)', '(64, 128, 128)', '(128, 0, 255)'],
    'Whitespace Settings': [2, 0, 0, 1, 0, 0, 2, 0, 2, 0],
    'Delete layer': [False, False, False, False, False, False, False, False, False, False],
    'Combined layers': [1, 2, 3, 4, 5, 6, 7, 8, 7, 9],
    'Nesting': [7, 9, 10, 1, 2, 3, 13, 8, 12, 11],
    'Component analysis': [False, False, False, False, False, False, False, False, False, False]
}

    data_combined_df = {
    'Layer Name': ['islets', 'normal duct', 'blood vessel', 'fat', 'acini', 'ecm', 'noisecombo', 'panincombo', 'nerve', 'immune', 'PDAC', 'weird'],
    'Color': ['(0, 255, 255)', '(0, 0, 255)', '(170, 255, 127)', '(255, 255, 127)', '(170, 0, 127)', '(255, 170, 255)', '(255, 255, 255)', '(85, 0, 0)', '(85, 85, 127)', '(0, 0, 0)', '(170, 170, 127)', '(255, 0, 0)'],
    'Whitespace Settings': [2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
    'Layer idx': [1, 2, 3, 4, 5, 6, '[7, 9]', '[8, 12]', 10, 11, 12, 13],
    'Delete layer': [False, False, False, False, False, False, False, False, False, False, False, False],
    'Deleted': [False, False, False, False, False, False, False, False, False, False, False, True]
}

    # Create DataFrames
    final_df = pd.DataFrame(data_fd)
    combined_df = pd.DataFrame(data_combined_df)

    save_model_metadata_GUI(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate, final_df, combined_df)




