"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 17, 2024
"""

import os
import matplotlib.pyplot as plt
from .plot_cmap_legend import plot_cmap_legend
import pickle

def save_model_metadata(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate):
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
    ndelete = WS[4]
    if isinstance(ndelete, list):
        ndelete.sort(reverse=True)
        if ndelete:
            for b in ndelete:
                ncombine = WS[2]
                nload = WS[3]
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
    elif isinstance(ndelete, int):
        ncombine = WS[2]
        nload = WS[3]
        oldnum = ncombine[ndelete - 1]
        ncombine[ndelete - 1] = 1
        ncombine = [n - 1 if n > oldnum else n for n in ncombine]
        nload = [n for n in nload if n != ndelete]

        if len(classNames) == max(WS[2]):
            zz = [i for i in range(len(classNames)) if i + 1 not in [oldnum]]
            classNames = [classNames[i] for i in zz]
            cmap = cmap[zz]
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
            {"pthim": pthim, "pthDL": pthDL, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
             "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "nvalidate": nvalidate})

        with open(datafile, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        print('Creating Net metadata file...')
        with open(datafile, 'wb') as f:
            pickle.dump({"pthim": pthim, "pthDL": pthDL, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
                         "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite,
                         "nvalidate": nvalidate}, f)

    # plot color legend
    plot_cmap_legend(cmap, classNames)
    plt.savefig(os.path.join(pthDL, 'model_color_legend.png'))


