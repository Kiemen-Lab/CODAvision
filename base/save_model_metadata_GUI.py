"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: November 15, 2024
"""

import os
import matplotlib.pyplot as plt
from .plot_cmap_legend import plot_cmap_legend
import pickle

def save_model_metadata_GUI(pthDL, pthim, pthtest,  WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate, nTA, final_df, combined_df, model_type, batch_size, uncomp_train_pth = '', uncomp_test_pth = '', scale = '', create_down = '', downsamp_annotated = ''):
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
                    print(classNames)
                    zz = [i for i in range(len(classNames)) if i + 1 not in [oldnum]] # Used to be not in [b, oldnum], but it was deleteing an extra class
                    classNames = [classNames[i] for i in zz]
                    print(classNames)
                    cmap = cmap[zz]
                    print(cmap)

                WS[2] = ncombine

            WS[2] = ncombine
            WS[3] = nload


    nwhite = WS[2]
    nwhite = nwhite[WS[1][0] - 1]
    print(f'Max WS[2]: {max(WS[2])}')
    print(f'Classnames: {classNames}')
    if max(WS[2]) != len(classNames):
        raise ValueError('The length of classNames does not match the number of classes specified in WS[2].')
    if classNames[-1] != "black":
        classNames.append("black")
    nblack = len(classNames)

    datafile = os.path.join(pthDL.rstrip('\\'), 'net.pkl')

    #if model_type has a '+' in it replaceit with '_plus'
    if '+' in model_type:
        model_type = model_type.replace('+', '_plus')


    # Save the data to a pickle file
    if os.path.exists(datafile):
        print('Net file already exists, updating data...')
        with open(datafile, 'rb') as f:
            try:
                existing_data = pickle.load(f)
            except EOFError:
                existing_data = {}

        if umpix == 'TBD':
            existing_data.update(
                {"pthim": pthim, "pthDL": pthDL, "pthtest":pthtest, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
                 "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "final_df": final_df,
                 "combined_df": combined_df, "nvalidate": nvalidate, "nTA": nTA, "model_type": model_type,
                 "batch_size": batch_size, "uncomp_train_pth" : uncomp_train_pth, "uncomp_test_pth": uncomp_test_pth,
                 "scale": scale, "create_down": create_down, "downsamp_annotated": downsamp_annotated})
        else:
            existing_data.update(
                {"pthim": pthim, "pthDL": pthDL, "pthtest": pthtest, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap,
                 "sxy": sxy,
                 "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "final_df": final_df,
                 "combined_df": combined_df, "nvalidate": nvalidate, "nTA": nTA, "model_type": model_type,
                 "batch_size": batch_size})

        with open(datafile, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        print('Creating Net metadata file...')
        with open(datafile, 'wb') as f:
            if umpix == 'TBD':
                pickle.dump({"pthim": pthim, "pthDL": pthDL, "pthtest": pthtest, "WS": WS, "nm": nm, "umpix": umpix,
                             "cmap": cmap, "sxy": sxy,
                             "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite,
                             "final_df": final_df,
                             "combined_df": combined_df, "nvalidate": nvalidate, "nTA": nTA, "model_type": model_type,
                             "batch_size": batch_size, "uncomp_train_pth": uncomp_train_pth, "uncomp_test_pth": uncomp_test_pth,
                             "scale": scale, "create_down": create_down, "downsamp_annotated": downsamp_annotated}, f)
            else:
                pickle.dump({"pthim": pthim, "pthDL": pthDL, "pthtest": pthtest, "WS": WS, "nm": nm, "umpix": umpix, "cmap": cmap, "sxy": sxy,
                         "classNames": classNames, "ntrain": ntrain, "nblack": nblack, "nwhite": nwhite, "final_df": final_df,
                         "combined_df": combined_df,"nvalidate": nvalidate, "nTA": nTA, "model_type":model_type, "batch_size": batch_size}, f)

    # plot color legend
    plot_cmap_legend(cmap, classNames)
    plt.savefig(os.path.join(pthDL, 'model_color_legend.jpg'))

