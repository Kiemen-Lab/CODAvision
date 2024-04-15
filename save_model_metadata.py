import os
import numpy as np
import matplotlib.pyplot as plt
import plot_cmap_legend
import pickle

# This function currently saves classNames as a Matlab character matrix and as a Python string list (it is the same
# variable but each interpreter reads it differently)
# Moreover, if there exist a neural network trained with matlab (DAGNetwork) or the variable "classNames" was saved
# from Matlab (string vector), these variables remain unchanged since python doesnt recognize these types and classifies
# them as "None", which crashes the code
# Because of this, the check of whether a network has already been trained hasnt been added yet

def save_model_metadata(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate):
    """
           Saves model metadata to pickle file.

           Parameters:
           - pthDL (str): The path where the model metadata is saved
           - pthIm (str): The path where the images are located.
           - WS (list): List of 5 int vectors that contains whitespace removal options, tissue order, tissues being deleted and whitespace distribution
           - nm (str): The name of the model
           - umpix (int/ndarray{1,1}): scaling factor
           - cmap (ndarray{n,3}): The color map of the model with one column per RGB value
           - sxy (ndarray{1,1}): Training tiles size
           - classNames (ndarray): Array of strings containing class names
           - ntrain (ndarray{1,1): Number of training tiles
           - nvalidate (ndarray{1,1}): Number of validation tiles

           Returns:
           Nothing, but saves metadata to pickle file.
    """
    if not os.path.isdir(pthDL):
        os.mkdir(pthDL)

    datafile = os.path.join(pthDL.rstrip('\\'), 'net.pkl')
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

                if len(classNames) + 1 == max(WS[2]):
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

    existing_variables = {}

    with open(datafile, 'rb') as f:
        loaded_data = pickle.load(f)
    for key, value in loaded_data.items():
        if value is not None:
            existing_variables[key] = value
    existing_variables.update({"WS": WS, "cmap": cmap, "nblack": nblack, "nwhite": nwhite})
    # Save the data to a pickle file
    with open(datafile, 'wb') as f:
        pickle.dump(existing_variables, f)

    # plot color legend
    plot_cmap_legend.plot_cmap_legend(cmap, classNames)
    plt.savefig(os.path.join(pthDL, 'model_color_legend.png'))


pthDL=r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests\5x\04_03_2024'
pthim=r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Test Dashboard\5x'
WS = [[0,0,0,0,0,2,0],[6,6],[1,2,3,4,5,6,3],[7,2,4,3,1,6],4]
nm = '04_03_2024'
umpix = 2
cmap = np.array([[0, 255, 0],
                 [255, 255, 0],
                 [255, 128, 0],
                 [0, 255, 255],
                 [0, 0, 255],
                 [0, 0, 0]])
sxy = 1000
classNames = ["bronchioles","alveolo","smooth_operator","mets","test","whitespace","black"]
ntrain = 15
nvalidate = 3
nblack = 6
nwhite = 4

save_model_metadata(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate)