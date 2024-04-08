import os
import numpy as np
import matplotlib.pyplot as plt


def save_model_metadata(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate):
    if not os.path.isdir(pthDL):
        os.mkdir(pthDL)

    datafile = os.path.join(pthDL.rstrip('\\'), 'net.mat')
    print('Saving model metadata and classification colormap...')

    if classNames[-1] != "black":
        classNames.append("black")

    if classNames[-1] == "black":
        classNames.pop()

    # fix WS and classNames if there are classes to delete
    ndelete = WS[4]
    ndelete.sort(reverse=True)
    if ndelete:
        for b in ndelete:
            ncombine = WS[2]
            nload = WS[3]
            oldnum = ncombine[b]
            ncombine[b] = 1
            ncombine = [n - 1 if n > oldnum else n for n in ncombine]
            nload = [n for n in nload if n != b]

            if len(classNames) == max(WS[2]):
                zz = [i for i in range(len(classNames)) if i + 1 not in [b, oldnum]]
                classNames = [classNames[i] for i in zz]
                cmap = cmap[zz]

            WS[2] = ncombine

        WS[2] = ncombine
        WS[3] = nload

    nwhite = WS[2][WS[1] - 1]
    nwhite = nwhite[0]

    if max(WS[2]) != len(classNames):
        raise ValueError('The length of classNames does not match the number of classes specified in WS[2].')

    if classNames[-1] != "black":
        classNames.append("black")

    nblack = len(classNames)

    if os.path.isfile(datafile):
        variable_info = scipy.io.whosmat(datafile)
        if 'net' in variable_info and 'pthim' in variable_info:
            raise ValueError(f'A network has already been trained for model {nm}. Choose a new model name to retrain.')
        elif 'net' in variable_info and 'pthim' not in variable_info:
            np.savez(datafile, pthim=pthim, WS=WS, nm=nm, umpix=umpix, cmap=cmap, sxy=sxy, nblack=nblack,
                     nwhite=nwhite, classNames=classNames, ntrain=ntrain, nvalidate=nvalidate)
            raise ValueError(
                f'A network has already been trained for model {nm}. Metadata added to net.mat file. Choose a new model name to retrain.')

    # if file doesn't exist, save all the variables
    np.savez(datafile, pthim=pthim, WS=WS, nm=nm, umpix=umpix, cmap=cmap, sxy=sxy, nblack=nblack,
             nwhite=nwhite, classNames=classNames, ntrain=ntrain, nvalidate=nvalidate)

    # plot color legend
    plot_cmap_legend(cmap, classNames)
    plt.savefig(os.path.join(pthDL, 'model_color_legend.png'))
