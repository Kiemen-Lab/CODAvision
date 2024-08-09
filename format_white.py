"""
Author: Jaime Gomez (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 24, 2024
"""
import numpy as np
import time
import pandas as pd

def format_white(J0, Ig, WS, szz):
    """
        Creates the annotation mask of the image from the annotation coordinates and the nesting order

        Parameters:
            - J0 (list) ; List containing the coordinates of the annotations of each layer
            - Ig (np.ndarray): The non-zero indexes of TA
            - WS (list): List containing whitespace removal options, tissue order, tissues being deleted,
                       and whitespace distribution
            - szz (turple); Contains the dimensions of the image

        Outputs:
            - J (np.ndarray): The annotation mask of the image.
            - ind (list)
    """
    p = 1  # image number I think
    ws = WS[0]  # defines keep or delete whitespace
    wsa0 = WS[1]  # defines non-tissue label
    wsa = wsa0[0]
    if len(wsa0)>1:
        wsfat = wsa0[1]
    else:
        wsfat = 0
    wsnew = WS[2]  # redefines CNN label names
    wsorder = WS[3]  # gives order of annotations
    wsdelete = WS[4]  # lists annotations to delete

    Jws = np.zeros(szz, dtype=int)
    ind = []

    # remove white pixels from annotations areas
    for k in wsorder:
        if any(np.isin(wsdelete, k)):
            continue  # delete unwanted annotation layers
        try:
            py_index = k - 1
            ii = J0[:,:,py_index]
        except IndexError:
            continue
        #part1 = time.time()
        #print(f'Part 1 took {time.time()-part1}s')
        #part11 = time.time()
        iiNW = ii*(Ig==0)
        iiW = ii*Ig
        iiW = np.flatnonzero(iiW)
        iiNW = np.flatnonzero(iiNW)
        #print(f'Part 1.1 took {time.time()-part11}s')
        #print(np.sum(iiNW==iiNW1)/iiNW1.size)
        #print(np.sum(iiW==iiW1)/iiW1.size)
        #part2 = time.time()
        if ws[k-1] == 0 and iiNW.size > 0:  # remove whitespace and add to wsa
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = wsa
        elif ws[k-1] == 1 and iiNW.size > 0:  # keep only whitespace
            Jws.flat[iiW] = k
            Jws.flat[iiNW] = wsfat
        elif ws[k-1] == 2 and iiNW.size > 0:  # keep both whitespace and non whitespace
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = k
        #print(f'Part 2 took {time.time()-part2}s')

    # remove small objects and redefine labels (combine labels if desired)
    J = np.zeros(szz, dtype=int)
    unique_k = np.unique(Jws)
    #part3 = time.time()
    for k in unique_k:
        if k == 0:
            continue
        tmp = Jws == k
        ii = np.flatnonzero(tmp)
        J[tmp] = wsnew[k - 1]
        if ii.size > 0:
            P = np.column_stack((np.full((ii.size, 2), [p, wsnew[k - 1]]), ii))
            ind.extend(P)
    #print(f'Part 3 took {time.time()-part3}s')
    return J, ind