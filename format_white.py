import numpy as np

def format_white(J0, Ig, WS, szz):
    p = 1  # image number I think
    ws = WS[0]  # defines keep or delete whitespace
    wsa0 = WS[1]  # defines non-tissue label
    wsa = wsa0[0]
    try:
        wsfat = wsa0[1]
    except IndexError:
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
            ii = J0[k]
        except IndexError:
            continue
        iiNW = np.setdiff1d(ii, Ig)  # indices that are not white
        iiW = np.intersect1d(ii, Ig)  # indices that are white
        if ws[k] == 0:  # remove whitespace and add to wsa
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = wsa
        elif ws[k] == 1:  # keep only whitespace
            Jws.flat[iiW] = k
            Jws.flat[iiNW] = wsfat
        elif ws[k] == 2:  # keep both whitespace and non whitespace
            Jws.flat[iiNW] = k
            Jws.flat[iiW] = k

    # remove small objects and redefine labels (combine labels if desired)
    J = np.zeros(szz, dtype=int)
    for k in range(1, Jws.max() + 1):
        tmp = Jws == k
        ii = np.flatnonzero(tmp)
        J.flat[tmp] = wsnew[k - 1]
        P = np.column_stack((np.full((ii.size, 2), [p, wsnew[k - 1]]), ii))
        ind.extend(P)

    return J, ind

