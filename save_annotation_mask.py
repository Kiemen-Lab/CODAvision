"""
Author: Jaime Gomez (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 24, 2024
"""

import format_white
import os
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_objects
import pickle

def save_annotation_mask(I,outpth,WS,umpix,TA,kpb=0):
    """
    Creates and saves the annotation mask of an image

    Parameters:
        - I (np.ndarray): The image as a numpy array.
        - outpth (str): The path where the image will be saved.
        - WS (list): List containing whitespace removal options, tissue order, tissues being deleted,
                   and whitespace distribution.
        - umpix (int/ndarray{1,1}): Scaling factor.
        - TA (np.ndarray): The binary copy of the image's green value after it has been thresholded.

    Returns:
        - J (np.ndarray): The annotation mask of the image.
    """
    if umpix == 100:
        umpix = 1
    elif umpix==200:
        umpix = 2
    elif umpix==400:
        umpix = 4
    print('     2. of 4. Interpolating annotated regions and saving mask image')
    num = len(WS[0])
    try:    # Try to load 'xyout' from 'annotations.pkl' if the pkl file exists
        with open(os.path.join(outpth, 'annotations.pkl'), 'rb') as f:
            data = pickle.load(f)
            xyout = data['xyout']
        if xyout.size == 0: # If xyout is empty, return a black image
            J = np.zeros(I.size)
        else:
            xyout[:,2:4] = np.round(xyout[:,2:4]/umpix) # Round the vertices coordinates for the annotations after converting them to pixels
            if TA.size > 0: # if TA is not empty, remove small objects and invert it
                TA = TA > 0
                TA = remove_small_objects(TA.astype(bool), min_size=30, connectivity=2)
                TA = np.logical_not(TA)
            else: # If TA is empty, create a TA from the image by performing some operations
                I = I.astype(float)
                TA1 = np.std(I[:, :, [0, 1]], 2, ddof=1)
                TA2 = np.std(I[:, :, [0, 2]], 2, ddof=1)
                TA = np.max(np.concatenate((TA1, TA2), axis=2), axis=2)
                TAa = TA < 10
                TAb = I[:, :, 1] > 210
                TA = TAa & TAb
                TA *= 255
                TA = label(TA, np.ones((3, 3)))[0] >= 5
            Ig = np.flatnonzero(TA)  # Find the linear indices of non-zero elements
            szz = TA.shape  # Get the size of the binary image
            J = [[] for _ in range(num)] # Create a list with as many entries as there are tissues

            # interpolate annotation points to make closed objects
            for k in np.unique(xyout[:, 0]):
                Jtmp = np.zeros(szz, dtype=int) # Create a temporal np.array of the same size as the image
                bwtypek = Jtmp.copy()
                xyz = xyout[xyout[:, 0] == k, :] # Get the annotations for the current layer
                for pp in np.unique(xyz[:, 1]): # For each annotation
                    if pp == 0: # Skip if the annotation is empty
                        continue
                    cc = np.flatnonzero(xyz[:, 1] == pp) # Get the indices for the current annotation
                    xyv = np.vstack((xyz[cc, 2:4], xyz[cc[0], 2:4])) #Stack the coordinates of the current annotation
                    dxyv = np.sqrt(np.sum((xyv[1:, :] - xyv[:-1, :]) ** 2, axis=1)) # Calculate the distance between each vertix
                    dxyv_nonzero = dxyv != 0 # Create an array containing which distances are non-zero
                    xyv = xyv[np.concatenate(([True], dxyv_nonzero)), :] # Filter out points with zero distance between them
                    dxyv = dxyv[dxyv_nonzero] # Update 'dxyv' to include only the nonzero distances
                    dxyv = np.concatenate(([0], dxyv)) # Prepend a zero to the 'dxyv' array
                    ssd = np.cumsum(dxyv) # Calculate the cumulative sum of distances and store it in 'ssd'
                    ss0 = np.arange(1, np.ceil(ssd.max()) + 0.49, 0.49) # Create an array of regularly spaced numbers representing positions along the cumulative distance
                    # Interpolate the x and y coordinates of new points along the cumulative distance based on the original coordinates
                    # Round the interpolated values to integers and store them in 'xnew' and 'ynew'
                    xnew = np.interp(ss0, ssd, xyv[:, 0]).round().astype(int)
                    ynew = np.interp(ss0, ssd, xyv[:, 1]).round().astype(int)
                    try:
                        indnew = np.ravel_multi_index((ynew, xnew), szz) # calculate the linear indices of the interpolated points
                    except ValueError:
                        print('annotation out of bounds')
                        continue
                    indnew = indnew[~np.isnan(indnew)].astype(int) # Remove NaN values
                    bwtypek.flat[indnew] = 1 #Make the values in bwtypek one in the coordinates with an annotation vertix
                bwtypek = binary_fill_holes(bwtypek > 0)
                Jtmp[bwtypek == 1] = k # Save the annotation vertices in Jtemp with a value equal to their position in WS[0]
                if not kpb: # Include padding in the image
                    Jtmp[:400, :] = 0
                    Jtmp[:, :400] = 0
                    Jtmp[-401:, :] = 0
                    Jtmp[:, -401:] = 0
                J[int(k) - 1] = np.flatnonzero(Jtmp == k) #Store the annotation vertices indexes in the entry of J with their position in WS[0]

            del bwtypek, Jtmp, xyz # Claer the temporary variables at the end of the iteration
            # format annotations to keep or remove whitespace
            J, ind = format_white.format_white(J, Ig, WS, szz)
            from PIL import Image
            Image.fromarray(np.uint8(J)).save(os.path.join(outpth.rstrip('\\'), 'view_annotations_raw.tif')) # Save the image containing the annotations in a TIFF file
    except FileNotFoundError:
        J=np.zeros(I.size)
    return J

#Example usage
# if __name__ == "__main__":
#     import calculate_tissue_mask
#     outpth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests\data\84 - 2024-02-26 10.33.40'
#     umpix=2
#     imnm = '84 - 2024-02-26 10.33.40'
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests\5x'
#     WS = [[0,2,0,0,0,2,0],[6,7],[1,2,3,4,1,5,6],[7,2,4,3,1,6],[5]]
#     [I0,TA,_] = calculate_tissue_mask.calculate_tissue_mask(pth,imnm)
#     J0 = save_annotation_mask(I0,outpth,WS,umpix,TA)