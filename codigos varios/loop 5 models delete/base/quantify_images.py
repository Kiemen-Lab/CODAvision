import pickle
import numpy as np
import os
import pandas as pd
import cv2

def quantify_images(pthDL, pthim):
    """
       Quantifies tissue composition in images and saves the results to a CSV file.

       Parameters:
       pthDL (str): Path to the directory containing the 'net.pkl' file with the model metadata.
       pthim (str): Path to the directory containing the images to be quantified.

       Outputs:
       A CSV file named 'image_quantifications.csv' containing the pixel counts and tissue composition percentages
       for each image. Saved in the same directory as the classified images.
       """

    print('Quantifying images...')
    # Load data
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        classNames = data['classNames']
        nwhite = data['nwhite']
        nm = data['nm']

    # Define the headers for the pixel count and tissue composition quantifications
    classQuantification = ['Image name'] + \
                          [f'{className} pixel count' for className in classNames[:-1]] + \
                          [f"{classNames[i]} tissue composition (%)" for i in range(len(classNames) - 1) if i != nwhite - 1]

    # Save headers to CSV
    quantPath = os.path.join(pthim, 'classification_' + nm)
    file = os.path.join(quantPath, 'image_quantifications.csv')
    df = pd.DataFrame(columns=classQuantification)
    df.to_csv(file, index=False)

    # Process images
    files = [f for f in os.listdir(quantPath) if f.endswith('.tif')]
    numfiles = len(files)

    for j, imageName in enumerate(files):
        print(f"Image {j + 1} / {numfiles}: {imageName}")
        im = cv2.imread(os.path.join(quantPath, imageName), cv2.IMREAD_GRAYSCALE)
        tissue = im != nwhite
        tissuepixels = np.sum(tissue)

        classCounts = [np.sum(im == i + 1) for i in range(len(classNames) - 1)]
        tissueCompositions = [(count / tissuepixels * 100) if i + 1 != nwhite else 0 for i, count in enumerate(classCounts)]

        imageData = [imageName] + classCounts + [comp for i, comp in enumerate(tissueCompositions) if i + 1 != nwhite]

        df = pd.DataFrame([imageData], columns=classQuantification)
        df.to_csv(file, mode='a', header=False, index=False)

    # Write additional information to CSV
    additional_info = pd.DataFrame([['Model name:', pthDL], ['File location:', quantPath]])
    additional_info.to_csv(file, mode='a', header=False, index=False)

# Example usage
if __name__ == '__main__':
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024'
    pthim = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x'
    quantify_images(pthDL, pthim)