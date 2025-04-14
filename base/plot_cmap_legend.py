import numpy as np
import matplotlib.pyplot as plt

def plot_cmap_legend(cmap, titles):
    """
               Plots the model colormap with the classnames in a figure .

               Parameters:
               - cmap (ndarray{n,3}): The color map of the model with one column per RGB value
               - titles (ndarray): Array of strings containing class names

               Returns:
               The figure displaying the model colormap.
        """
    im = np.zeros((50, 50 * len(cmap), 3), dtype=np.uint8)
    for k in range(len(cmap)):
        tmp = cmap[k].reshape(1, 1, 3)
        tmp = np.tile(tmp, (50, 50, 1))
        im[:, k * 50:(k + 1) * 50, :] = tmp

    if len(cmap) == len(titles) - 1:
        titles = titles[:-1]

    titles = [title.replace(' ', '_') for title in titles]

    if titles:
        im = np.rot90(im)
        plt.tick_params(axis='both', width=1)
        plt.imshow(im)
        plt.ylim(0, im.shape[0])
        plt.yticks(np.arange(25, im.shape[0], 50), labels=titles[::-1])
        plt.xticks([])
        plt.tick_params(axis='y', length=0)
        plt.tick_params(axis='both', labelsize=15)
        #plt.show() # Uncomment this line to display the plot. It might be unable save_model_metadata to save the plot.
    else:
        plt.figure()
        plt.imshow(im)