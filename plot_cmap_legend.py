import numpy as np
import matplotlib.pyplot as plt
def plot_cmap_legend(cmap, titles):
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
        plt.show()
    else:
        plt.figure()
        plt.imshow(im)

# Example usage:
#cmap = np.array([[121, 248, 252],
#                [0, 0, 255],
#                [80, 237, 80],
#                [255, 255, 0],
#                [149, 35, 184],
#                [25, 194, 245],
#                [255, 255, 255],
#                [255, 0, 0],
#                [73, 120, 111],
#                [0, 0, 0],
#                [135, 7, 7],
#                [240, 159, 10]])
#titles = ['islet', 'duct', 'blood vessel', 'fat','acini','ecm','whitespace','LG PanIN','nerves','immune','HG PanIN','PDAC']
#plot_cmap_legend(cmap, titles)
