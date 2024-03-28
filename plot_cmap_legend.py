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
        #plt.figure(figsize=(2, 0.5 * (len(cmap))))
        plt.gca().set_xlim([0,5])
        plt.tick_params(axis='both', width=1)
        plt.imshow(im)
        plt.axis('equal')
        #plt.xlim(0, im.shape[1])
        plt.ylim(0, im.shape[0])
        plt.yticks(np.arange(15, im.shape[0], 50), labels=titles[::-1])
        plt.xticks([])
        plt.tick_params(axis='y', length=0)
        plt.tick_params(axis='both', labelsize=15)
        plt.show()
    else:
        plt.figure()
        plt.imshow(im)

# Example usage:
cmap = np.array([[255, 0, 0],
                [0, 77, 120],
                [255, 120, 0]])
titles = ['Title 1', 'Title 2', 'Title 3', 'Title 4']
plot_cmap_legend(cmap, titles)