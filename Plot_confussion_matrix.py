import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 26, 2024
"""

def plot_confusion_matrix(confusion_data, classNames, pthDL):
    # Ensure confusion_data is a numpy array
    confusion_data = np.array(confusion_data)

    # Calculate precision, recall, and accuracy
    precision = np.diag(confusion_data) / np.sum(confusion_data, axis=0)
    recall = np.diag(confusion_data) / np.sum(confusion_data, axis=1)
    accuracy = np.sum(np.diag(confusion_data)) / np.sum(confusion_data)

    # Round to one decimal place
    precision = np.round(precision * 100, 1)
    recall = np.round(recall * 100, 1)
    accuracy = np.round(accuracy * 100, 1)

    # Custom colormap for precision and recall
    colors = [(1, 0, 0), (1, 0.8, 0.8), (1, 1, 0.8), (0.8, 1, 0.8)]  # Light Red -> Light Yellow -> Light Green
    positions = [0, 0.6, 0.8, 1]  # Positions for 0%, 60%, 100%
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'red_yellow_green'
    cm = LinearSegmentedColormap.from_list(cmap_name, list(zip(positions, colors)), N=n_bins)

    # Normalize the data to the new range
    norm = plt.Normalize(60, 100)

    # Add precision column and recall row
    confusion_with_metrics = np.zeros((confusion_data.shape[0] + 1, confusion_data.shape[1] + 1))
    confusion_with_metrics[:-1, :-1] = confusion_data
    confusion_with_metrics[:-1, -1] = recall
    confusion_with_metrics[-1, :-1] = precision
    confusion_with_metrics[-1, -1] = accuracy

    # Set up the plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_with_metrics, annot=True, fmt='g', cmap='Blues',
                xticklabels=classNames + ['RECALL'],
                yticklabels=classNames + ['PRECISION'], cbar=False)


    # Plot the precision, recall, and accuracy values with the custom colormap
    mask = np.zeros_like(confusion_with_metrics, dtype=bool)
    mask[:-1, :-1] = True
    sns.heatmap(confusion_with_metrics, annot=True, fmt='g', cmap=cm, mask=mask,
                xticklabels=classNames + ['RECALL'], yticklabels=classNames + ['PRECISION'], cbar=False, norm=norm)

    # Add black bold border around the accuracy cell
    ax = plt.gca()
    rect = Rectangle((confusion_data.shape[1], confusion_data.shape[0]), 1, 1, fill=False, edgecolor='black', lw=3)
    ax.add_patch(rect)


    # Labels, title and ticks
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix', fontweight='bold')

    # Rotate the tick labels and set their alignment
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Move the x-axis label to the top
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()

    plt.tight_layout()
    plt.savefig(os.path.join(pthDL, 'confusion_matrix.jpg'))
    plt.show()

    print(f"\nConfusion matrix saved to {os.path.join(pthDL, 'confusion_matrix.jpg')}")

    print(f"\nOverall Accuracy: {accuracy}%")

    return confusion_with_metrics