import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from Semanticseg import read_image, semantic_seg

def plot_confusion_matrix(test_images_path, test_masks_path, classNames, image_size, model):
    """
    Plot the confusion matrix for the semantic segmentation model.

    Inputs:
    - test_images_path (list of str): Paths to the test image files.
    - test_masks_path (list of str): Paths to the ground truth mask files.
    - class_names (list of str): List of class names.
    - image_size (int): Size to which the input images should be resized, it should be the same as the training tile size
    - model (tf.keras.Model): Pre-trained TensorFlow/Keras model for semantic segmentation.

    Outputs:
    - Confusion matrix for the semantic segmentation model

    """
    y_true = []
    y_pred = []

    # Read images and masks, perform segmentation, and collect predictions and true labels
    for img_path, mask_path in zip(test_images_path, test_masks_path):
        image_tensor = read_image(img_path, image_size)
        mask_tensor = read_image(mask_path, image_size, mask=True)

        if image_tensor is not None and mask_tensor is not None:
            pred_mask = semantic_seg(img_path, image_size, model)
            true_mask = tf.squeeze(mask_tensor).numpy().astype(int)

            y_true.extend(true_mask.flatten())
            y_pred.extend(pred_mask.flatten())
        else:
            print(f"Error processing image or mask at {img_path} or {mask_path}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classNames)))

    # # Calculate precission, recall and overall accuracy
    # precision = []
    # recall = []
    # total_correct = np.sum(np.diag(cm))  # Calculate total correct predictions
    # total_elements = np.sum(cm)  # Calculate total elements in the matrix (all predictions)
    # overall_accuracy = total_correct / total_elements
    #
    # for i in range(len(classNames)):
    #     tp = cm[i, i]
    #     fp = cm[:, i].sum() - cm[i, i]
    #     fn = cm[i, :].sum() - cm[i, i]
    #     precision.append(tp / (tp + fp))
    #     recall.append(tp / (tp + fn))


    #  Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')

    # # Add precission and recall
    # plt.xticks(np.arange(len(classNames)) + 0.5, classNames, rotation=45,
    #            ha='right')  # Adjust x-axis for additional columns
    # plt.sca(disp.ax_)  # Add columns to existing axis
    # plt.barh(classNames, precision, color='skyblue', label='Precision')
    # plt.barh(classNames, recall, color='coral', label='Recall')
    # plt.legend(loc='upper left')
    #
    # # Add overall accuracy text at bottom right corner
    # plt.text(len(cm) + 0.2, 0.05, f"Overall Accuracy: {overall_accuracy:.2f}",
    #          ha='right', va='bottom', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.show()

# # Example usage:
# if __name__ == '__main__':
#     test_images_path = ['path_to_image1.png', 'path_to_image2.png', ...]
#     test_masks_path = ['path_to_mask1.png', 'path_to_mask2.png', ...]
#     class_names = ['class1', 'class2', 'class3', ...]
#     image_size = 1024  # Example image size
#     model = ...  # Your pre-trained model
#
#     plot_confusion_matrix(test_images_path, test_masks_path, class_names, image_size, model)
