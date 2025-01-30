import numpy as np
import tensorflow as tf


def calculate_class_weights(label_paths, class_names):
    """
    Calculate class weights based on pixel frequency in labels

    Args:
        label_paths (list): List of paths to label images
        class_names (list): List of class names

    Returns:
        dict: Class weights dictionary
    """
    # Initialize counters
    pixel_counts = {name: 0 for name in class_names}
    total_pixels = {name: 0 for name in class_names}

    # Count pixels for each class
    for label_path in label_paths:
        label = np.array(tf.keras.preprocessing.image.load_img(
            label_path, color_mode='grayscale'))

        unique, counts = np.unique(label, return_counts=True)
        total_image_pixels = label.size

        for val, count in zip(unique, counts):
            class_name = class_names[val]
            pixel_counts[class_name] += count
            total_pixels[class_name] += total_image_pixels

    # Calculate frequency for each class
    image_freq = {name: pixel_counts[name] / total_pixels[name]
                  for name in class_names}

    # Calculate class weights using median frequency balancing
    median_freq = np.median(list(image_freq.values()))
    class_weights = {name: median_freq / freq for name, freq in image_freq.items()}

    print("Class frequencies:")
    for name, freq in image_freq.items():
        print(f"{name}: {freq:.4f}")

    print("\nClass weights:")
    for name, weight in class_weights.items():
        print(f"{name}: {weight:.4f}")

    class_weights = list(class_weights.values())

    return class_weights

from glob import glob
import os
pthTrain = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_01_06_2024_sametilesas12_03_2024_notGUI\training'
train_masks = sorted(glob(os.path.join(pthTrain, 'label', "*.png")))
classNames = ['PDAC','bile_duct','vasculature','hepatocyte','immune','stroma','whitespace','black']
calculate_class_weights(train_masks,classNames)