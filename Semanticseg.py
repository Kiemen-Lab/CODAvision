# https://keras.io/examples/vision/deeplabv3_plus/

"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 11, 2024
"""
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import numpy as np


def read_image(image_path, image_size, mask=False):
    try:
        image = tf_io.read_file(image_path)
        if mask:
            image = tf_image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf_image.resize(images=image, size=[image_size, image_size])
        else:
            image = tf_image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf_image.resize(images=image, size=[image_size, image_size])
        return image
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims(image_tensor, axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def semantic_seg(image_path, image_size, model):
    """
        Perform semantic segmentation on an input image using a pre-trained model.

        Inputs:
        - image_path (str): Path to the input image file.
        - image_size (int): Size to which the input image should be resized.
        - model (tf.keras.Model): Pre-trained TensorFlow/Keras model for semantic segmentation.

        Outputs:
        - np.ndarray: 2D array representing the segmentation mask, where each pixel value corresponds to a class label.
                    Raises an assertion error if there is an error reading the image.
    """
    image_tensor = read_image(image_path, image_size)
    assert image_tensor is not None, f"Error: Could not read the image from path {image_path}"
    prediction_mask = infer(model, image_tensor)
    return prediction_mask


# Example usage

# if __name__ == '__main__':
#     image_path = 'path_to_your_image.png'  # only PNG IMAGES!!!!!
#     prediction_mask = semantic_seg(image_path, image_size=1024, model)
#     if prediction_mask is not None:
#         print(prediction_mask.shape)
#     else:
#         print("Prediction failed due to image read error.")