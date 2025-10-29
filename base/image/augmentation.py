"""
Image Augmentation Utilities for CODAvision

This module provides functions for augmenting images and masks during training to
improve model generalization through data diversity.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, dilation
import logging

# Set up logging
logger = logging.getLogger(__name__)


def augment_image(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    rotation: bool = True,
    scaling: bool = True,
    hue_shift: bool = True,
    blur: bool = False,
    resize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to an image and its mask.

    This function performs a series of data augmentation techniques on the input
    image and its corresponding mask to increase the diversity of training data.

    Args:
        image: Input image to augment (H&E stained image)
        mask: Optional mask corresponding to the input image. If None, the image is used as the mask
        rotation: Whether to apply random rotation
        scaling: Whether to apply random scaling
        hue_shift: Whether to apply random hue adjustment
        blur: Whether to apply random Gaussian blur
        resize: Whether to resize images after scaling augmentation

    Returns:
        Tuple containing:
        - Augmented image
        - Augmented mask
    """
    if mask is None:
        mask = image.copy()

    augmented_image = image.astype(np.float64)
    augmented_mask = mask.astype(np.float64)
    original_size = image.shape[0]

    # Rotation augmentation
    if rotation:
        # Choose a random angle from 0 to 355 in steps of 5 degrees
        angles = np.arange(0, 360, 5)
        angle = np.random.choice(angles)
        height, width = augmented_image.shape[:2]

        # Store original dimensions for cropping (MATLAB behavior)
        original_height, original_width = height, width

        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Calculate new dimensions after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)

        rotation_matrix[0, 2] += (new_width / 2) - width / 2
        rotation_matrix[1, 2] += (new_height / 2) - height / 2

        # Apply rotation
        augmented_image = cv2.warpAffine(
            augmented_image,
            rotation_matrix,
            (new_width, new_height),
            borderValue=(0, 0, 0),
            flags=cv2.INTER_NEAREST
        )

        augmented_mask = cv2.warpAffine(
            augmented_mask,
            rotation_matrix,
            (new_width, new_height),
            borderValue=(0, 0, 0),
            flags=cv2.INTER_NEAREST
        )

        # Crop back to original size to match MATLAB's imrotate behavior
        if new_height != original_height or new_width != original_width:
            # Calculate center crop coordinates
            center_y = new_height // 2
            center_x = new_width // 2
            half_height = original_height // 2
            half_width = original_width // 2

            # Crop to original dimensions
            y_start = max(0, center_y - half_height)
            y_end = min(new_height, center_y + half_height)
            x_start = max(0, center_x - half_width)
            x_end = min(new_width, center_x + half_width)

            augmented_image = augmented_image[y_start:y_end, x_start:x_end, :]
            augmented_mask = augmented_mask[y_start:y_end, x_start:x_end]

            # Ensure exact original size (handle odd dimensions)
            if augmented_image.shape[0] != original_height or augmented_image.shape[1] != original_width:
                augmented_image = cv2.resize(augmented_image, (original_width, original_height))
                augmented_mask = cv2.resize(augmented_mask, (original_width, original_height),
                                           interpolation=cv2.INTER_NEAREST)

    # Scaling augmentation
    if scaling:
        # Match MATLAB exactly: [0.6:0.01:0.95, 1.1:0.01:1.4]
        # Using explicit upper bounds (0.951, 1.401) to ensure endpoints are included
        # and guard against floating-point edge cases
        scales = np.concatenate((
            np.arange(0.6, 0.951, 0.01),  # Include 0.95 (numpy is exclusive on upper bound)
            np.arange(1.1, 1.401, 0.01)   # Include 1.4 (numpy is exclusive on upper bound)
        ))
        scale_factor = np.random.choice(scales)

        augmented_image = cv2.resize(
            augmented_image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )

        augmented_mask = cv2.resize(
            augmented_mask,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )

    # Hue augmentation (color shifting)
    if hue_shift:
        # Random factors for each channel, either reducing or increasing intensity
        factors = np.concatenate((np.arange(0.88, 0.98, 0.01), np.arange(1.02, 1.12, 0.01)))

        # Apply different random factors to each channel
        for channel in range(3):
            channel_factor = np.random.choice(factors)
            channel_values = 255 - augmented_image[:, :, channel]
            augmented_image[:, :, channel] = 255 - (channel_values * channel_factor)

    # Blur augmentation
    if blur:
        blur_levels = np.ones(50)
        blur_levels[0:4] = [1.05, 1.1, 1.15, 1.2]

        blur_level = np.random.choice(blur_levels)
        if blur_level != 1:
            augmented_image = gaussian_filter(augmented_image, sigma=blur_level)

    # Resize back to original size if needed
    if resize:
        current_height = augmented_image.shape[0]

        if current_height > original_size:
            # Crop to original size
            center = int(np.round(current_height / 2))
            half_size = int((original_size - 1) / 2)
            remainder = int(np.ceil((original_size - 1) / 2))

            augmented_image = augmented_image[
                center - half_size:center + remainder,
                center - half_size:center + remainder,
                :
            ]

            augmented_mask = augmented_mask[
                center - half_size:center + remainder,
                center - half_size:center + remainder
            ]

        elif current_height < original_size:
            # Pad to original size
            padding = original_size - current_height
            augmented_image = np.pad(
                augmented_image,
                ((0, padding), (0, padding), (0, 0)),
                mode='constant',
                constant_values=0
            )

            augmented_mask = np.pad(
                augmented_mask,
                ((0, padding), (0, padding)),
                mode='constant',
                constant_values=0
            )

    # Zero out pixels outside the mask
    mask_indices = (augmented_mask != 0)
    mask_3d = np.dstack((mask_indices, mask_indices, mask_indices))
    augmented_image = augmented_image * mask_3d

    return augmented_image, augmented_mask


# For backward compatibility
augment_annotation = augment_image


def edit_annotation_tiles(
    im: np.ndarray,
    TA: np.ndarray,
    do_augmentation: bool,
    class_id: int,
    num_pixels_class: np.ndarray,
    big_tile_size: int,
    kpall: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Edit annotation tiles by performing augmentation and adjusting class distribution.

    This function applies augmentation to images and their corresponding annotation masks,
    then performs filtering and morphological operations to adjust class distribution.

    Args:
        im: Input image
        TA: Input label mask
        do_augmentation: Flag indicating whether to perform augmentation
        class_id: ID of the class to adjust distribution for
        num_pixels_class: Array containing the pixel counts for each class
        big_tile_size: Size of the big tile
        kpall: Flag indicating whether to keep all classes (1) or not (0)

    Returns:
        Tuple containing:
        - Augmented and adjusted image
        - Augmented and adjusted label mask
        - Unique labels after processing
    """
    logger.debug(f"edit_annotation_tiles for class_id {class_id}")
    logger.debug(f"num_pixels_class shape: {num_pixels_class.shape}")

    # Apply appropriate augmentation based on the flag
    if do_augmentation:
        im, TA = augment_image(im, TA, rotation=True, scaling=True, hue_shift=True, blur=True)
    else:
        im, TA = augment_image(im, TA, rotation=True, scaling=True, hue_shift=False, blur=False)

    # Filter classes based on kpall flag
    if kpall == 0:
        maxn = num_pixels_class[class_id]
        kp = num_pixels_class <= maxn * 1.05
    else:
        kp = num_pixels_class >= 0

    # Extend kp to match TA classes (adding a 0 index)
    kp = np.concatenate(([0], kp))
    tmp = kp[TA.astype(int)]
    tmp = tmp > 0

    # Dilate the mask
    dil = np.random.randint(15) + 15
    tmp = dilation(tmp, disk(dil))

    # Apply the mask to TA and im
    TA = TA * tmp
    for i in range(im.shape[2]):
        im[:, :, i] *= tmp

    # Get unique labels in the masked TA
    kpout = np.unique(TA)[1:].astype(int)

    # Crop to big_tile_size
    p1, p2 = min(big_tile_size, TA.shape[0]), min(big_tile_size, TA.shape[1])
    im = im[0:p1, 0:p2, :]
    TA = TA[0:p1, 0:p2]

    # Convert TA to uint8
    TA = TA.astype(np.uint8)

    logger.debug(f"Returning kpout: {kpout}")

    return im, TA, kpout