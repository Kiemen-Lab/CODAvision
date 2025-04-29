"""
Semantic Segmentation Model Testing and Evaluation

This module provides functionality to test trained semantic segmentation models
and evaluate their performance using confusion matrices and accuracy metrics.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated March 2025
"""

import os
import numpy as np
from typing import Dict, Tuple, Any

from tifffile import imread
from skimage.morphology import remove_small_objects
from PIL import Image

from base.data.annotation import load_annotation_data
from base.image.classification import classify_images
from base.evaluation.confusion_matrix import ConfusionMatrixVisualizer
from base.models.utils import get_model_paths
from base.data.loaders import load_model_metadata

import warnings

warnings.filterwarnings("ignore")


class SegmentationModelTester:
    """
    Class for testing and evaluating semantic segmentation models.

    This class handles the testing of trained segmentation models against
    a test dataset, calculating performance metrics, and generating
    confusion matrices for evaluation.
    """

    def __init__(self, model_path: str, test_annotation_path: str, test_image_path: str):
        """
        Initialize the SegmentationModelTester with paths to model and test data.

        Args:
            model_path: Path to the directory containing model data
            test_annotation_path: Path to the directory containing test annotations
            test_image_path: Path to the directory containing test images
        """
        self.model_path = model_path
        self.test_annotation_path = test_annotation_path
        self.test_image_path = test_image_path

        self.model_data = None
        self.nblack = None
        self.nwhite = None
        self.class_names = None
        self.model_type = None
        self.model_paths = None

        self._load_model_data()

    def _load_model_data(self) -> None:
        """
        Load model metadata from the pickle file.

        Raises:
            FileNotFoundError: If the model data file doesn't exist
            ValueError: If essential parameters are missing
        """
        try:
            self.model_data = load_model_metadata(self.model_path)

            self.nblack = self.model_data.get('nblack')
            self.nwhite = self.model_data.get('nwhite')
            self.class_names = self.model_data.get('classNames')
            self.model_type = self.model_data.get('model_type', "DeepLabV3_plus")

            # Get standard paths for model files
            self.model_paths = get_model_paths(self.model_path, self.model_type)

        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")

    def read_image_as_double(self, file_path: str) -> np.ndarray:
        """
        Read an image file and convert it to a double-precision numpy array.

        Args:
            file_path: Path to the image file

        Returns:
            Numpy array representation of the image as double-precision values

        Raises:
            RuntimeError: If there's an error reading the image file
        """
        try:
            img = Image.open(file_path)
            img = img.convert('L')
            return np.array(img).astype(np.double)
        except Exception as e:
            raise RuntimeError(f"Error reading image file {file_path}: {e}")

    def prepare_test_data(self) -> str:
        """
        Prepare test data by loading annotations and classifying test images.

        Returns:
            Path to the directory containing classified test images

        Raises:
            ValueError: If preparation fails
        """
        try:
            # Load test annotation data
            test_data_path = os.path.join(self.test_annotation_path, 'data py')

            # Load annotation data
            load_annotation_data(self.model_path, self.test_annotation_path, self.test_image_path,0, True)

            # Classify test images
            classified_path = classify_images(
                self.test_image_path,
                self.model_path,
                self.model_type,
                color_overlay_HE=True,
                color_mask=False
            )

            return classified_path

        except Exception as e:
            raise ValueError(f"Failed to prepare test data: {e}")

    def collect_prediction_data(self, classified_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect ground truth and prediction data from test images.

        Args:
            classified_path: Path to the directory containing classified test images

        Returns:
            Tuple of (ground truth labels, predicted labels)

        Raises:
            ValueError: If data collection fails
        """
        true_labels = []
        predicted_labels = []

        test_data_path = os.path.join(self.test_annotation_path, 'data py')

        # Process each annotation folder
        for folder in os.listdir(test_data_path):
            annotation_path = os.path.join(test_data_path, folder)
            annotation_file_png = os.path.join(annotation_path, 'view_annotations.png')
            annotation_file_raw_png = os.path.join(annotation_path, 'view_annotations_raw.png')

            # Check if annotation file exists
            if os.path.exists(annotation_file_png) or os.path.exists(annotation_file_raw_png):
                try:
                    # Load ground truth
                    if os.path.exists(annotation_file_png):
                        ground_truth = self.read_image_as_double(annotation_file_png)
                    else:
                        ground_truth = self.read_image_as_double(annotation_file_raw_png)
                except RuntimeError as e:
                    print(e)
                    continue

                # Load prediction
                try:
                    prediction = imread(os.path.join(classified_path, folder + '.tif'))
                    prediction_array = np.array(prediction)
                except Exception as e:
                    print(f"Error loading prediction for {folder}: {e}")
                    continue

                # Clean small objects from ground truth
                for label in range(0, int(ground_truth.max())):
                    mask = ground_truth == label
                    ground_truth[mask] = 0
                    cleaned_mask = remove_small_objects(mask.astype(bool), min_size=25, connectivity=2)
                    ground_truth[cleaned_mask] = label

                # Extract non-zero pixels
                non_zero_indices = np.where(ground_truth > 0)

                if len(non_zero_indices[0]) > 0:
                    true_labels.extend(ground_truth[non_zero_indices])
                    predicted_labels.extend(prediction_array[non_zero_indices])

        if not true_labels or not predicted_labels:
            raise ValueError("No valid annotation data found in test dataset")

        return np.array(true_labels), np.array(predicted_labels)

    def analyze_class_distribution(self, true_labels: np.ndarray) -> np.ndarray:
        """
        Analyze the distribution of classes in the ground truth labels.

        Args:
            true_labels: Array of ground truth labels

        Returns:
            Array of class counts

        Raises:
            ValueError: If there are missing classes in the test dataset
        """
        # Get class names excluding last one (usually background)
        class_names = self.class_names[:-1]
        num_classes = len(class_names)

        # Count labels
        label_counts = np.histogram(true_labels, bins=num_classes)[0]
        label_percentages = (label_counts / label_counts.max() * 100).astype(int)

        # Display statistics
        print('\nCalculating total number of pixels in the testing dataset...')
        for i, count in enumerate(label_counts):
            if label_percentages[i] == 100:
                print(f"  There are {count} pixels of {class_names[i]}. This is the most common class.")
            else:
                print(
                    f"  There are {count} pixels of {class_names[i]}, {label_percentages[i]}% of the most common class.")

        # Check for missing classes
        if 0 in label_counts:
            for i, count in enumerate(label_counts):
                if count == 0:
                    print(f"\n No testing annotations exist for class {class_names[i]}.")
            raise ValueError("Cannot make confusion matrix. Please add testing annotations of missing class(es).")

        # Check for insufficient annotations
        min_recommended_pixels = 15000
        for i, count in enumerate(label_counts):
            if count < min_recommended_pixels:
                print(f"\n  Only {count} testing pixels of {class_names[i]} found.")
                print("    We suggest a minimum of 15,000 pixels for a good assessment of model accuracy.")
                print("    Confusion matrix may be misleading.")

        return label_counts

    def balance_samples(self, true_labels: np.ndarray, predicted_labels: np.ndarray,
                        label_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the sample sizes across classes for fair evaluation.

        Args:
            true_labels: Array of ground truth labels
            predicted_labels: Array of predicted labels
            label_counts: Array of class counts

        Returns:
            Tuple of (balanced true labels, balanced predicted labels)
        """
        # Calculate minimum count for balanced sampling
        min_count = label_counts.min()
        if min_count < 100:
            min_count = min_count * 10
        elif min_count < 1000:
            min_count = min_count * 3
        else:
            min_count = min_count

        # Create balanced datasets
        balanced_true_labels = []
        balanced_predicted_labels = []

        for label in np.unique(true_labels):
            indices = np.where(np.array(true_labels) == label)[0]
            # Ensure min_count is not larger than the population size
            if min_count > len(indices):
                min_count = len(indices)
            selected_indices = np.random.choice(indices, min_count, replace=False)
            balanced_true_labels.extend(np.array(true_labels)[selected_indices])
            balanced_predicted_labels.extend(np.array(predicted_labels)[selected_indices])

        return np.array(balanced_true_labels), np.array(balanced_predicted_labels)

    def create_confusion_matrix(self, balanced_true: np.ndarray,
                                balanced_predicted: np.ndarray) -> np.ndarray:
        """
        Create a confusion matrix from the balanced true and predicted labels.

        Args:
            balanced_true: Balanced array of ground truth labels
            balanced_predicted: Balanced array of predicted labels

        Returns:
            Confusion matrix as a numpy array
        """
        # Process predictions
        processed_predictions = balanced_predicted.copy()
        processed_predictions[processed_predictions == self.nblack] = self.nwhite

        # Create confusion matrix
        max_true = int(np.max(balanced_true))
        max_pred = int(np.max(processed_predictions))
        confusion_data = np.zeros((max_true, max_true))

        for true_label in range(1, max_true + 1):
            for pred_label in range(1, max_pred + 1):
                confusion_data[true_label - 1, pred_label - 1] = np.sum(
                    (balanced_true == true_label) &
                    (processed_predictions == pred_label)
                )

        # Handle invalid values
        confusion_data[np.isnan(confusion_data)] = 0

        return confusion_data

    def test(self) -> Dict[str, Any]:
        """
        Test the segmentation model and evaluate its performance.

        Returns:
            Dictionary containing confusion matrix and performance metrics

        Raises:
            ValueError: If testing fails
        """
        print("Testing segmentation model......")

        try:
            # Prepare test data
            classified_path = self.prepare_test_data()

            # Collect prediction data
            true_labels, predicted_labels = self.collect_prediction_data(classified_path)

            # Analyze class distribution
            label_counts = self.analyze_class_distribution(true_labels)

            # Balance samples for fair evaluation
            balanced_true, balanced_predicted = self.balance_samples(
                true_labels, predicted_labels, label_counts
            )

            # Create confusion matrix
            confusion_matrix = self.create_confusion_matrix(balanced_true, balanced_predicted)

            # Visualize confusion matrix
            class_names = self.class_names[:-1]  # Exclude background class
            visualizer = ConfusionMatrixVisualizer(
                class_names=class_names,
                output_dir=self.model_path,
                model_name=self.model_type
            )

            confusion_with_metrics = visualizer.visualize(confusion_matrix)

            # Return results
            metrics = {
                'confusion_matrix': confusion_matrix,
                'confusion_with_metrics': confusion_with_metrics
            }

            return metrics

        except Exception as e:
            raise ValueError(f"Model testing failed: {e}")


def test_segmentation_model(pthDL: str, pthtest: str, pthtestim: str) -> None:
    """
    Test a segmentation model with the provided paths.

    This function serves as a compatibility wrapper around the SegmentationModelTester class,
    preserving the original function signature for backward compatibility.

    Args:
        pthDL: Path to the directory containing model data
        pthtest: Path to the directory containing test annotations
        pthtestim: Path to the directory containing test images
    """
    tester = SegmentationModelTester(pthDL, pthtest, pthtestim)
    tester.test()
    return