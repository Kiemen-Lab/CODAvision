"""
PDF Report Generation Module Tests for CODAvision

This module provides test cases for verifying the functionality of the PDF report
generation module, ensuring proper creation of evaluation reports with expected
sections including confusion matrices, annotations, classifications, and metrics.

"""

import pytest
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import PyPDF2
from unittest.mock import patch, MagicMock

from base.evaluation.pdf_report import create_output_pdf

# Helper function to create dummy image files
def create_dummy_image(path: Path, width: int = 10, height: int = 10):
    """Creates a simple dummy PNG image file."""
    img = Image.new('RGB', (width, height), color = 'red')
    img.save(path, 'PNG')

def create_dummy_jpg(path: Path, width: int = 10, height: int = 10):
    """Creates a simple dummy JPG image file."""
    img = Image.new('RGB', (width, height), color = 'blue')
    # Save as JPEG, ensuring the path ends with .jpg for the test logic
    jpg_path = path.with_suffix('.jpg')
    img.save(jpg_path, 'JPEG')
    return jpg_path

# Helper function to create dummy CSV file
def create_dummy_csv(path: Path):
    """Creates a dummy quantification CSV file."""
    data = {
        'Image name': ['image1.tif', 'image2.tif'],
        'ClassA PC': [1000, 1500],
        'ClassB PC': [2000, 2500],
        'Whitespace PC': [500, 600],
        'ClassA TC(%)': [33.33, 37.5],
        'ClassB TC(%)': [66.67, 62.5],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

# Helper function to create dummy net.pkl file
def create_dummy_pkl(path: Path, model_name: str = "TestModel"):
    """Creates a dummy net.pkl file."""
    data = {'nm': model_name,
            # Add other keys if create_output_pdf starts depending on them
            'WS': [[0,0],[1,2],[1,2],[2,1],[]], # Dummy WS data
            'cmap': np.array([[255,0,0],[0,255,0]]), # Dummy cmap
            'classNames': ['ClassA', 'ClassB', 'Whitespace'], # Dummy classNames
            'nwhite': 3, # Assuming whitespace is class 3
            'umpix': 1,
            'sxy': 1024,
            'nblack': 4,
            'ntrain': 10,
            'nvalidate': 2,
            'final_df': pd.DataFrame(),
            'combined_df': pd.DataFrame(),
            'model_type': 'TestNet',
            'batch_size': 4,
            'nTA': 3,
           }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# --- The Test Function ---
def test_create_output_pdf_generates_valid_pdf(tmp_path):
    """
    Tests if create_output_pdf runs and generates a readable PDF
    with expected section titles.
    """
    # Set up mock files and directories
    model_dir = tmp_path / "model_output"
    model_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    annotations_check_dir = image_dir / "check_annotations"
    annotations_check_dir.mkdir()
    classification_check_dir = image_dir / "check_classification"
    classification_check_dir.mkdir()
    output_dir = tmp_path / "report_output"
    output_dir.mkdir()

    # Create dummy files
    net_pkl_path = model_dir / "net.pkl"
    create_dummy_pkl(net_pkl_path, model_name="MyBestModel")

    # Create a JPG confusion matrix to test conversion
    confusion_matrix_path_jpg = model_dir / "confusion_matrix.jpg"
    confusion_matrix_path_jpg = create_dummy_jpg(confusion_matrix_path_jpg)
    # The function expects the *original* path, it handles conversion internally
    confusion_matrix_input_path = str(confusion_matrix_path_jpg)

    color_legend_path = model_dir / "color_legend.png"
    create_dummy_image(color_legend_path)

    quantifications_csv_path = model_dir / "quantifications.csv"
    create_dummy_csv(quantifications_csv_path)

    annotation_check_img_path = annotations_check_dir / "anno_check_01.png"
    create_dummy_image(annotation_check_img_path)

    classification_check_img_path = classification_check_dir / "classif_check_01.png"
    create_dummy_image(classification_check_img_path)

    # Define the expected output PDF path
    output_pdf_path = output_dir / "model_evaluation_report.pdf"

    # Define dummy times
    times = {
        'Data Loading': '0:01:15.123',
        'Training': '1:30:05.456',
        'Classification': '0:15:20.789',
        'Total': '1:46:41.368'
    }

    # Define paths for the function call
    pthDL_str = str(model_dir)
    color_legend_path_str = str(color_legend_path)
    check_annotations_path_str = str(annotations_check_dir)
    check_classification_path_str = str(classification_check_dir)
    quantifications_csv_path_str = str(quantifications_csv_path)
    output_pdf_path_str = str(output_pdf_path)


    # Call the function under test
    try:
        create_output_pdf(
            output_path=output_pdf_path_str,
            pthDL=pthDL_str,
            confusion_matrix_path=confusion_matrix_input_path, # Use the original JPG path
            color_legend_path=color_legend_path_str,
            check_annotations_path=check_annotations_path_str,
            check_classification_path=check_classification_path_str,
            quantifications_csv_path=quantifications_csv_path_str,
            times=times
        )
    except Exception as e:
        pytest.fail(f"create_output_pdf raised an exception: {e}")

    # Basic check: Does the PDF exist?
    assert output_pdf_path.exists(), "Output PDF file was not created."
    assert output_pdf_path.is_file(), "Output path is not a file."
    assert output_pdf_path.stat().st_size > 0, "Output PDF file is empty."

    # More advanced check: Is the PDF readable and contains key sections?
    try:
        with open(output_pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            assert len(reader.pages) > 5, f"Expected more than 5 pages, found {len(reader.pages)}" # Check page count

            # Extract text from first few pages to check for headers
            extracted_text = ""
            for i in range(min(7, len(reader.pages))): # Read first 7 pages or fewer
                 page = reader.pages[i]
                 extracted_text += page.extract_text()

            print("\nExtracted text from PDF (first few pages):\n---")
            print(extracted_text)
            print("---")


            # Check for expected section titles
            expected_titles = [
                "Confusion Matrix",
                "Color Legend",
                "Check Annotations",
                "Check Classification",
                "Pixel and Tissue Composition Quantifications",
                "Runtimes",
                "Annex",
                "Understanding the Confusion Matrix" # From Annex
            ]
            for title in expected_titles:
                assert title in extracted_text, f"Expected section title '{title}' not found in PDF text."

            # Check if model name from pkl is in the header
            assert "MyBestModel" in extracted_text, "Model name from pkl not found in PDF header text."

            # Check if a runtime is present
            assert "Data Loading" in extracted_text, "Runtime section name not found."
            assert "00:01:15.123" in extracted_text, "Runtime value not found/formatted incorrectly."

            # Check if a quantification header is present
            assert "ClassA PC" in extracted_text, "Quantification header 'ClassA PC' not found."
            assert "ClassB TC(%)" in extracted_text, "Quantification header 'ClassB TC(%)' not found."


    except PyPDF2.errors.PdfReadError as e:
        pytest.fail(f"Generated PDF is invalid or unreadable: {e}")
    except Exception as e:
         pytest.fail(f"An error occurred during PDF content verification: {e}")