"""
Model Evaluation Report Generation

This module provides functionality for generating PDF reports that visualize
the results of model evaluation, including confusion matrices, color legends,
annotations, classifications, and quantitative metrics.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
from typing import Dict, Optional, Union, List, Tuple, Any
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF


class ModelReportPDF(FPDF):
    """
    Custom PDF class for generating model evaluation reports.
    
    This class extends FPDF to provide specialized formatting and layout
    for model evaluation reports, including headers, formatting for different
    content types, and image handling.
    
    Attributes:
        model_name (str): Name of the model being evaluated
    """
    
    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initialize the PDF with model name and formatting.
        
        Args:
            model_name: Name of the model to display in the header
            *args: Additional arguments for FPDF
            **kwargs: Additional keyword arguments for FPDF
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def header(self):
        """Add a header to each page with the model name."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'Performance report for model: {self.model_name}', 0, 1, 'C')

    def chapter_title(self, title: str):
        """
        Add a formatted chapter title.
        
        Args:
            title: Chapter title text
        """
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')

    def chapter_body(self, body: str):
        """
        Add formatted body text.
        
        Args:
            body: Body text content
        """
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body, align='L')

    def path_bold(self, path: str):
        """
        Add a path with bold formatting.
        
        Args:
            path: File path to display
        """
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 6, path, align='L')
        self.ln()

    def underlined_body(self, body: str):
        """
        Add underlined body text.
        
        Args:
            body: Text content to underline
        """
        self.set_font('Arial', 'U', 10)
        self.multi_cell(0, 6, body, align='L')

    def add_image(self, image_path: str, x: float, y: float, w: float, h: float = 0):
        """
        Add an image to the PDF at specified coordinates.
        
        Args:
            image_path: Path to the image file
            x: X-coordinate for image placement
            y: Y-coordinate for image placement
            w: Width of the image
            h: Height of the image (0 for automatic scaling)
            
        Raises:
            RuntimeError: If the image format is not supported
        """
        supported_extensions = ['jpg', 'jpeg', 'png']
        file_extension = image_path.split('.')[-1].lower()

        if file_extension not in supported_extensions:
            raise RuntimeError(f'Unsupported image type: {file_extension}')

        self.image(image_path, x, y, w, h)


def convert_to_png(image_path: str) -> str:
    """
    Convert a JPG image to PNG format.
    
    Args:
        image_path: Path to the JPG image
        
    Returns:
        Path to the converted PNG image
    """
    img = Image.open(image_path)
    png_path = image_path.replace('.jpg', '.png')
    img.save(png_path, 'PNG')
    return png_path


def get_first_image(directory: str, supported_extensions: List[str] = None) -> str:
    """
    Find the first image file in a directory with a supported extension.
    
    Args:
        directory: Directory to search for images
        supported_extensions: List of supported file extensions (default: ['jpg', 'jpeg', 'png'])
        
    Returns:
        Path to the first image found
        
    Raises:
        RuntimeError: If no supported image files are found
    """
    if supported_extensions is None:
        supported_extensions = ['jpg', 'jpeg', 'png']
        
    for file in os.listdir(directory):
        if file.split('.')[-1].lower() in supported_extensions:
            return os.path.join(directory, file)
            
    raise RuntimeError(f"No supported image files found in {directory}")


def fit_path_in_line(path: str, line_length: int = 100) -> str:
    """
    Format a long path with line breaks for better readability.
    
    Args:
        path: File path to format
        line_length: Maximum length of each line
        
    Returns:
        Formatted path with line breaks
    """
    if len(path) <= line_length:
        return path
        
    formatted_path = path
    for i in range(int(np.floor(len(path) / line_length)), 0, -1):
        separator_index = path.find('\\', (i-1) * line_length + 80)
        if separator_index != -1:
            formatted_path = formatted_path[:separator_index+1] + '\n' + formatted_path[separator_index+1:]
            
    return formatted_path


def create_evaluation_report(
    output_path: str,
    model_path: str,
    confusion_matrix_path: str, 
    color_legend_path: str,
    check_annotations_path: str,
    check_classification_path: str,
    quantifications_csv_path: str
) -> None:
    """
    Create a comprehensive PDF report visualizing model evaluation results.
    
    The report includes:
    1. Confusion matrix with accuracy metrics
    2. Color legend for annotation classes
    3. Sample annotated images
    4. Sample classified images
    5. Pixel and tissue composition quantifications
    6. Explanatory information about interpreting the results
    
    Args:
        output_path: Path where the PDF report will be saved
        model_path: Path to the directory containing model data
        confusion_matrix_path: Path to the confusion matrix image
        color_legend_path: Path to the color legend image
        check_annotations_path: Path to directory containing annotation check images
        check_classification_path: Path to directory containing classification check images
        quantifications_csv_path: Path to CSV file containing quantification results
        
    Raises:
        FileNotFoundError: If required files cannot be found
        ValueError: If data in the CSV file is invalid or missing
    """
    print('Generating model evaluation report...')

    # Load model data
    try:
        with open(os.path.join(model_path, 'net.pkl'), 'rb') as f:
            data = pickle.load(f)
        model_name = data.get('nm', 'Unknown Model')
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Could not load model data: {str(e)}")

    # Initialize PDF
    pdf = ModelReportPDF(model_name)
    pdf.add_page()

    # Set up formatting
    pdf.set_font('Arial', 'B', 16)
    
    # Section 1: Confusion Matrix
    pdf.chapter_title('1. Confusion Matrix')
    pdf.chapter_body('Confusion matrix, encompassing precision, recall, and accuracy scores.\nPath:')
    pdf.path_bold(f'{fit_path_in_line(confusion_matrix_path)}\n')
    
    # Convert JPG to PNG if needed for compatibility
    if confusion_matrix_path.lower().endswith('.jpg'):
        confusion_matrix_path = convert_to_png(confusion_matrix_path)

    # Add confusion matrix image
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.add_image(confusion_matrix_path, pdf.l_margin, 60, page_width, 0)

    # Section 2: Color Legend
    pdf.add_page()
    pdf.chapter_title('2. Color Legend')
    pdf.chapter_body(
        f'Color legend associated with the trained model situated in the same path as the confusion matrix.\nPath:\n'
    )
    pdf.path_bold(f'{fit_path_in_line(color_legend_path)}\n')
    pdf.add_image(color_legend_path, 10, 60, 100, 0)

    # Section 3: Check Annotations
    pdf.add_page()
    pdf.chapter_title('3. Check Annotations')
    pdf.chapter_body(
        f"Annotated images used for model training can be found in the following folder:"
    )
    pdf.path_bold(f'{fit_path_in_line(check_annotations_path)}')
    pdf.chapter_body(
        f"These images, including the one shown on this page, facilitate the visualization of annotation layout for the "
        f"annotations employed in training the model."
    )
    
    try:
        check_annotations_image = get_first_image(check_annotations_path)
        pdf.ln()
        pdf.add_image(check_annotations_image, pdf.l_margin, pdf.get_y(), page_width, 0)
    except RuntimeError as e:
        pdf.chapter_body(f"Warning: {str(e)}")

    # Section 4: Check Classification
    pdf.add_page()
    pdf.chapter_title('4. Check Classification')
    parent_classification_path = os.path.dirname(check_classification_path)
    pdf.chapter_body(f"Classified images by the model will be stored in the following folder:")
    pdf.path_bold(f'{fit_path_in_line(parent_classification_path)}')
    pdf.chapter_body(
        f"This folder contains grayscale images with labels of the segmented annotation classes. "
        f"A subfolder in this path, 'check_classification', contains JPG images like the one shown on this page, "
        f"with a mask overlay using the chosen color map for the classification."
    )
    
    try:
        check_classification_image = get_first_image(check_classification_path)
        pdf.ln()
        pdf.add_image(check_classification_image, pdf.l_margin, pdf.get_y(), page_width, 0)
    except RuntimeError as e:
        pdf.chapter_body(f"Warning: {str(e)}")

    # Section 5: Pixel and Tissue Composition Quantifications
    pdf.add_page()
    pdf.chapter_title('5. Pixel and Tissue Composition Quantifications')
    pdf.chapter_body(
        f'Pixel and tissue composition quantifications of the first image. The quantification of this and the other images has been saved in the CSV file.\nPath:'
    )
    pdf.path_bold(f'{fit_path_in_line(quantifications_csv_path)}\n')
    
    try:
        df = pd.read_csv(quantifications_csv_path)
        
        # Display quantification data
        pdf.set_font('Arial', '', 8)
        page_width = pdf.w - 2 * pdf.l_margin
        cell_width = 0
        
        # Find max cell width needed
        for column in df.columns:
            if pdf.get_string_width(column) > cell_width:
                cell_width = pdf.get_string_width(column)
        cell_width += 10
        
        # Header row
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(cell_width-10, 10, '')
        pdf.cell(cell_width-20, 10, df['Image name'][0], 1, align='C')
        pdf.cell(20, 10, '')
        pdf.cell(cell_width, 10, '')
        pdf.cell(cell_width-20, 10, df['Image name'][0], 1, align='C')
        pdf.ln()
        vertical_limit = False
        
        # Add pixel count and tissue composition rows
        for row in df.head().columns[1:int(len(df.head().columns)/2)+1]:
            if pdf.get_y() + 70 >= pdf.h:
                pdf.set_y(260)
                vertical_limit = True
                pdf.set_font('Arial', 'B', 40)
                pdf.multi_cell(0, 6, '...', align='C')
                pdf.set_y(270)
                pdf.set_font('Arial', 'B', 12)
                pdf.multi_cell(0, 6, 'Unable to fit entire quantification on the page. For more details, consult CSV file', align='C')
                break
            else:
                try:
                    comp_row = row[:row.index('pixel count')] + 'tissue composition (%)'
                    composition_value = f"{float(df[comp_row][0]):.2f}"
                    pdf.set_font('Arial', 'B', 8)
                    pdf.cell(cell_width - 10, 10, row, 1)
                    pdf.set_font('Arial', '', 8)
                    pixel_value = f"{int(df[row][0])}"
                    pdf.cell(cell_width - 20, 10, pixel_value, 1, align='R')
                    pdf.cell(20, 10, '')
                    pdf.set_font('Arial', 'B', 8)
                    pdf.cell(cell_width, 10, comp_row, 1)
                    pdf.set_font('Arial', '', 8)
                    pdf.cell(cell_width - 20, 10, composition_value, 1, align='R')
                    pdf.ln()
                except Exception:
                    whitespace_value = f"{int(df[row][0])}"
                    whitespace_row = row
        
        # Add whitespace row if there's room
        if not vertical_limit:
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(cell_width-10, 10, whitespace_row, 1)
            pdf.set_font('Arial', '', 8)
            pdf.cell(cell_width-20, 10, whitespace_value, 1, align='R')
    
    except Exception as e:
        pdf.chapter_body(f"Error processing quantification data: {str(e)}")

    # Section 6: Annex with explanatory text
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    pdf.chapter_title('6. Annex')
    explanatory_text = """
    Within the code workflow, a quantitative evaluation of the model is conducted using the specified annotation testing dataset on the 'File location' page of the GUI. This evaluation yields a confusion matrix, providing a detailed analysis of the model's classification performance for each annotated class. As shown on the first page, the confusion matrix is structured with precision scores for predicted annotation classes in the bottom row and sensitivity (recall) values for each annotated class in the final right column. The overall accuracy score of the model is situated at the bottom right corner of the confusion matrix table.

    The confusion matrix table employs a color-coded scheme, where classes that are prone to misclassification are colored following a yellow-to-red gradient, with scores over 90 colored in green.

    It is crucial to note that, for a model to be deemed decent, it should exhibit an overall accuracy score exceeding 85%, and each annotated class should have a precision score surpassing 85%. These metrics serve as benchmarks to guide the addition of more annotations to improve the model's performance and achieve a higher overall accuracy score.
    """
    pdf.underlined_body('Understanding the Confusion Matrix')
    pdf.chapter_body(explanatory_text)

    # Output the PDF
    pdf.output(output_path)
    print(f'PDF report saved at: {output_path}')


def create_output_pdf(
    output_path: str,
    pthDL: str,
    confusion_matrix_path: str,
    color_legend_path: str,
    check_annotations_path: str,
    check_classification_path: str,
    quantifications_csv_path: str
) -> None:
    """
    Legacy wrapper function for backward compatibility.
    
    This function maintains the original API while delegating to the new implementation.
    
    Args:
        output_path: Path where the PDF report will be saved
        pthDL: Path to the directory containing model data
        confusion_matrix_path: Path to the confusion matrix image
        color_legend_path: Path to the color legend image
        check_annotations_path: Path to directory containing annotation check images
        check_classification_path: Path to directory containing classification check images
        quantifications_csv_path: Path to CSV file containing quantification results
    """
    create_evaluation_report(
        output_path,
        pthDL,
        confusion_matrix_path,
        color_legend_path,
        check_annotations_path,
        check_classification_path,
        quantifications_csv_path
    )