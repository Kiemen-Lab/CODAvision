"""
Model Evaluation Report Generator for CODAvision

This module provides functionality for generating PDF reports
that summarize model evaluation results, including confusion matrices,
color legends, annotations, classifications, tissue quantifications, and runtimes.

The module creates structured reports with sections for each evaluation component,
making it easy to assess model performance and share results.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: May 2025
"""

import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from fpdf import FPDF
from pathlib import Path
from typing import Dict, List, Tuple, Optional

pd.set_option('display.max_columns', None)

# --- Constants ---
MODEL_METADATA_FILE = 'net.pkl'
CONFUSION_MATRIX_TITLE = '1. Confusion Matrix'
COLOR_LEGEND_TITLE = '2. Color Legend'
CHECK_ANNOTATIONS_TITLE = '3. Check Annotations'
CHECK_CLASSIFICATION_TITLE = '4. Check Classification'
QUANTIFICATIONS_TITLE = '5. Pixel and Tissue Composition Quantifications'
RUNTIMES_TITLE = '6. Runtimes'
ANNEX_TITLE = '7. Annex'
SUPPORTED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']


# --- Helper Functions ---

def _convert_to_png(image_path: Path) -> Path:
    """Converts a JPG image to PNG format."""
    if image_path.suffix.lower() == '.png':
        return image_path
    if image_path.suffix.lower() not in ['.jpg', '.jpeg']:
         raise ValueError(f"Input path must be a JPG/JPEG file: {image_path}")

    png_path = image_path.with_suffix('.png')
    try:
        img = Image.open(image_path)
        img.save(png_path, 'PNG')
        return png_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert {image_path} to PNG: {e}")

def _get_first_image(directory: Path) -> Path:
    """Finds the first supported image file in a directory."""
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower().strip('.') in SUPPORTED_IMAGE_EXTENSIONS:
            return file
    raise FileNotFoundError(f"No supported image files ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) found in {directory}")


# --- Custom PDF Class ---

class PDF(FPDF):
    """Custom FPDF class for generating the model evaluation report."""

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def header(self):
        """Adds the report header with the model name."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'Performance report for model: {self.model_name}', 0, 1, 'C')
        self.ln(5) # Add some space after header

    def chapter_title(self, title: str):
        """Adds a chapter title."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2) # Add some space after title

    def chapter_body(self, body: str):
        """Adds multi-line body text."""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body, align='L')
        self.ln(2) # Add some space after body text

    def path_text(self, label: str, path_str: str):
        """Adds a path label and the formatted path."""
        self.set_font('Arial', '', 10)
        self.cell(0, 6, label, 0, 0, 'L')
        self.ln()
        self.set_font('Arial', 'B', 10)
        formatted_path = self._fit_path_in_line(path_str)
        self.multi_cell(0, 6, formatted_path, align='L')
        self.ln(4) # Add some space after path

    def underlined_body(self, body: str):
        """Adds underlined body text."""
        self.set_font('Arial', 'U', 10)
        self.multi_cell(0, 6, body, align='L')
        self.ln(2)

    def add_full_width_image(self, image_path: Path):
        """Adds an image, scaling it to fit the page width."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if image_path.suffix.lower().strip('.') not in SUPPORTED_IMAGE_EXTENSIONS:
             raise ValueError(f"Unsupported image type: {image_path.suffix}")

        page_width = self.w - 2 * self.l_margin
        # Use current y position, scale width to page width, height adjusts automatically (0)
        self.image(str(image_path), x=self.l_margin, y=self.get_y(), w=page_width, h=0)
        # Move Y position down after adding image (estimate height or use get_y after adding)
        # FPDF's automatic height calculation (h=0) handles the Y positioning. We add a newline for space.
        self.ln(5)

    def add_scaled_image(self, image_path: Path, width: float, height: float = 0):
        """Adds an image with specified width, maintaining aspect ratio if height is 0."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if image_path.suffix.lower().strip('.') not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {image_path.suffix}")

        # Use current y position
        self.image(str(image_path), x=self.l_margin, y=self.get_y(), w=width, h=height)
        self.ln(5) # Add space after image

    def _fit_path_in_line(self, path: str, max_line_len: int = 100, indent_chars: int = 80) -> str:
        """Formats a long path string to fit within lines by inserting newlines."""
        if len(path) <= max_line_len:
            return path

        lines = []
        current_pos = 0
        while current_pos < len(path):
            # Find the last separator within the next chunk of allowed length
            search_end = min(current_pos + max_line_len, len(path))
            split_pos = path.rfind(os.sep, current_pos, search_end)

            # If no separator found or it's too early, force break at max_line_len
            if split_pos <= current_pos or split_pos < current_pos + indent_chars :
                 split_pos = search_end

            # If we are at the end of the string after the split point
            if split_pos == len(path):
                 lines.append(path[current_pos:])
                 break
            # If we split at a separator, add 1 to include it on the current line
            elif path[split_pos] == os.sep and split_pos < search_end:
                 lines.append(path[current_pos:split_pos+1])
                 current_pos = split_pos + 1
            # Else, we are splitting mid-word or at max_line_len
            else:
                 lines.append(path[current_pos:split_pos])
                 current_pos = split_pos

        return '\n'.join(lines)


# --- Report Generator Class ---

class PdfReportGenerator:
    """Generates a PDF performance report for a segmentation model."""

    def __init__(
        self,
        output_path: str,
        model_dir: str,
        confusion_matrix_path_str: str,
        color_legend_path_str: str,
        check_annotations_path_str: str,
        check_classification_path_str: str,
        quantifications_csv_path_str: str,
        times: Dict[str, str],
    ):
        self.output_path = Path(output_path)
        self.model_dir = Path(model_dir)
        self.confusion_matrix_path = Path(confusion_matrix_path_str)
        self.color_legend_path = Path(color_legend_path_str)
        self.check_annotations_path = Path(check_annotations_path_str)
        self.check_classification_path = Path(check_classification_path_str)
        self.quantifications_csv_path = Path(quantifications_csv_path_str)
        self.times = times

        self.model_name: str = self._load_model_name()
        self.pdf = PDF(self.model_name)

    def _load_model_name(self) -> str:
        """Loads the model name from the net.pkl file."""
        pkl_path = self.model_dir / MODEL_METADATA_FILE
        if not pkl_path.exists():
            print(f"Warning: Model metadata file not found at {pkl_path}. Using 'Unknown Model'.")
            return "Unknown Model"
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data.get('nm', 'Unknown Model')
        except Exception as e:
            print(f"Warning: Failed to load model metadata from {pkl_path}: {e}. Using 'Unknown Model'.")
            return "Unknown Model"

    def _add_confusion_matrix_section(self):
        """Adds the confusion matrix section to the PDF."""
        self.pdf.add_page()
        self.pdf.chapter_title(CONFUSION_MATRIX_TITLE)
        self.pdf.chapter_body('Confusion matrix, encompassing precision, recall, and accuracy scores.')
        self.pdf.path_text('Path:', str(self.confusion_matrix_path))

        # Convert JPG to PNG if necessary
        cm_image_path = self.confusion_matrix_path
        if self.confusion_matrix_path.suffix.lower() in ['.jpg', '.jpeg']:
            try:
                print(f"Converting confusion matrix {self.confusion_matrix_path} to PNG...")
                cm_image_path = _convert_to_png(self.confusion_matrix_path)
                print(f"Converted to: {cm_image_path}")
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                self.pdf.chapter_body(f"Error processing confusion matrix image: {e}")
                print(f"Error: {e}")
                return # Skip adding image if conversion fails

        try:
            self.pdf.add_full_width_image(cm_image_path)
        except (FileNotFoundError, ValueError) as e:
             self.pdf.chapter_body(f"Error adding confusion matrix image: {e}")
             print(f"Error: {e}")


    def _add_color_legend_section(self):
        """Adds the color legend section to the PDF."""
        self.pdf.add_page()
        self.pdf.chapter_title(COLOR_LEGEND_TITLE)
        self.pdf.chapter_body('Color legend associated with the trained model, typically found in the same path as the confusion matrix.')
        self.pdf.path_text('Path:', str(self.color_legend_path))
        try:
            self.pdf.add_scaled_image(self.color_legend_path, width=100)
        except (FileNotFoundError, ValueError) as e:
             self.pdf.chapter_body(f"Error adding color legend image: {e}")
             print(f"Error: {e}")

    def _add_check_annotations_section(self):
        """Adds the check annotations section to the PDF."""
        self.pdf.add_page()
        self.pdf.chapter_title(CHECK_ANNOTATIONS_TITLE)
        self.pdf.chapter_body("Annotated images used for model training can be found in the following folder:")
        self.pdf.path_text('Path:', str(self.check_annotations_path))
        self.pdf.chapter_body(
            "These images, including the one shown on this page, facilitate the visualization "
            "of annotation layout for the annotations employed in training the model."
        )
        try:
            check_annotations_image = _get_first_image(self.check_annotations_path)
            self.pdf.add_full_width_image(check_annotations_image)
        except (FileNotFoundError, ValueError) as e:
             self.pdf.chapter_body(f"Error adding check annotations image: {e}")
             print(f"Error: {e}")

    def _add_check_classification_section(self):
        """Adds the check classification section to the PDF."""
        self.pdf.add_page()
        self.pdf.chapter_title(CHECK_CLASSIFICATION_TITLE)
        parent_classification_path = self.check_classification_path.parent
        self.pdf.chapter_body("Classified images by the model will be stored in the following folder:")
        self.pdf.path_text('Path:', str(parent_classification_path))
        self.pdf.chapter_body(
            "This folder contains grayscale images with labels of the segmented annotation classes. "
            "A subfolder in this path, 'check_classification', contains JPG images like the one shown on this page, "
            "with a mask overlay using the chosen color map for the classification."
        )
        try:
            check_classification_image = _get_first_image(self.check_classification_path)
            self.pdf.add_full_width_image(check_classification_image)
        except (FileNotFoundError, ValueError) as e:
             self.pdf.chapter_body(f"Error adding check classification image: {e}")
             print(f"Error: {e}")

    def _add_quantification_section(self):
        """Adds the pixel and tissue composition quantification section."""
        self.pdf.add_page()
        self.pdf.chapter_title(QUANTIFICATIONS_TITLE)
        self.pdf.chapter_body(
            'Pixel and tissue composition quantifications of the first image processed. '
            'The quantification of this and the other images has been saved in the CSV file.'
        )
        self.pdf.path_text('Path:', str(self.quantifications_csv_path))
        self.pdf.chapter_body('PC: Pixel Count          TC: Tissue Composition (%)')
        self.pdf.ln(5)

        try:
            df = pd.read_csv(self.quantifications_csv_path)
            if df.empty:
                self.pdf.chapter_body("Quantification CSV file is empty.")
                return

            quantifications = df.head(1) # Process only the first image for the PDF table
            if quantifications.empty:
                self.pdf.chapter_body("No data found for the first image in the CSV.")
                return

            self.pdf.set_font('Arial', '', 8)
            page_width = self.pdf.w - 2 * self.pdf.l_margin
            page_height = self.pdf.h - self.pdf.t_margin - self.pdf.b_margin

            # Estimate cell width based on headers - simplified approach
            max_header_width = 0
            for col in quantifications.columns:
                 max_header_width = max(max_header_width, self.pdf.get_string_width(col))

            cell_width = max_header_width + 15 # Add padding

            # Check if table fits horizontally (assuming 4 columns for PC/TC pairs)
            if cell_width * 4 > page_width:
                 # Adjust cell width if too wide (may truncate)
                 cell_width = page_width / 4.5 # Adjusted factor to leave some space

            # Header Row - Image Name
            image_name = quantifications['Image name'].iloc[0]
            display_name = image_name
            max_name_width = cell_width - 10 # Width for name cell
            if self.pdf.get_string_width(display_name) > max_name_width:
                for i in range(len(display_name), 0, -1):
                    short_name = display_name[:i] + '...'
                    if self.pdf.get_string_width(short_name) <= max_name_width:
                        display_name = short_name
                        break

            self.pdf.set_font('Arial', 'B', 8)
            self.pdf.cell(cell_width, 10, "Pixel Count", 1, 0, 'C')
            self.pdf.cell(cell_width, 10, display_name, 1, 0, 'C')
            self.pdf.cell(20, 10, '', 0, 0) # Spacer
            self.pdf.cell(cell_width, 10, "Tissue Comp (%)", 1, 0, 'C')
            self.pdf.cell(cell_width, 10, display_name, 1, 1, 'C') # Newline

            # Data Rows
            vertical_limit_reached = False
            whitespace_row = None
            whitespace_value = None

            for col_header in quantifications.columns[1:]: # Skip 'Image name'
                if ' PC' in col_header:
                    if self.pdf.get_y() + 10 > page_height - 30: # Check vertical limit with buffer
                        vertical_limit_reached = True
                        break

                    pc_value = quantifications[col_header].iloc[0]
                    # Try to find corresponding TC column
                    tc_header = col_header.replace(' PC', ' TC(%)')
                    tc_value = None
                    if tc_header in quantifications.columns:
                         tc_value = quantifications[tc_header].iloc[0]

                    self.pdf.set_font('Arial', 'B', 8)
                    self.pdf.cell(cell_width, 10, col_header, 1, 0)
                    self.pdf.set_font('Arial', '', 8)
                    self.pdf.cell(cell_width, 10, f"{int(pc_value):,}", 1, 0, 'R') # Format number with commas
                    self.pdf.cell(20, 10, '', 0, 0) # Spacer

                    if tc_value is not None:
                         self.pdf.set_font('Arial', 'B', 8)
                         self.pdf.cell(cell_width, 10, tc_header, 1, 0)
                         self.pdf.set_font('Arial', '', 8)
                         self.pdf.cell(cell_width, 10, f"{float(tc_value):.2f}", 1, 1, 'R') # Format float
                    else:
                         # Handle cases like whitespace which might only have PC
                         self.pdf.cell(cell_width, 10, "", 1, 0) # Empty cell for TC header
                         self.pdf.cell(cell_width, 10, "", 1, 1) # Empty cell for TC value
                         # Store whitespace info separately if it's the special case
                         if "Whitespace" in col_header:
                             whitespace_row = col_header
                             whitespace_value = f"{int(pc_value):,}"


            # Add vertical limit message if needed
            if vertical_limit_reached:
                self.pdf.set_font('Arial', 'B', 40)
                self.pdf.cell(0, 10, '...', 0, 1, 'C')
                self.pdf.set_font('Arial', 'B', 12)
                self.pdf.multi_cell(0, 6, 'Unable to fit entire quantification on the page. For more details, consult the CSV file.', align='C')

        except FileNotFoundError:
            self.pdf.chapter_body(f"Error: Quantification CSV file not found at {self.quantifications_csv_path}")
        except pd.errors.EmptyDataError:
             self.pdf.chapter_body(f"Error: Quantification CSV file is empty or invalid at {self.quantifications_csv_path}")
        except KeyError as e:
             self.pdf.chapter_body(f"Error: Missing expected column in CSV: {e}")
        except Exception as e:
            self.pdf.chapter_body(f"An unexpected error occurred while processing quantifications: {e}")
            print(f"Quantification Error: {e}")


    def _add_runtime_section(self):
        """Adds the runtimes section to the PDF."""
        self.pdf.add_page()
        self.pdf.chapter_title(RUNTIMES_TITLE)
        self.pdf.chapter_body('Summary of approximate module runtimes:')
        self.pdf.ln(5)

        if not self.times:
            self.pdf.chapter_body("No runtime data available.")
            return

        self.pdf.set_font('Arial', '', 8)

        # Determine optimal cell widths
        max_name_width = 0
        max_time_width = 0
        for key, value in self.times.items():
            max_name_width = max(max_name_width, self.pdf.get_string_width(key))
            # Format time before measuring width
            try:
                parts = value.split(":")
                hour = int(parts[0])
                minute = int(parts[1])
                sec_str = parts[2]
                if '.' in sec_str:
                    sec_int, sec_frac = sec_str.split(".")
                    second = f"{int(sec_int):02}.{sec_frac}"
                else:
                    second = f"{int(sec_str):02}"
                formatted_time = f"{hour:02}:{minute:02}:{second}"
                max_time_width = max(max_time_width, self.pdf.get_string_width(formatted_time))
            except:
                 max_time_width = max(max_time_width, self.pdf.get_string_width(value)) # Fallback

        cell_width_names = max_name_width + 10
        cell_width_times = max_time_width + 10

        # Table Header (Optional, can skip if just key-value pairs)
        # self.pdf.set_font('Arial', 'B', 8)
        # self.pdf.cell(cell_width_names, 10, 'Module', 1, 0, 'C')
        # self.pdf.cell(cell_width_times, 10, 'Runtime (HH:MM:SS.ms)', 1, 1, 'C')

        # Table Data
        for key, time_str in self.times.items():
            self.pdf.set_font('Arial', 'B', 8)
            self.pdf.cell(cell_width_names, 10, key, 1, 0, 'L') # Left align name

            # Format time string
            try:
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1])
                sec_str = parts[2]
                if '.' in sec_str:
                    sec_int, sec_frac = sec_str.split(".")
                    # Ensure fractional part has consistent length if needed, e.g., pad with zeros
                    # sec_frac = sec_frac.ljust(3, '0') # Example for 3 decimal places
                    second = f"{int(sec_int):02}.{sec_frac}"
                else:
                    second = f"{int(sec_str):02}" # Ensure seconds part is zero-padded

                formatted_time = f"{hour:02}:{minute:02}:{second}" # Ensure HH and MM are zero-padded
            except Exception:
                formatted_time = time_str # Fallback to original string if parsing fails

            self.pdf.set_font('Arial', '', 8)
            self.pdf.cell(cell_width_times, 10, formatted_time, 1, 1, 'R') # Right align time

    def _add_annex_section(self):
        """Adds the annex section with explanatory text."""
        self.pdf.add_page()
        self.pdf.chapter_title(ANNEX_TITLE)
        self.pdf.set_font('Arial', '', 10)

        explanatory_text = """
        Within the code workflow, a quantitative evaluation of the model is conducted using the specified annotation testing dataset on the 'File location' page of the GUI. This evaluation yields a confusion matrix, providing a detailed analysis of the model's classification performance for each annotated class. As shown on the first page, the confusion matrix is structured with precision scores for predicted annotation classes in the bottom row and sensitivity (recall) values for each annotated class in the final right column. The overall accuracy score of the model is situated at the bottom right corner of the confusion matrix table.
        
        The confusion matrix table employs a color-coded scheme, where classes that are prone to misclassification are colored following a yellow-to-red gradient, with scores over 90 colored in green.
        
        It is crucial to note that, for a model to be deemed decent, it should exhibit an overall accuracy score exceeding 85%, and each annotated class should have a precision score surpassing 85%. These metrics serve as benchmarks to guide the addition of more annotations to improve the model's performance and achieve a higher overall accuracy score.
        """
        self.pdf.underlined_body('Understanding the Confusion Matrix')
        self.pdf.chapter_body(explanatory_text.strip()) # Use strip() to remove leading/trailing whitespace

    def generate_report(self):
        """Generates the complete PDF report."""
        print('Generating model evaluation report...')

        self._add_confusion_matrix_section()
        self._add_color_legend_section()
        self._add_check_annotations_section()
        self._add_check_classification_section()
        self._add_quantification_section()
        self._add_runtime_section()
        self._add_annex_section()

        try:
            # Ensure the output directory exists before saving
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.pdf.output(str(self.output_path), 'F')
            print(f'PDF report saved at: {self.output_path}')
        except Exception as e:
            print(f"Error saving PDF report: {e}")
            # Consider logging the full traceback for debugging
            import traceback
            print(traceback.format_exc())
            raise  # Re-raise exception after logging

# --- Wrapper Function (for backward compatibility) ---

def create_output_pdf(
    output_path: str,
    pthDL: str,
    confusion_matrix_path: str,
    color_legend_path: str,
    check_annotations_path: str,
    check_classification_path: str,
    quantifications_csv_path: str,
    times: Dict[str, str]
) -> None:
    """
    Creates a PDF report summarizing model evaluation results.

    This function serves as a wrapper around the PdfReportGenerator class.

    Args:
        output_path: Path to save the output PDF file.
        pthDL: Path to the model directory (containing net.pkl).
        confusion_matrix_path: Path to the confusion matrix image file.
        color_legend_path: Path to the color legend image file.
        check_annotations_path: Path to the directory containing annotation check images.
        check_classification_path: Path to the directory containing classification check images.
        quantifications_csv_path: Path to the CSV file with quantification results.
        times: Dictionary containing runtime information for different modules.
    """
    try:
        report_generator = PdfReportGenerator(
            output_path=output_path,
            model_dir=pthDL,
            confusion_matrix_path_str=confusion_matrix_path,
            color_legend_path_str=color_legend_path,
            check_annotations_path_str=check_annotations_path,
            check_classification_path_str=check_classification_path,
            quantifications_csv_path_str=quantifications_csv_path,
            times=times,
        )
        report_generator.generate_report()
    except Exception as e:
        print(f"Failed to generate PDF report: {e}")
        # Optionally re-raise the exception if calling code needs to handle it
        # raise e