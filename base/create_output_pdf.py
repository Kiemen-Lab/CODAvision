import os
from PIL import Image
from fpdf import FPDF
import pandas as pd
import pickle
import numpy as np
pd.set_option('display.max_columns', None)

def convert_to_png(image_path):
    img = Image.open(image_path)
    png_path = image_path.replace('.jpg', '.png')
    img.save(png_path, 'PNG')
    return png_path

class PDF(FPDF):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'Performance report for model: {self.model_name}', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body,align= 'L' )

    def path_bold(self,path):
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 6, path, align='L')
        self.ln()

    def underlined_body(self, body):
        self.set_font('Arial', 'U', 10)
        self.multi_cell(0, 6, body, align='L')

    def add_image(self, image_path, x, y, w, h):
        # Check if the image file has a supported extension
        supported_extensions = ['jpg', 'jpeg', 'png']
        file_extension = image_path.split('.')[-1].lower()

        if file_extension not in supported_extensions:
            raise RuntimeError(f'Unsupported image type: {file_extension}')

        # Add the image to the PDF
        self.image(image_path, x, y, w, h)

def get_first_image(directory, supported_extensions=['jpg', 'jpeg', 'png']):
    for file in os.listdir(directory):
        if file.split('.')[-1].lower() in supported_extensions:
            return os.path.join(directory, file)
    raise RuntimeError(f"No supported image files found in {directory}")

def create_output_pdf(output_path, pthDL, confusion_matrix_path, color_legend_path, check_annotations_path, check_classification_path, quantifications_csv_path, times):
    print('Generating model evaluation report...')

    # Load model name from pickle file
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
    model_name = data.get('nm', 'Unknown Model')

    pdf = PDF(model_name)
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)

    # Confusion Matrix
    def fit_path_in_line(path):
        if len(path) > 100:
            for i in range(int(np.floor(len(path)/100)), 0, -1):
                path = path[:path.find('\\', (i-1)*100+80)+1] + '\n' + path[path.find('\\', (i-1)*100+80)+1:]
        return path

    pdf.chapter_title('1. Confusion Matrix')
    pdf.chapter_body('Confusion matrix, encompassing precision, recall, and accuracy scores.\nPath:',)
    pdf.path_bold(f'{fit_path_in_line(confusion_matrix_path)}\n')
    if confusion_matrix_path.lower().endswith('.jpg'):
        confusion_matrix_path = convert_to_png(confusion_matrix_path)

    # Calculate the width of the image to fit the page width excluding margins
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.add_image(confusion_matrix_path, pdf.l_margin, 60, page_width, 0)

    # Color Legend
    pdf.add_page()
    pdf.chapter_title('2. Color Legend')
    pdf.chapter_body(f'Color legend associated with the trained model situated in the same path as the confusion matrix.\nPath:\n')
    pdf.path_bold(f'{fit_path_in_line(color_legend_path)}\n')
    pdf.add_image(color_legend_path, 10, 60, 100, 0)

    # Check Annotations
    pdf.add_page()
    pdf.chapter_title('3. Check Annotations')
    pdf.chapter_body(
       f"Annotated images used for model training can be found in the following folder:")
    pdf.path_bold(f'{fit_path_in_line(check_annotations_path)}')
    pdf.chapter_body(
        f"These images, including the one shown on this page, facilitate the visualization of annotation layout for the "
        f"annotations employed in training the model.")
    check_annotations_image = get_first_image(check_annotations_path)
    pdf.ln()
    pdf.add_image(check_annotations_image, pdf.l_margin, pdf.get_y(), page_width, 0)

    # Check Classification
    pdf.add_page()
    pdf.chapter_title('4. Check Classification')
    parent_classification_path = os.path.dirname(check_classification_path)
    pdf.chapter_body(f"Classified images by the model will be stored in the following folder:")
    pdf.path_bold(f'{fit_path_in_line(parent_classification_path)}')
    pdf.chapter_body(f"This folder contains grayscale images with labels of the segmented annotation classes. "
                     f"A subfolder in this path, 'check_classification', contains JPG images like the one shown on this page, "
                     f"with a mask overlay using the chosen color map for the classification.")
    check_classification_image = get_first_image(check_classification_path)
    pdf.ln()
    pdf.add_image(check_classification_image, pdf.l_margin, pdf.get_y(), page_width, 0)

    # Quantifications
    pdf.add_page()
    pdf.chapter_title('5. Pixel and Tissue Composition Quantifications')
    pdf.chapter_body(
        f'Pixel and tissue composition quantifications of the first image. The quantification of this and the other images has been saved in the CSV file.\nPath:')
    pdf.path_bold(f'{fit_path_in_line(quantifications_csv_path)}\n')
    df = pd.read_csv(quantifications_csv_path)
    quantifications = df.head(5)

    # Set font for table
    pdf.set_font('Arial', '', 8)

    # Calculate available width
    page_width = pdf.w - 2 * pdf.l_margin
    page_heigh = pdf.h

    # Calculate the width of each header
    cell_width = 0
    for column in quantifications.columns:
        if pdf.get_string_width(column) > cell_width:
            cell_width = pdf.get_string_width(column)
    cell_width += 10
    num_rows = len(df.columns)

    # Print header
    pdf.set_font('Arial', 'B', 8)  # Set font to bold for headers
    pdf.cell(cell_width - 10, 10, '')
    image_name = df['Image name'][0]
    if pdf.get_string_width(df['Image name'][0])> cell_width-20:
        for i in range(len(image_name), 0, -1):
            short_name = image_name[:i]
            if pdf.get_string_width(short_name+'...')<= cell_width-20:
                image_name = short_name+'...'
                break
    pdf.cell(cell_width - 20, 10, image_name, 1, align='C')
    pdf.cell(20, 10, '')
    pdf.cell(cell_width, 10, '')
    pdf.cell(cell_width - 20, 10, image_name, 1, align='C')
    pdf.ln()
    vertical_limit = False

    # Print rows
    for row in df.head().columns[1:int(len(df.head().columns) / 2) + 1]:
        if pdf.get_y() + 70 >= page_heigh:
            pdf.set_y(260)
            vertical_limit = True
            pdf.set_font('Arial', 'B', 40)
            pdf.multi_cell(0, 6, '...', align='C')
            pdf.set_y(270)
            pdf.set_font('Arial', 'B', 12)
            pdf.multi_cell(0, 6,
                           'Unable to fit entire quantification on the page. For more details, consult CSV file',
                           align='C')
            break
        else:
            try:  # Whitespace info will be last since it has no tissue composition %
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
            except:
                whitespace_value = f"{int(df[row][0])}"
                whitespace_row = row

    # Print whitespace row
    if not vertical_limit:
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(cell_width - 10, 10, whitespace_row, 1)
        pdf.set_font('Arial', '', 8)
        pdf.cell(cell_width - 20, 10, whitespace_value, 1, align='R')

        # Runtimes
        pdf.add_page()
        pdf.chapter_title('6. Runtimes')
        pdf.chapter_body(
            f'Summary of each module runtime')

        # Set font for table
        pdf.set_font('Arial', '', 8)

        # Calculate the width of each header
        cell_width_names = 0
        cell_width_times = 0
        for index, key in enumerate(times):
            if pdf.get_string_width(key) > cell_width_names:
                cell_width_names = pdf.get_string_width(key)
            if pdf.get_string_width(times[key]) > cell_width_times:
                cell_width_times = pdf.get_string_width(times[key])
        cell_width_names += 10
        cell_width_times += 10

        # Print table
        pdf.set_font('Arial', 'B', 8)  # Set font to bold for headers
        for index, key in enumerate(times):
            pdf.cell(cell_width_names + 10, 10, key, 1, align='C')
            time = times[key]
            parts = time.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            # Handle seconds with or without decimal
            if '.' in parts[2]:
                sec_int, sec_frac = parts[2].split(".")
                second = f"{int(sec_int):02}.{sec_frac}"
            else:
                second = f"{int(parts[2]):02}"
            time = f"{hour:02}:{minute:02}:{second}"
            pdf.cell(cell_width_times + 10, 10, time, 1, align='C')
            pdf.ln()

    # Additional Explanatory Text
    pdf.add_page()
    pdf.set_font('Arial', '', 10)  # Set the font to Arial, regular, size 10
    pdf.chapter_title('7. Annex')
    explanatory_text = """
    Within the code workflow, a quantitative evaluation of the model is conducted using the specified annotation testing dataset on the 'File location' page of the GUI. This evaluation yields a confusion matrix, providing a detailed analysis of the model's classification performance for each annotated class. As shown on the first page, the confusion matrix is structured with precision scores for predicted annotation classes in the bottom row and sensitivity (recall) values for each annotated class in the final right column. The overall accuracy score of the model is situated at the bottom right corner of the confusion matrix table.

    The confusion matrix table employs a color-coded scheme, where classes that are prone to misclassification are colored following a yellow-to-red gradient, with scores over 90 colored in green.

    It is crucial to note that, for a model to be deemed decent, it should exhibit an overall accuracy score exceeding 85%, and each annotated class should have a precision score surpassing 85%. These metrics serve as benchmarks to guide the addition of more annotations to improve the model's performance and achieve a higher overall accuracy score.
    """
    pdf.underlined_body('Understanding the Confusion Matrix')
    pdf.chapter_body(explanatory_text)

    pdf.output(output_path)
    print(f'PDF report saved at: {output_path}')

