import os
from PIL import Image
from fpdf import FPDF
import pandas as pd
import pickle

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
        self.multi_cell(0, 10, body)
        self.ln()
        self.ln(5)

    def add_image(self, image_path, x, y, w, h):
        self.image(image_path, x, y, w, h)

def create_output_pdf(output_path, pthDL, confusion_matrix_path, color_legend_path, check_annotations_path, check_classification_path, quantifications_csv_path):
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
    pdf.chapter_title('1. Confusion Matrix')
    pdf.chapter_body(f'Confusion matrix, encompassing precision, recall, and accuracy scores.\nPath: {confusion_matrix_path}')
    if confusion_matrix_path.lower().endswith('.jpg'):
        confusion_matrix_path = convert_to_png(confusion_matrix_path)

    # Calculate the width of the image to fit the page width excluding margins
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.add_image(confusion_matrix_path, pdf.l_margin, 60, page_width, 0)

    # Color Legend
    pdf.add_page()
    pdf.chapter_title('2. Color Legend')
    pdf.chapter_body(f'Color legend associated with the trained model situated in the same path as the confusion matrix.\nPath: {color_legend_path}')
    pdf.add_image(color_legend_path, 10, 60, 100, 0)

    # Check Classification
    pdf.add_page()
    pdf.chapter_title('4. Check Classification')
    parent_classification_path = os.path.dirname(check_classification_path)
    pdf.chapter_body(f"Classified images by the model will be stored in the following folder: {parent_classification_path}. "
                     f"This folder contains grayscale images with labels of the segmented annotation classes. "
                     f"A subfolder in this path, 'check_classification', contains JPG images like the one shown on this page, "
                     f"with a mask overlay using the chosen color map for the classification.")
    check_classification_image = os.path.join(check_classification_path, os.listdir(check_classification_path)[0])
    pdf.add_image(check_classification_image, pdf.l_margin, 70, page_width, 0)

    # Check Annotations
    pdf.add_page()
    pdf.chapter_title('3. Check Annotations')
    pdf.chapter_body(
        f"Annotated images used for model training can be found in the following folder: {check_annotations_path}. "
        f"These images, including the one shown on this page, facilitate the visualization of annotation layout for the "
        f"annotations employed in training the model.")
    check_annotations_image = os.path.join(check_annotations_path, os.listdir(check_annotations_path)[0])
    pdf.add_image(check_annotations_image, pdf.l_margin, 60, page_width, 0)

    # Quantifications
    pdf.add_page()
    pdf.chapter_title('5. Pixel and Tissue Composition Quantifications')
    pdf.chapter_body(
        f'First 5 rows of the Pixel and tissue composition quantifications saved in the CSV file.\nPath: {quantifications_csv_path}')
    df = pd.read_csv(quantifications_csv_path)
    quantifications = df.head(5)

    # Set font for table
    pdf.set_font('Arial', '', 8)

    # Calculate available width
    page_width = pdf.w - 2 * pdf.l_margin
    # cell_width = 50
    # max_cols_per_row = int(page_width // cell_width)  # Convert to integer

    # Calculate the width of each header
    header = quantifications.columns
    cell_widths = [pdf.get_string_width(col) + 10 for col in header]  # Add padding

    # Calculate the maximum number of columns per row
    max_cols_per_row = 0
    current_width = 0
    for width in cell_widths:
        if current_width + width > page_width:
            break
        current_width += width
        max_cols_per_row += 1

    # Add table header and rows
    num_cols = len(header)
    num_rows = len(quantifications)

    for start_col in range(0, num_cols, max_cols_per_row):

        # Reset current_width and max_cols_per_row for the next table
        current_width = 0
        max_cols_per_row = 0
        for width in cell_widths:
            if current_width + width > page_width:
                break
            current_width += width
            max_cols_per_row += 1

        end_col = min(start_col + max_cols_per_row, num_cols)

        # Print image name header
        pdf.set_font('Arial', 'B', 8)  # Set font to bold for image names
        pdf.cell(cell_widths[0], 10, header[0], 1)
        pdf.ln()

        # Print header
        pdf.set_font('Arial', 'B', 8)  # Set font to bold for headers
        for col in header[start_col:end_col]:
            pdf.cell(cell_widths[header.get_loc(col)], 10, col, 1)
        pdf.ln()

        # Print rows
        pdf.set_font('Arial', '', 8)  # Set font back to normal for rows
        for index, row in quantifications.iterrows():
            for col in header[start_col:end_col]:
                try:
                    if 'pixel' in col.lower():
                        value = f"{int(row[col])}"
                    else:
                        value = f"{float(row[col]):.2f}"
                except ValueError:
                    value = str(row[col])
                pdf.cell(cell_widths[header.get_loc(col)], 10, value, 1)
            pdf.ln()

        # Leave a space of 10 units between each table
        pdf.ln(10)




    # Additional Explanatory Text
    pdf.add_page()
    pdf.set_font('Arial', '', 10)  # Set the font to Arial, regular, size 10
    pdf.chapter_title('6. Annex')
    pdf.set_font('Arial', 'U', 10)  # Set the font to Arial, underline, size 10
    pdf.cell(0, 10, 'Understanding the Confusion Matrix', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)  # Reset the font to Arial, regular, size 10
    explanatory_text = """
    Within the code workflow, a quantitative evaluation of the model is conducted using the specified annotation testing dataset on the 'File location' page of the Excel GUI. This evaluation yields a confusion matrix, providing a detailed analysis of the model's classification performance for each annotated class. As shown on the first page, the confusion matrix is structured with precision scores for predicted annotation classes in the bottom row and sensitivity (recall) values for each annotated class in the final right column. The overall accuracy score of the model is situated at the bottom right corner of the confusion matrix table.

    The confusion matrix table employs a color-coded scheme, where classes that are prone to misclassification are colored following a yellow-to-red gradient, with scores over 90 colored in green.

    It is crucial to note that, for a model to be deemed decent, it should exhibit an overall accuracy score exceeding 85%, and each annotated class should have a precision score surpassing 85%. These metrics serve as benchmarks to guide the addition of more annotations to improve the model's performance and achieve a higher overall accuracy score.
    """
    pdf.chapter_body(explanatory_text)

    pdf.output(output_path)
    print(f'PDF report saved at: {output_path}')

# Example usage
if __name__ == '__main__':
    create_output_pdf(
        output_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\model_evaluation_report.pdf',
        confusion_matrix_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\confusion_matrix.jpg',
        color_legend_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\model_color_legend.png',
        check_annotations_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\check_annotations',
        check_classification_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x\classification_CODA_python_08_30_2024\check_classification',
        quantifications_csv_path=r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x\classification_CODA_python_08_30_2024\image_quantifications.csv',
        pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024'
    )