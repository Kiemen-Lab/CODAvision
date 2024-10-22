import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image
import pandas as pd


def convert_to_png(image_path):
    img = Image.open(image_path)
    png_path = image_path.replace('.jpg', '.png')
    img.save(png_path, 'PNG')
    return png_path

def create_output_pdf(output_path, confusion_matrix_path, color_legend_path, check_annotations_path, check_classification_path, quantifications_csv_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Model Evaluation Report")

    # Confusion Matrix
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 1.5 * inch, "1. Confusion Matrix")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 1.75 * inch, "Confusion matrix, encompassing precision, recall, and accuracy scores.")
    img = Image.open(confusion_matrix_path)
    img.verify()  # Verify that it is, in fact, an image
    if confusion_matrix_path.lower().endswith('.jpg'):
        confusion_matrix_path = convert_to_png(confusion_matrix_path)
    c.drawImage(confusion_matrix_path, 1 * inch, height - 5 * inch, width=4 * inch, preserveAspectRatio=True,
                mask='auto')

    # Color Legend
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 5.5 * inch, "2. Color Legend")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 5.75 * inch, "Color legend associated with the trained model situated in the same path as the confusion matrix.")
    c.drawImage(color_legend_path, 1 * inch, height - 8 * inch, width=4 * inch, preserveAspectRatio=True, mask='auto')

    # Check Annotations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 8.5 * inch, "3. Check Annotations")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 8.75 * inch, "A folder named 'Check annotations,' found in the same path as the 'training annotations'.")
    c.drawString(1 * inch, height - 9 * inch, "These images facilitate the visualization of annotations employed in training the model.")
    check_annotations_image = os.path.join(check_annotations_path, os.listdir(check_annotations_path)[0])
    c.drawImage(check_annotations_image, 1 * inch, height - 12 * inch, width=4 * inch, preserveAspectRatio=True, mask='auto')

    # Check Classification
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 1 * inch, "4. Check Classification")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 1.25 * inch, "Classified images by the model will be stored in a folder named 'Classification_model_name'.")
    c.drawString(1 * inch, height - 1.5 * inch, "This folder includes a 'check_classification' subfolder containing JPG images.")
    check_classification_image = os.path.join(check_classification_path, os.listdir(check_classification_path)[0])
    c.drawImage(check_classification_image, 1 * inch, height - 4.5 * inch, width=4 * inch, preserveAspectRatio=True, mask='auto')

    # Quantifications
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 5 * inch, "5. Pixel and Tissue Composition Quantifications")
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 5.25 * inch, "First 5 rows of the Pixel and tissue composition quantifications saved in the CSV file.")
    df = pd.read_csv(quantifications_csv_path)
    quantifications = df.head(5).to_string(index=False)
    text_object = c.beginText(1 * inch, height - 6 * inch)
    text_object.setFont("Helvetica", 8)
    text_object.textLines(quantifications)
    c.drawText(text_object)

    # Additional Explanatory Text
    c.showPage()
    c.setFont("Helvetica", 10)
    explanatory_text = """
    Within the code workflow, a quantitative evaluation of the model is conducted using the specified annotation testing dataset specified on the 'File location' page of the Excel GUI. This evaluation yields a confusion matrix, providing a detailed analysis of the model's classification performance for each annotated class. In the presented example (Figure 33), the confusion matrix is structured with precision scores for predicted annotation classes in the bottom row and sensitivity (recall) values for each annotated class in the final right column. The overall accuracy score of the model is situated at the bottom right corner of the confusion matrix table.

    The confusion matrix table employs a color-coded scheme, where classes that are prone to misclassification are visually highlighted by a darker shade of red. In the provided example, the misclassification of some stroma with Epithelium is evident through the intensified red coloring of those squares in the table.

    It is crucial to note that, for a model to be deemed decent, it should exhibit an overall accuracy score exceeding 85%, and each annotated class should have a precision score surpassing 85%. These metrics serve as benchmarks to guide the addition of more annotations to improve the model's performance and achieve a higher overall accuracy score.
    """
    text_object = c.beginText(1 * inch, height - 1 * inch)
    text_object.setFont("Helvetica", 10)
    text_object.textLines(explanatory_text)
    c.drawText(text_object)

    c.save()

# Example usage
if __name__ == '__main__':
    create_output_pdf(
        output_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\model_evaluation_report.pdf",
        confusion_matrix_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\confusion_matrix.jpg",
        color_legend_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024\model_color_legend.png",
        check_annotations_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\check_annotations",
        check_classification_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x\classification_CODA_python_08_30_2024\check_classification",
        quantifications_csv_path=r"\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x\classification_CODA_python_08_30_2024\image_quantifications.csv"
    )