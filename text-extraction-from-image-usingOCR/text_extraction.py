from PIL import Image       # To load the image
import pytesseract          # To extract text using OCR
import os

# (Optional) Set tesseract path if you're on Windows
# Example:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 1: Load the image file
image_path = 'sample_image3.jpeg'   # Replace with your actual image file path
if not os.path.exists(image_path):
    print("Image file not found!")
else:
    image = Image.open(image_path)

    # Step 2: Run OCR on the image
    extracted_text = pytesseract.image_to_string(image)

    # Step 3: Print the extracted text
    print("=== Extracted Text ===")
    print(extracted_text)
