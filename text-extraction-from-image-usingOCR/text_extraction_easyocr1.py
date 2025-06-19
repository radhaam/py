import easyocr
import matplotlib.pyplot as plt
import cv2

# Load image
image_path = 'sample_image1.jpeg'  # Change this to your image path

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'hi']

# Perform OCR
results = reader.readtext(image_path)

# Print and draw results
for (text, prob) in results:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")
