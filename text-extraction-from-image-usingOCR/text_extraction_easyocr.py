import easyocr
import matplotlib.pyplot as plt
import cv2

# Load image
image_path = 'sample_image3.jpeg'  # Change this to your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'hi']

# Perform OCR
results = reader.readtext(image_path)

# Print and draw results
for (bbox, text, prob) in results:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")
    # Draw bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image_rgb, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image_rgb, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show image with detections
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
