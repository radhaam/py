import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import matplotlib.pyplot as plt

# Load trained model (no need to compile for prediction)
model = load_model("image_classifier.h5", compile=False)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predicted_class, confidence = classify_image('cat1.webp')
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

    # Show the image with predicted label
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
