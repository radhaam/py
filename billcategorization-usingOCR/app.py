# app.py
import os
import easyocr
import joblib
import pandas as pd
from pdf2image import convert_from_path

reader = easyocr.Reader(['en'])
model = joblib.load('bill_categorizer_model.pkl')

def extract_text(file_path):
    text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path, dpi=200)
        for i, img in enumerate(images):
            temp_path = f"page_{i}.jpg"
            img.save(temp_path)
            result = reader.readtext(temp_path, detail=0)
            text += " ".join(result)
            os.remove(temp_path)
    else:
        result = reader.readtext(file_path, detail=0)
        text = " ".join(result)
    return text

def categorize_bills(folder_path):
    results = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
            continue
        path = os.path.join(folder_path, file)
        print(f"Processing: {file}")
        text = extract_text(path)
        category = model.predict([text])[0]
        results.append({'filename': file, 'text_excerpt': text[:100], 'category': category})

    return pd.DataFrame(results)

if __name__ == "__main__":
    folder = "/Users/radham/Desktop/py/billcategorization-usingOCR/bills"
    df = categorize_bills(folder)
    df.to_csv('ml_categorized_bills.csv', index=False)
    print("\n Saved as ml_categorized_bills.csv")
    print(df)
