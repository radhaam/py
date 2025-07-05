# app.py
from flask import Flask, request, jsonify
#Flask: Used to create the web app.
#request: To access incoming request data (like JSON sent in a POST request).
#jsonify: Converts Python dictionaries into JSON responses for the client.
import pickle
#pickle: A Python module used to load and save Python objects (like models or vectorizers) from .pkl files.

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
#Loads the trained machine learning model (sentiment_model.pkl) in read binary (rb) mode.
#The model is unpickled (deserialized) and stored in the variable model.

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    # Vectorize and predict
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'text': text, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
