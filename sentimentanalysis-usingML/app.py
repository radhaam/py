# app.py
from flask import Flask, request, jsonify
import pickle

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

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
