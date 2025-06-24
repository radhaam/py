# app.py
from flask import Flask, request, jsonify
import pickle

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/predict-results', methods=['POST'])
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

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json()
    text = data.get('text')
    label = data.get('label')

    if text is None or label is None:
        return jsonify({'error': 'Text and label are required'}), 400

    # Update model
    vec = vectorizer.transform([text])
    model.partial_fit(vec, [label])

    # Save updated model
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return jsonify({'message': 'Model retrained with new input'}), 200


if __name__ == '__main__':
    app.run(debug=True)
