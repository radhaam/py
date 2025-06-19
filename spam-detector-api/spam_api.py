from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

app = Flask(__name__)

# --- Train a basic spam classifier on startup ---
data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", header=None, names=["label", "message"])
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label_num']

model = MultinomialNB()
model.fit(X, y)

# Optional: Save the model and vectorizer
# pickle.dump(model, open("spam_model.pkl", "wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

@app.route('/')
def home():
    return "Spam Detector API is running!"

@app.route('/predict', methods=["POST"])
def predict():
    content = request.get_json()
    if "message" not in content:
        return jsonify({"error": "Please provide a 'message' in the request."}), 400
    
    msg = content["message"]
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    result = "spam" if prediction == 1 else "ham"

    return jsonify({"message": msg, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
