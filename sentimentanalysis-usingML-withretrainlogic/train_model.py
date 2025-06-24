import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Sample initial dataset
data = pd.DataFrame({
    'text': ['I love this', 'I hate this', 'Awesome product', 'Terrible experience'],
    'label': [1, 0, 1, 0]
})

# Load or create vectorizer
if os.path.exists("vectorizer.pkl"):
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
else:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['text'])
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

# Vectorize
X = vectorizer.transform(data['text'])

# Load or create model
if os.path.exists("sentiment_model.pkl"):
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    first_time = False
else:
    model = SGDClassifier(loss='log')
    first_time = True

# Incremental training
if first_time:
    model.partial_fit(X, data['label'], classes=[0, 1])
else:
    model.partial_fit(X, data['label'])

# Save model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model updated.")
