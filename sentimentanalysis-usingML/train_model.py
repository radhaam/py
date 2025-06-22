# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
#NOTE: Need to delete vectorizer.pkl and sentiment_model.pkl after making any changes here to train the model
# Sample dataset
data = pd.DataFrame({
    'text': [
        'I love this product!', 'This is terrible.', 'Amazing experience.', 'I like this',
        'Worst ever!', 'Highly recommend.', 'Not worth the money.', 'I hate this', 'This is not good', 'this is good'
    ],
    'label': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]  # 1 = positive, 0 = negative
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
