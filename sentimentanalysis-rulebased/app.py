from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
#used textblb for sentiment analysis
    text = data["text"]
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Determine sentiment based on polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return jsonify({
        "text": text,
        "polarity_score": polarity,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
