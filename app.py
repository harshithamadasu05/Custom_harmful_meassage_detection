from flask import Flask, render_template, request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# =============================
# Training Data
# =============================
messages = [
    "Hello how are you",
    "Let's meet tomorrow",
    "Good morning have a nice day",
    "Please call me when free",
    "Thank you for your help",

    "Win a free iPhone now click here",
    "Congratulations you won lottery claim now",
    "Limited offer buy now",
    "Earn money fast without work",
    "Click the link to get free rewards",

    "You are stupid and useless",
    "I hate you idiot",
    "Go and die",
    "You are a complete failure",
    "Shut up you moron"
]

labels = [
    "Safe", "Safe", "Safe", "Safe", "Safe",
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Harmful", "Harmful", "Harmful", "Harmful", "Harmful"
]

# =============================
# Text Cleaning
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

messages = [clean_text(msg) for msg in messages]

# =============================
# Model Training
# =============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = LogisticRegression()
model.fit(X, labels)

# =============================
# Web Route
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    result_message = ""

    if request.method == "POST":
        user_message = request.form["message"]
        cleaned = clean_text(user_message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "Safe":
            result_message = "✅ This message appears to be SAFE."
        elif prediction == "Spam":
            result_message = "⚠️ This message is likely SPAM. Avoid clicking unknown links."
        else:
            result_message = "🚨 This message contains ABUSIVE or HARMFUL content."

    return render_template(
        "index.html",
        prediction=prediction,
        result_message=result_message
    )

# =============================
# Run Server
# =============================
if __name__ == "__main__":
    app.run(debug=True)
