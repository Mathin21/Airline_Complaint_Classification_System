from flask import Flask, render_template, request
import numpy as np
import joblib
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model
model = load_model(
    "airline_complaint_classifier.keras",
    compile = False
)

tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
MAX_LEN = 180

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict the classification
@app.route("/predict", methods=["POST"])
def predict():
    complaint = request.form["complaint"]

    cleaned_text = clean_text(complaint)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)

    predicted_id = np.argmax(prediction)

    category = label_encoder.inverse_transform([predicted_id])[0]

    return render_template(
        "result.html",
        complaint=complaint,
        category=category
    )

# run the application
if __name__ == "__main__":
    app.run(debug=True)