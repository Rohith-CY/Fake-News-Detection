# ============================================================
# FAKE NEWS DETECTION SYSTEM
# Flask Web Application
# Model: Linear SVM
# Features: TF-IDF
# ============================================================

import os
import pickle
from pathlib import Path

from flask import Flask, render_template, request, jsonify


# ============================================================
# APPLICATION SETUP
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"


# Maximum text that can be submitted
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("FAKE NEWS DETECTION SYSTEM")
print("=" * 60)

print("\nLoading trained model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}\n\n"
        "Make sure models/model.pkl exists."
    )

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        f"Vectorizer file not found:\n{VECTORIZER_PATH}\n\n"
        "Make sure models/vectorizer.pkl exists."
    )


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


print("Model loaded successfully.")
print("Model type:", type(model).__name__)
print("Vectorizer loaded successfully.")
print("=" * 60)


# ============================================================
# NEWS PREDICTION FUNCTION
# ============================================================

def predict_news(news_text):
    """
    Predict whether the given news article is fake or real.

    Label:
        0 -> Fake News
        1 -> Real News
    """

    # Convert text into the same TF-IDF representation
    # used during model training.
    transformed_text = vectorizer.transform([news_text])

    # Make prediction
    prediction = model.predict(transformed_text)[0]

    if prediction == 0:
        return {
            "prediction": "FAKE NEWS",
            "label": 0,
            "status": "fake"
        }

    return {
        "prediction": "REAL NEWS",
        "label": 1,
        "status": "real"
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get news text from form
        news_text = request.form.get("news_text", "").strip()

        # Check empty input
        if not news_text:

            return render_template(
                "index.html",
                error="Please enter a news article before checking."
            )


        # Check minimum useful length
        if len(news_text) < 20:

            return render_template(
                "index.html",
                error="Please enter a longer news article."
            )


        # Perform prediction
        result = predict_news(news_text)


        # Send result to webpage
        return render_template(
            "index.html",
            result=result,
            news_text=news_text
        )


    except Exception as error:

        print("Prediction error:", error)

        return render_template(
            "index.html",
            error="An error occurred while processing the news."
        )


# ============================================================
# API PREDICTION ROUTE
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "success": False,
                "error": "Invalid JSON request."
            }), 400


        news_text = data.get("news_text", "").strip()


        if not news_text:

            return jsonify({
                "success": False,
                "error": "News text is required."
            }), 400


        if len(news_text) < 20:

            return jsonify({
                "success": False,
                "error": "Please provide a longer news article."
            }), 400


        result = predict_news(news_text)


        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "label": result["label"],
            "status": result["status"]
        })


    except Exception as error:

        print("API error:", error)

        return jsonify({
            "success": False,
            "error": "Prediction failed."
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": "Linear SVM",
        "vectorizer": "TF-IDF",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("\nStarting Fake News Detection Web Application...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )