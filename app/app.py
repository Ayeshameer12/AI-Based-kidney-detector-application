"""
Kidney Abnormality Detector — Flask app
-----------------------------------------
Upload a kidney CT/ultrasound image, get a Normal/Abnormal prediction
with a Grad-CAM heatmap showing which region influenced the decision,
plus a downloadable PDF report.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import io
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, send_file, url_for
import tensorflow as tf
import cv2

from gradcam import make_gradcam_heatmap, overlay_heatmap
from report import generate_pdf_report

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "kidney_model.h5")
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "Conv_1"  # last conv layer name in MobileNetV2 — verify against model.summary()

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# Load model once at startup
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"WARNING: model not found at {MODEL_PATH}. "
          f"Train it in notebooks/train_model.ipynb and place kidney_model.h5 here.")


def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr_batch = np.expand_dims(arr, axis=0)
    arr_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(arr_batch.copy())
    return arr, arr_preprocessed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not loaded yet — train it first and add kidney_model.h5.")

    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an image to analyze.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return render_template("index.html", error="Please upload a JPG or PNG image.")

    session_id = uuid.uuid4().hex[:10]
    original_filename = f"{session_id}_original.jpg"
    heatmap_filename = f"{session_id}_heatmap.jpg"
    original_path = os.path.join(UPLOAD_DIR, original_filename)
    heatmap_path = os.path.join(UPLOAD_DIR, heatmap_filename)
    file.save(original_path)

    # Preprocess and predict
    raw_arr, preprocessed = preprocess_image(original_path)
    prob_abnormal = float(model.predict(preprocessed, verbose=0)[0][0])
    predicted_label = "Abnormal" if prob_abnormal > 0.5 else "Normal"
    confidence = prob_abnormal if predicted_label == "Abnormal" else 1 - prob_abnormal

    # Grad-CAM heatmap
    try:
        heatmap = make_gradcam_heatmap(preprocessed, model, LAST_CONV_LAYER)
        overlay_heatmap(original_path, heatmap, heatmap_path)
        heatmap_available = True
    except Exception as e:
        print("Grad-CAM failed:", e)
        heatmap_available = False

    result = {
        "session_id": session_id,
        "label": predicted_label,
        "confidence": round(confidence * 100, 1),
        "prob_abnormal": round(prob_abnormal * 100, 1),
        "original_url": url_for("static", filename=f"uploads/{original_filename}"),
        "heatmap_url": url_for("static", filename=f"uploads/{heatmap_filename}") if heatmap_available else None,
        "timestamp": datetime.now().strftime("%B %d, %Y at %H:%M"),
    }

    # Stash paths for PDF generation
    app.config[f"RESULT_{session_id}"] = {
        **result,
        "original_path": original_path,
        "heatmap_path": heatmap_path if heatmap_available else None,
    }

    return render_template("result.html", result=result)


@app.route("/report/<session_id>")
def download_report(session_id):
    data = app.config.get(f"RESULT_{session_id}")
    if not data:
        return "Report not found or session expired.", 404

    pdf_buffer = generate_pdf_report(data)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"kidney_report_{session_id}.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
