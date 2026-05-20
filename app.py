from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import json
from werkzeug.utils import secure_filename

# =========================
# FLASK SETUP
# =========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# =========================
# LOAD CLASS NAMES
# =========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# BUILD MODEL (SAME AS TRAINING)
# =========================
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# =========================
# LOAD MODEL WEIGHTS
# =========================
model = build_model(len(class_names))
model.load_weights("model.weights.h5")  # 🔥 IMPORTANT NAME

print("✅ Model loaded successfully!")

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    # preprocess
    img_array = preprocess_image(filepath)

    # predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()

    # top 3 predictions
    top3_idx = np.argsort(score)[-3:][::-1]

    results = []
    for i in top3_idx:
        results.append({
            "breed": class_names[i],
            "confidence": float(score[i] * 100)
        })

    return jsonify({"predictions": results})

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)