import os
import cv2
import tempfile
import tensorflow as tf
import keras

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable

# Required for Lambda layer loading
keras.config.enable_unsafe_deserialization()

@register_keras_serializable()
def abs_diff(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Load trained Siamese model
model = load_model(
    "D:/Project/models/signature_siamese_model.keras",
    custom_objects={"abs_diff": abs_diff}
)

app = Flask(__name__)

# Image preprocessing (must match training)
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 100, 100, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if not file1 or not file2:
            prediction = "⚠️ Please upload both images"
        else:
            # Save files temporarily
            temp_dir = tempfile.gettempdir()
            path1 = os.path.join(temp_dir, file1.filename)
            path2 = os.path.join(temp_dir, file2.filename)
            file1.save(path1)
            file2.save(path2)

            # Predict
            score = model.predict([
                preprocess_image(path1),
                preprocess_image(path2)
            ])[0][0]

            prediction = "✅ Genuine" if score > 0.8 else "❌ Forged"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
