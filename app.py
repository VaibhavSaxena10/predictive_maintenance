from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv1D, GlobalAveragePooling1D
import pandas as pd
from PyPDF2 import PdfReader

app = Flask(__name__)

# ----------------------------
# Utility: Scan all saved models
# ----------------------------
def get_all_models(base_dir="saved_models"):
    model_paths = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".keras"):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    dataset, model_name = parts[0], parts[1].replace(".keras", "")
                    model_paths[f"{dataset}_{model_name}"] = os.path.join(root, file)
    return model_paths


MODEL_PATHS = get_all_models()
MODELS = {}

CUSTOM_OBJECTS = {
    "MultiHeadAttention": MultiHeadAttention,
    "LayerNormalization": LayerNormalization,
    "Conv1D": Conv1D,
    "GlobalAveragePooling1D": GlobalAveragePooling1D,
    "Dense": Dense,
    "Dropout": Dropout,
}

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html", models=list(MODEL_PATHS.keys()))


@app.route("/predict", methods=["POST"])
def predict_page():
    model_name = request.form.get("model_name")
    input_mode = request.form.get("input_mode")  # dataset / random / pdf

    if not model_name or not input_mode:
        return render_template("result.html", error="⚠️ Please select a model and input mode.")

    # ----------------------------
    # Input mode 1: From dataset
    # ----------------------------
    if input_mode == "dataset":
        dataset_name = model_name.split("_")[0]
        file_path = f"data/train_{dataset_name}.txt"
        if not os.path.exists(file_path):
            return render_template("result.html", error=f"Dataset file {file_path} not found.")

        df = pd.read_csv(file_path, sep=" ", header=None)
        df.dropna(axis=1, how="all", inplace=True)
        sensor_data = df.iloc[0:10, 5:26].values  # take 10 cycles, 21 sensors
        input_data = np.expand_dims(sensor_data, axis=0)

    # ----------------------------
    # Input mode 2: Random sample
    # ----------------------------
    elif input_mode == "random":
        input_data = np.random.rand(1, 10, 21).astype(np.float32)

    # ----------------------------
    # Input mode 3: PDF upload
    # ----------------------------
    elif input_mode == "pdf":
        if "pdf_file" not in request.files or request.files["pdf_file"].filename == "":
            return render_template("result.html", error="⚠️ Please upload a PDF file.")
        pdf_file = request.files["pdf_file"]

        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            # Extract numbers from the text
            numbers = [float(x) for x in text.replace("\n", " ").split() if x.replace(".", "", 1).isdigit()]
            if len(numbers) < 210:
                return render_template("result.html", error="⚠️ PDF doesn't contain enough numeric sensor values.")
            input_data = np.array(numbers[:210], dtype=np.float32).reshape(1, 10, 21)
        except Exception as e:
            return render_template("result.html", error=f"Failed to read PDF: {e}")

    else:
        return render_template("result.html", error="Invalid input mode selected.")

    # ----------------------------
    # Model Loading and Prediction
    # ----------------------------
    if model_name not in MODEL_PATHS:
        return render_template("result.html", error=f"Model '{model_name}' not found.")

    if model_name not in MODELS:
        try:
            MODELS[model_name] = load_model(MODEL_PATHS[model_name], compile=False, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            return render_template("result.html", error=f"Failed to load model {model_name}: {e}")

    model = MODELS[model_name]
    prediction = model.predict(input_data)
    rul_value = float(prediction.flatten()[0])

    # ----------------------------
    # Health Status
    # ----------------------------
    if rul_value > 80:
        status, color = "Healthy ✅", "green"
    elif rul_value > 40:
        status, color = "Moderate ⚠️", "orange"
    else:
        status, color = "Critical 🔴", "red"

    return render_template(
        "result.html",
        model_name=model_name,
        rul=round(rul_value, 2),
        status=status,
        color=color,
        input_mode=input_mode
    )


if __name__ == "__main__":
    print("🔍 Scanning for models...")
    if MODEL_PATHS:
        print(f"✅ Found {len(MODEL_PATHS)} models: {list(MODEL_PATHS.keys())}")
    else:
        print("⚠️ No models found.")
    app.run(debug=True)
