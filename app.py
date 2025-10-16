import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    MultiHeadAttention, LayerNormalization, Dense, Dropout,
    Conv1D, GlobalAveragePooling1D
)
from PyPDF2 import PdfReader

app = Flask(__name__)

# ----------------------------
# Load and map all models
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
# Helper functions
# ----------------------------
def get_health_status(rul_value):
    if rul_value > 80:
        return "Healthy ✅"
    elif rul_value > 40:
        return "Moderate ⚠️"
    else:
        return "Critical 🔴"


def extract_pdf_data(pdf_file):
    """Extracts numeric sensor data from a PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    numbers = []
    for token in text.replace("\n", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue

    if len(numbers) < 21:
        raise ValueError("Not enough numeric data found in the PDF.")

    arr = np.array(numbers[:21], dtype=np.float32).reshape(1, 1, -1)
    return arr


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    models = list(MODEL_PATHS.keys())

    if request.method == "POST":
        try:
            model_name = request.form.get("model_name")
            input_option = request.form.get("input_option")

            # ✅ Handle input selection
            if input_option == "random":
                input_data = np.random.rand(1, 10, 21).astype(np.float32)

            elif input_option == "dataset":
                sample_path = os.path.join("sample_data", "sample_fd001.csv")
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    arr = df.values[:10, :21]
                    input_data = np.expand_dims(arr, axis=0)
                else:
                    raise FileNotFoundError("Sample dataset not found in /sample_data/")

            elif input_option == "pdf":
                pdf_file = request.files.get("pdf_file")
                if not pdf_file or pdf_file.filename == "":
                    raise ValueError("No PDF uploaded.")
                input_data = extract_pdf_data(pdf_file)

            else:
                raise ValueError("Invalid input option selected.")

            # ✅ Load model if not cached
            if model_name not in MODELS:
                MODELS[model_name] = load_model(
                    MODEL_PATHS[model_name],
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS
                )

            model = MODELS[model_name]
            prediction = model.predict(input_data)
            rul_value = float(prediction.flatten()[0])

            # ✅ Prepare results for display
            result = {
                "model_used": model_name,
                "predicted_RUL": round(rul_value, 2),
                "machine_status": get_health_status(rul_value)
            }

            return render_template("result.html", result=result)

        except Exception as e:
            error_message = str(e)
            return render_template("result.html", result={"error": error_message})

    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict_api():
    """API endpoint for JSON requests."""
    data = request.get_json()
    if not data or "model_name" not in data or "input" not in data:
        return jsonify({"error": "Missing 'model_name' or 'input' keys."}), 400

    model_name = data["model_name"]
    input_data = np.array(data["input"], dtype=np.float32)

    try:
        if model_name not in MODELS:
            MODELS[model_name] = load_model(
                MODEL_PATHS[model_name],
                compile=False,
                custom_objects=CUSTOM_OBJECTS
            )

        model = MODELS[model_name]
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)

        prediction = model.predict(input_data)
        rul_value = float(prediction.flatten()[0])

        return jsonify({
            "model_used": model_name,
            "predicted_RUL": round(rul_value, 2),
            "machine_status": get_health_status(rul_value)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔍 Scanning models...")
    if MODEL_PATHS:
        print(f"✅ Found {len(MODEL_PATHS)} models: {list(MODEL_PATHS.keys())}")
    else:
        print("⚠️ No models found. Please run training first.")
    app.run(debug=True)
