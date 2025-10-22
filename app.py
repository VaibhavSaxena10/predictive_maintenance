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
# Note: Removed PdfReader as it's no longer used

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
SEQ_LENGTH = 10
NUM_FEATURES = 21

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
MODELS = {} # Model cache

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
    """
    Returns a status message and color for the result.html template.
    """
    if rul_value > 80:
        return {"status": "Healthy ✅", "color": "#238636"} # Green
    elif rul_value > 40:
        return {"status": "Warning 🟡", "color": "#f1e05a"} # Yellow
    else:
        return {"status": "Critical 🔴", "color": "#da3633"} # Red

# Note: Removed extract_pdf_data function

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    models = list(MODEL_PATHS.keys())

    if request.method == "POST":
        try:
            model_name = request.form.get("model_name")
            input_text = request.form.get("input_data")

            if not model_name or model_name not in MODEL_PATHS:
                raise ValueError("Please select a valid model.")

            if not input_text:
                raise ValueError("Sensor data cannot be empty.")

            # --- CORE FIX: Parse and reshape text data ---
            numbers = []
            for val in input_text.split(','):
                if val.strip():
                    numbers.append(float(val.strip()))
            
            expected_numbers = SEQ_LENGTH * NUM_FEATURES
            if len(numbers) != expected_numbers:
                raise ValueError(
                    f"Invalid data shape. Expected {expected_numbers} numbers "
                    f"(for {SEQ_LENGTH} timesteps x {NUM_FEATURES} features), "
                    f"but received {len(numbers)}."
                )

            # Reshape to (1, 10, 21)
            input_data = np.array(numbers, dtype=np.float32).reshape(1, SEQ_LENGTH, NUM_FEATURES)
            # --- End of Core Fix ---

            # Load model if not cached
            if model_name not in MODELS:
                print(f"Loading model: {model_name}...")
                MODELS[model_name] = load_model(
                    MODEL_PATHS[model_name],
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS
                )

            model = MODELS[model_name]
            prediction = model.predict(input_data)
            rul_value = float(prediction.flatten()[0])
            
            status_info = get_health_status(rul_value)

            # ✅ Render the new result.html page with correct variables
            return render_template(
                "result.html",
                model_name=model_name,
                rul=round(rul_value, 2),
                status=status_info["status"],
                color=status_info["color"]
            )

        except Exception as e:
            # ✅ Render result.html with the error
            return render_template("result.html", error=str(e))

    # GET request: Just show the main page
    return render_template("index.html", models=models)


# --- FIX ---
# Removed the entire /predict API route as it's not used by the form.


if __name__ == "__main__":
    print("🔍 Scanning models...")
    if MODEL_PATHS:
        print(f"✅ Found {len(MODEL_PATHS)} models: {list(MODEL_PATHS.keys())}")
    else:
        print("⚠️ No models found. Please run training first.")
    app.run(debug=True)