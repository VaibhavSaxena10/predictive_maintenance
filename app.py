from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import load_model

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


# Load all models available
MODEL_PATHS = get_all_models()
MODELS = {}

# Custom objects for TCN and Transformer models
CUSTOM_OBJECTS = {
    "MultiHeadAttention": MultiHeadAttention,
    "LayerNormalization": LayerNormalization,
    "Conv1D": Conv1D,
    "GlobalAveragePooling1D": GlobalAveragePooling1D,
    "Dense": Dense,
    "Dropout": Dropout,
}

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/")
def home():
    """Show available models and API usage."""
    return jsonify({
        "message": "Welcome to Predictive Maintenance API 🚀",
        "total_models_found": len(MODEL_PATHS),
        "available_models": list(MODEL_PATHS.keys())[:5] + ["..."] if MODEL_PATHS else "No models found",
        "available_endpoints": {
            "/models": "List all models available for prediction",
            "/predict": "POST JSON: { 'model_name': 'FD001_LSTM', 'input': [[...]] }"
        },
        "tip": "Use /models to check all available models or /predict to test RUL prediction."
    })


@app.route("/models", methods=["GET"])
def list_models():
    """List all available models."""
    if not MODEL_PATHS:
        return jsonify({"error": "No models found in saved_models/ directory"}), 404
    return jsonify({
        "total_models": len(MODEL_PATHS),
        "available_models": list(MODEL_PATHS.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Perform RUL prediction using specified model."""
    data = request.get_json()

    if not data or "model_name" not in data or "input" not in data:
        return jsonify({"error": "Request must include 'model_name' and 'input' JSON keys."}), 400

    model_name = data["model_name"]
    input_data = np.array(data["input"], dtype=np.float32)

    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    # Load model if not already loaded
    if model_name not in MODELS:
        try:
            print(f"Loading model {model_name} ...")
            MODELS[model_name] = load_model(
                MODEL_PATHS[model_name],
                compile=False,
                custom_objects=CUSTOM_OBJECTS
            )
            print(f"✅ {model_name} loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return jsonify({"error": f"Failed to load model {model_name}: {e}"}), 500

    model = MODELS[model_name]

    # Reshape input for model (1, seq_len, num_features)
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)

    # Perform prediction
    try:
        prediction = model.predict(input_data)
        rul_value = float(prediction.flatten()[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Determine machine status
    if rul_value > 80:
        status = "Healthy ✅"
    elif rul_value > 40:
        status = "Moderate ⚠️"
    else:
        status = "Critical 🔴"

    return jsonify({
        "model_used": model_name,
        "predicted_RUL": round(rul_value, 2),
        "machine_status": status
    })


# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    print("🔍 Scanning for models...")
    if MODEL_PATHS:
        print(f"✅ Found {len(MODEL_PATHS)} models: {list(MODEL_PATHS.keys())}")
    else:
        print("⚠️ No models found. Please run training first.")
    app.run(debug=True)
