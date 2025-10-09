from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf

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

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to Predictive Maintenance API 🚀",
        "available_endpoints": ["/models", "/predict"]
    })

# ----------------------------
# List all available models
# ----------------------------
@app.route("/models", methods=["GET"])
def list_models():
    if not MODEL_PATHS:
        return jsonify({"error": "No models found in saved_models/ directory"}), 404
    return jsonify({
        "available_models": list(MODEL_PATHS.keys())
    })

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "model_name" not in data or "input" not in data:
        return jsonify({"error": "Request must include 'model_name' and 'input' JSON keys."}), 400

    model_name = data["model_name"]
    input_data = np.array(data["input"], dtype=np.float32)

    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    # Load model if not already loaded
    if model_name not in MODELS:
        print(f"Loading model {model_name} ...")
        MODELS[model_name] = tf.keras.models.load_model(MODEL_PATHS[model_name], compile=False)
        print(f"✅ {model_name} loaded successfully!")

    model = MODELS[model_name]

    # Reshape input to match (1, seq_len, num_features)
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data)
    return jsonify({
        "model_used": model_name,
        "predicted_RUL": float(prediction.flatten()[0])
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
