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

def get_health_status(rul_value):
    if rul_value > 80:
        return "Healthy ✅"
    elif rul_value > 40:
        return "Moderate ⚠️"
    else:
        return "Critical 🔴"

def extract_pdf_data(pdf_file):
    """Extract numeric sensor data from PDF."""
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
    if len(numbers) < 210:
        raise ValueError("Not enough numeric data found in the PDF (expected 210 values).")
    arr = np.array(numbers[:210], dtype=np.float32).reshape(1, 10, 21)
    return arr

def extract_csv_data(csv_file):
    """Extract numeric sensor data from uploaded CSV file."""
    df = pd.read_csv(csv_file, header=None)
    values = df.values.flatten()
    if len(values) < 210:
        raise ValueError("Not enough data in CSV (expected at least 210 values).")
    arr = np.array(values[:210], dtype=np.float32).reshape(1, 10, 21)
    return arr

def get_dataset_sample(dataset, model_name):
    # Parse dataset part from model_name (e.g. FD001)
    dataset_folder = os.path.join("data")
    # Find appropriate RUL file or test file names; here use test_FDxxx.txt as sample
    file_map = {
        "FD001": "test_FD001.txt",
        "FD002": "test_FD002.txt",
        "FD003": "test_FD003.txt",
        "FD004": "test_FD004.txt"
    }
    if dataset not in file_map:
        raise ValueError(f"Dataset file for {dataset} not found.")
    file_path = os.path.join(dataset_folder, file_map[dataset])
    # Load sensor data from file: Assuming space-separated text; parse accordingly
    data = []
    with open(file_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            # Sensors start from column 2 (index 2 to 22: total 21 sensors)
            sensors = vals[2:23]
            data.append([float(x) for x in sensors])
    # get first 10 rows as sample arbitrarily
    sample_array = np.array(data[:10], dtype=np.float32).reshape(1, 10, 21)
    return sample_array


@app.route("/", methods=["GET", "POST"])
def home():
    models = list(MODEL_PATHS.keys())
    if request.method == "POST":
        try:
            model_name = request.form.get("model_name")
            input_option = request.form.get("input_option")

            if input_option == "dataset":
                dataset = model_name.split("_")[0]
                input_data = get_dataset_sample(dataset, model_name)

            elif input_option == "random":
                input_data = np.random.rand(1,10,21).astype(np.float32)

            elif input_option == "csv":
                csv_file = request.files.get("csv_file")
                if csv_file is None or csv_file.filename == "":
                    raise ValueError("No CSV file uploaded.")
                input_data = extract_csv_data(csv_file)

            elif input_option == "pdf":
                pdf_file = request.files.get("pdf_file")
                if pdf_file is None or pdf_file.filename == "":
                    raise ValueError("No PDF file uploaded.")
                input_data = extract_pdf_data(pdf_file)

            else:
                raise ValueError("Invalid input option selected.")

            if model_name not in MODELS:
                MODELS[model_name] = load_model(
                    MODEL_PATHS[model_name],
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS
                )

            model = MODELS[model_name]
            prediction = model.predict(input_data)
            rul_value = float(prediction.flatten()[0])

            result = {
                "model_used": model_name,
                "dataset_used": input_option,
                "predicted_RUL": round(rul_value,2),
                "machine_status": get_health_status(rul_value)
            }

            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("result.html", error=str(e))
    else:
        return render_template("index.html", models=models)


if __name__ == "__main__":
    print("Scanning models...")
    if MODEL_PATHS:
        print(f"Found {len(MODEL_PATHS)} models.")
    else:
        print("No saved models found. Please check the saved_models folder.")
    app.run(debug=True)
