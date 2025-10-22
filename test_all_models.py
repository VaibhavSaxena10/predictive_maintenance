import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# --- FIX: Removed TCN imports ---

# ---------------------------
# Configuration
# ---------------------------
MODEL_DIR = "saved_models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
# --- FIX: Removed "tcn" ---
MODEL_TYPES = ["lstm", "gru", "transformer"]

SEQ_LEN = 10
NUM_FEATURES = 21


# ---------------------------
# Utility: Load models safely
# ---------------------------
def safe_load_model(path):
    try:
        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={
                # --- FIX: Removed "TCN" ---
                "MultiHeadAttention": tf.keras.layers.MultiHeadAttention,
                "LayerNormalization": tf.keras.layers.LayerNormalization,
                "Conv1D": tf.keras.layers.Conv1D,
                "GlobalAveragePooling1D": tf.keras.layers.GlobalAveragePooling1D,
                "Dense": tf.keras.layers.Dense,
                "Dropout": tf.keras.layers.Dropout,
            },
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# ---------------------------
# Run predictions on all models
# ---------------------------
summary = []

print("🚀 Testing all models...\n")

for dataset in DATASETS:
    for model_type in MODEL_TYPES:
        model_name = f"{dataset}_{model_type}"
        model_path = os.path.join(MODEL_DIR, dataset, f"{model_type}.keras")

        if not os.path.exists(model_path):
            summary.append({
                "Dataset": dataset,
                "Model": model_type.upper(),
                "Error": f"Model not found at {model_path}",
            })
            continue

        try:
            # ✅ Load model
            model = safe_load_model(model_path)

            # ✅ Prepare dummy input for testing
            input_data = np.random.rand(1, SEQ_LEN, NUM_FEATURES).astype(np.float32)

            # For Transformer models, ensure correct shape
            if "transformer" in model_type.lower():
                if input_data.shape[1] < SEQ_LEN:
                    pad_len = SEQ_LEN - input_data.shape[1]
                    input_data = np.pad(input_data, ((0, 0), (pad_len, 0), (0, 0)), mode='edge')
                elif input_data.shape[1] > SEQ_LEN:
                    input_data = input_data[:, -SEQ_LEN:, :]

            # ✅ Predict RUL
            prediction = model.predict(input_data)
            rul_value = float(prediction.flatten()[0])

            # ✅ Machine health status
            if rul_value > 80:
                status = "Healthy ✅"
            elif rul_value > 40:
                status = "Warning 🟡"
            else:
                status = "Critical 🔴"

            summary.append({
                "Dataset": dataset,
                "Model": model_type.upper(),
                "Predicted_RUL": round(rul_value, 2),
                "Health_Status": status,
            })
            print(f"✅ {model_name} → RUL: {rul_value:.2f}, Status: {status}")

        except Exception as e:
            print(f"❌ {model_name} → Error: {e}")
            summary.append({
                "Dataset": dataset,
                "Model": model_type.upper(),
                "Error": str(e),
            })

# ---------------------------
# Save summary to CSV
# ---------------------------
df = pd.DataFrame(summary)
csv_path = os.path.join(RESULTS_DIR, "all_models_test_summary.csv")
df.to_csv(csv_path, index=False)

print("\n📊 Summary of All Model Predictions:\n")
# Handle case where Predicted_RUL might not exist if all models fail
if "Predicted_RUL" in df.columns:
    print(df[["Dataset", "Model", "Predicted_RUL", "Health_Status"]].dropna())
else:
    print("No valid predictions were made.")

# ---------------------------
# Identify best model
# ---------------------------
valid_results = df[df["Predicted_RUL"].notna()] if "Predicted_RUL" in df.columns else None
if valid_results is not None and not valid_results.empty:
    best_row = valid_results.loc[valid_results["Predicted_RUL"].idxmax()]
    best_model = f"{best_row['Dataset']} - {best_row['Model']}"
    best_rul = best_row["Predicted_RUL"]
    best_status = best_row["Health_Status"]
    print(f"\n🏆 Best Model: {best_model} | RUL: {best_rul} | Status: {best_status}")
else:
    print("\n⚠️ No valid predictions found!")

# ---------------------------
# Save bar chart
# ---------------------------
chart_path = os.path.join(RESULTS_DIR, "model_rul_comparison.png")
plt.figure(figsize=(10, 6))
if valid_results is not None and not valid_results.empty:
    plt.bar(valid_results["Dataset"] + " - " + valid_results["Model"], valid_results["Predicted_RUL"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Predicted RUL")
    plt.title("Model RUL Comparison")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📈 Saved bar chart to {chart_path}")
else:
    print("⚠️ Bar chart not saved as no valid predictions were found.")

print(f"✅ Saved summary to {csv_path}")