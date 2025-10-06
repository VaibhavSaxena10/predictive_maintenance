# ===============================
# Predictive Maintenance (Multi-Dataset Version)
# ===============================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Helper: Load and Preprocess Dataset
# -----------------------------
def load_and_preprocess(file_path):
    column_names = [
        'engine_id', 'cycle',
        'op_setting_1', 'op_setting_2', 'op_setting_3'
    ] + [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(file_path, sep=" ", header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = column_names

    # Compute RUL
    rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
    rul_df.columns = ['engine_id', 'max_cycle']
    df = df.merge(rul_df, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)

    # Normalize sensors
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    # Create sequences
    SEQ_LENGTH = 10
    X, y = [], []
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        sensors = engine_data[sensor_cols].values
        rul = engine_data['RUL'].values
        for i in range(len(engine_data) - SEQ_LENGTH + 1):
            X.append(sensors[i:i+SEQ_LENGTH])
            y.append(rul[i+SEQ_LENGTH-1])

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')

    return X, y

# -----------------------------
# Helper: Build Model
# -----------------------------
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
    else:
        model.add(GRU(100, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -----------------------------
# Main Loop for All Datasets
# -----------------------------
datasets = ["FD001", "FD002", "FD003", "FD004"]
results = []

for ds in datasets:
    print(f"\n🚀 Processing Dataset: {ds}")
    file_path = f"data/train_{ds}.txt"

    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} not found, skipping...")
        continue

    X, y = load_and_preprocess(file_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model_dir = f"saved_models/{ds}"
    os.makedirs(model_dir, exist_ok=True)

    lstm_path = os.path.join(model_dir, "lstm_model.h5")
    gru_path = os.path.join(model_dir, "gru_model.h5")

    if os.path.exists(lstm_path) and os.path.exists(gru_path):
        print("✅ Loading pre-trained models...")
        lstm_model = load_model(lstm_path)
        gru_model = load_model(gru_path)
    else:
        print("⚙️ Training new models...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        lstm_model = build_model("LSTM", (X.shape[1], X.shape[2]))
        gru_model = build_model("GRU", (X.shape[1], X.shape[2]))

        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=32,
            callbacks=[early_stop], verbose=2
        )
        gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=32,
            callbacks=[early_stop], verbose=2
        )

        lstm_model.save(lstm_path)
        gru_model.save(gru_path)
        print("✅ Models trained and saved!")

    # Evaluate
    y_pred_lstm = lstm_model.predict(X_val)
    y_pred_gru = gru_model.predict(X_val)

    mae_lstm = mean_absolute_error(y_val, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_val, y_pred_lstm))
    r2_lstm = r2_score(y_val, y_pred_lstm)

    mae_gru = mean_absolute_error(y_val, y_pred_gru)
    rmse_gru = np.sqrt(mean_squared_error(y_val, y_pred_gru))
    r2_gru = r2_score(y_val, y_pred_gru)

    results.append([ds, "LSTM", mae_lstm, rmse_lstm, r2_lstm])
    results.append([ds, "GRU", mae_gru, rmse_gru, r2_gru])

    # Plot Comparison
    result_dir = f"results/{ds}"
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(y_val[:200], label='Actual RUL')
    plt.plot(y_pred_lstm[:200], label='LSTM Predicted')
    plt.plot(y_pred_gru[:200], label='GRU Predicted')
    plt.title(f'Actual vs Predicted RUL ({ds})')
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig(f"{result_dir}/rul_comparison_{ds}.png", dpi=300)
    plt.close()

# -----------------------------
# Save Final Evaluation Results
# -----------------------------
results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'MAE', 'RMSE', 'R2'])
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/all_datasets_evaluation.csv", index=False)

print("\n✅ All datasets processed and results saved in 'results/all_datasets_evaluation.csv'")
