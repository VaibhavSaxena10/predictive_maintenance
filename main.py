"""
Predictive Maintenance: LSTM, GRU, TCN, Transformer benchmarking across FD001-FD004
Now includes:
- Runtime accuracy tracker (console + CSV)
- Training curve plotting
- Final summary plots for all datasets and models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LSTM, GRU,
                                     LayerNormalization, MultiHeadAttention,
                                     GlobalAveragePooling1D, Conv1D)
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Try importing TCN
try:
    from tcn import TCN
except Exception:
    TCN = None
    print("⚠️ keras-tcn not installed. Install with: pip install keras-tcn")

# ---------------------------
# Config
# ---------------------------
SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 40
RANDOM_STATE = 42
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

MODEL_DIR = "saved_models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Utility Functions
# ---------------------------
def load_cmapss_train(file_path):
    cols = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = cols
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def make_sequences(df, seq_length=SEQ_LENGTH):
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    X, y = [], []
    for uid in df['engine_id'].unique():
        sub = df[df['engine_id'] == uid]
        sensors = sub[sensor_cols].values
        rul = sub['RUL'].values
        for i in range(len(sub) - seq_length + 1):
            X.append(sensors[i:i + seq_length])
            y.append(rul[i + seq_length - 1])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32'), scaler


def save_plot(y_true, preds_dict, out_path, max_points=200):
    plt.figure(figsize=(10, 5))
    n = min(len(y_true), max_points)
    plt.plot(y_true[:n], label='Actual RUL', linewidth=2)
    for name, pred in preds_dict.items():
        plt.plot(pred[:n], label=f'{name} Predicted')
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_training_curves(history, model_name, dataset_name, out_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{model_name} MAE ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_accuracy_curve_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 Saved training curve for {model_name} → {out_path}")


def plot_training_curves_from_csv(csv_file, model_name, dataset_name, out_dir):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df['mae'], label='Train MAE')
    plt.plot(df['val_mae'], label='Val MAE')
    plt.title(f'{model_name} MAE ({dataset_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_accuracy_curve_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 Saved training curve from CSV for {model_name} → {out_path}")


# ---------------------------
# Callbacks
# ---------------------------
class LiveAccuracyTrackerCSV(Callback):
    """Prints live metrics and saves per-epoch metrics to CSV."""
    def __init__(self, out_file):
        super().__init__()
        self.out_file = out_file
        self.logs_list = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_log = {
            "epoch": epoch + 1,
            "loss": logs.get('loss'),
            "val_loss": logs.get('val_loss'),
            "mae": logs.get('mae'),
            "val_mae": logs.get('val_mae')
        }
        self.logs_list.append(epoch_log)
        print(f"🟩 Epoch {epoch_log['epoch']:02d} | Loss: {epoch_log['loss']:.4f} | "
              f"Val Loss: {epoch_log['val_loss']:.4f} | MAE: {epoch_log['mae']:.4f} | Val MAE: {epoch_log['val_mae']:.4f}")
        pd.DataFrame(self.logs_list).to_csv(self.out_file, index=False)


# ---------------------------
# Model Builders
# ---------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(100, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(100, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_tcn(input_shape):
    if TCN is not None:
        i = Input(shape=input_shape)
        x = TCN(nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8],
                use_skip_connections=True)(i)
        out = Dense(1, activation='linear')(x)
        model = Model(i, out)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    i = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(i)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1)(x)
    model = Model(i, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pe, dtype=tf.float32)


def build_transformer(input_shape, d_model=64, num_heads=4, ff_dim=128):
    seq_len, n_features = input_shape
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(d_model)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ---------------------------
# Summary Plot Function
# ---------------------------
def plot_summary(results_df, out_path):
    """Plot final summary of MAE and RMSE for all models across datasets."""
    datasets = results_df['Dataset'].unique()
    models = results_df['Model'].unique()
    width = 0.2
    x = np.arange(len(datasets))

    plt.figure(figsize=(12, 5))
    # MAE plot
    plt.subplot(1, 2, 1)
    for i, model in enumerate(models):
        vals = results_df[results_df['Model'] == model]['MAE']
        plt.bar(x + i * width, vals, width=width, label=model)
    plt.xticks(x + width * (len(models)-1)/2, datasets)
    plt.xlabel('Dataset')
    plt.ylabel('MAE')
    plt.title('MAE Comparison')
    plt.legend()

    # RMSE plot
    plt.subplot(1, 2, 2)
    for i, model in enumerate(models):
        vals = results_df[results_df['Model'] == model]['RMSE']
        plt.bar(x + i * width, vals, width=width, label=model)
    plt.xticks(x + width * (len(models)-1)/2, datasets)
    plt.xlabel('Dataset')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 Saved final summary plot → {out_path}")


# ---------------------------
# Main Loop
# ---------------------------
all_results = []

for ds in DATASETS:
    print(f"\n🚀 Processing Dataset: {ds}")
    file_path = f"data/train_{ds}.txt"
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found. Skipping.")
        continue

    df = load_cmapss_train(file_path)
    X, y, _ = make_sequences(df, seq_length=SEQ_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    ds_model_dir = os.path.join(MODEL_DIR, ds)
    ds_result_dir = os.path.join(RESULTS_DIR, ds)
    os.makedirs(ds_model_dir, exist_ok=True)
    os.makedirs(ds_result_dir, exist_ok=True)

    paths = {
        'LSTM': os.path.join(ds_model_dir, 'lstm.keras'),
        'GRU': os.path.join(ds_model_dir, 'gru.keras'),
        'TCN': os.path.join(ds_model_dir, 'tcn.keras'),
        'TRF': os.path.join(ds_model_dir, 'transformer.keras')
    }

    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"📂 Loading {name} model for {ds}")
            models[name] = load_model(path, compile=False)
            models[name].compile(optimizer='adam', loss='mse', metrics=['mae'])
            csv_file = os.path.join(ds_result_dir, f"{name}_live_metrics_{ds}.csv")
            if os.path.exists(csv_file):
                plot_training_curves_from_csv(csv_file, name, ds, ds_result_dir)
        else:
            print(f"🧠 Building new {name} model for {ds}")
            if name == 'LSTM': models[name] = build_lstm((X.shape[1], X.shape[2]))
            elif name == 'GRU': models[name] = build_gru((X.shape[1], X.shape[2]))
            elif name == 'TCN': models[name] = build_tcn((X.shape[1], X.shape[2]))
            else: models[name] = build_transformer((X.shape[1], X.shape[2]))

    for name, model in models.items():
        if not os.path.exists(paths[name]):
            print(f"🏋️‍♂️ Training {name} on {ds} ...")
            cb = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
            csv_tracker_path = os.path.join(ds_result_dir, f"{name}_live_metrics_{ds}.csv")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[cb, LiveAccuracyTrackerCSV(csv_tracker_path)],
                verbose=0
            )
            model.save(paths[name])
            print(f"✅ Saved {name} model to {paths[name]}")
            plot_training_curves(history, name, ds, ds_result_dir)

    preds = {}
    for name, model in models.items():
        print(f"🔍 Evaluating {name} on {ds} ...")
        y_pred = model.predict(X_val, batch_size=BATCH_SIZE)
        preds[name] = y_pred.flatten()
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        all_results.append([ds, name, mae, rmse, r2])
        print(f"📈 {name} Results → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    plot_path = os.path.join(ds_result_dir, f"rul_comparison_{ds}.png")
    save_plot(y_val, preds, plot_path)
    print(f"🖼️ Saved RUL comparison plot → {plot_path}")

# Save overall comparison
results_df = pd.DataFrame(all_results, columns=['Dataset', 'Model', 'MAE', 'RMSE', 'R2'])
csv_out = os.path.join(RESULTS_DIR, "all_datasets_comparison.csv")
results_df.to_csv(csv_out, index=False)
print(f"\n✅ All done! Consolidated results saved → {csv_out}")

# Generate summary plot
summary_plot_path = os.path.join(RESULTS_DIR, "summary_mae_rmse_comparison.png")
plot_summary(results_df, summary_plot_path)
