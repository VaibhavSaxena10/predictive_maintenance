# ===============================
# Predictive Maintenance: LSTM & GRU
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
column_names = [
    'engine_id', 'cycle', 
    'op_setting_1', 'op_setting_2', 'op_setting_3'
] + [f'sensor_{i}' for i in range(1, 22)]

train_df = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)
train_df.dropna(axis=1, how='all', inplace=True)
train_df.columns = column_names

# -----------------------------
# 2️⃣ Compute RUL
# -----------------------------
rul_df = train_df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(rul_df, on='engine_id', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop('max_cycle', axis=1, inplace=True)

# -----------------------------
# 3️⃣ Normalize Sensor Data
# -----------------------------
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
scaler = MinMaxScaler()
train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])

# -----------------------------
# 4️⃣ Create Sequences
# -----------------------------
SEQ_LENGTH = 10
X, y = [], []

for engine_id in train_df['engine_id'].unique():
    engine_data = train_df[train_df['engine_id'] == engine_id]
    sensors = engine_data[sensor_cols].values
    rul = engine_data['RUL'].values
    
    for i in range(len(engine_data) - SEQ_LENGTH + 1):
        X.append(sensors[i:i+SEQ_LENGTH])
        y.append(rul[i+SEQ_LENGTH-1])

X = np.array(X, dtype='float32')
y = np.array(y, dtype='float32')

# -----------------------------
# 5️⃣ Train-Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# 6️⃣ LSTM Model
# -----------------------------
lstm_model = Sequential()
lstm_model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='linear'))
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# -----------------------------
# 7️⃣ GRU Model
# -----------------------------
gru_model = Sequential()
gru_model.add(GRU(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(50, return_sequences=False))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(1, activation='linear'))
gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

gru_history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# -----------------------------
# 8️⃣ Plot Validation Loss Comparison
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(lstm_history.history['val_loss'], label='LSTM val_loss')
plt.plot(gru_history.history['val_loss'], label='GRU val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM vs GRU Validation Loss')
plt.legend()
plt.show()

# -----------------------------
# 9️⃣ Predict and Compare RUL
# -----------------------------
y_pred_lstm = lstm_model.predict(X_val)
y_pred_gru = gru_model.predict(X_val)

plt.figure(figsize=(10,5))
plt.plot(y_val, label='Actual RUL')
plt.plot(y_pred_lstm, label='LSTM Predicted RUL')
plt.plot(y_pred_gru, label='GRU Predicted RUL')
plt.xlabel('Sample')
plt.ylabel('RUL')
plt.title('Actual vs Predicted RUL')
plt.legend()
plt.savefig("rul_comparison.png", dpi=300)
plt.show()
