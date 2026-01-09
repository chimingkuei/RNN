import numpy as np
import pandas as pd
from typing_extensions import TypedDict
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 下載股票資料
# -----------------------------
df = yf.download("AAPL", start="2019-01-01", end="2024-01-01")
data = df[['Close']]

# -----------------------------
# 正規化
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 保存 scaler 以便推理時使用
import joblib
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 建立時間序列資料
# -----------------------------
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# -----------------------------
# 切訓練 / 測試
# -----------------------------
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 建立 LSTM 模型
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -----------------------------
# 訓練
# -----------------------------
with tf.device('/GPU:0'):
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

# -----------------------------
# 儲存模型
# -----------------------------
model.save("lstm_stock_model.h5")
print("模型已儲存為 lstm_stock_model.h5")
