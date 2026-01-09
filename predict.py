import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# 下載股票資料（只取測試/最新）
# -----------------------------
df = yf.download("AAPL", start="2020-01-01", end="2025-12-30")
data = df[['Close']]

# -----------------------------
# 載入之前訓練時用的 scaler
# -----------------------------
scaler = joblib.load("scaler.pkl")
scaled_data = scaler.transform(data)

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
# 切訓練 / 測試（推理用）
# -----------------------------
split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]

# -----------------------------
# 讀取模型
# -----------------------------
model = load_model("lstm_stock_model.h5")

# -----------------------------
# 預測
# -----------------------------
pred = model.predict(X_test)
pred_price = scaler.inverse_transform(pred.reshape(-1, 1))
real_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# 畫圖
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(real_price, label="Real Price")
plt.plot(pred_price, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction (LSTM)")
plt.show()
