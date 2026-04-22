# -*- coding: utf-8 -*-
"""
Train: Market Price Prediction — LSTM Time Series
Generates per-crop synthetic daily price data and trains a shared LSTM
model to forecast the next N days given a 30-day history window.

Run: python train_market.py   (from this file's directory)
  OR: python ml_models/market_price/train_market.py  (from project root)
Expected val MAE: < 80 ₹/quintal on synthetic data
Training time: ~30–60 seconds (CPU)
"""
import os, sys, io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Suppress TF INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'datasets', 'market_prices.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN   = 30   # days of history fed into LSTM
PRED_DAYS = 1    # steps to predict ahead
EPOCHS    = 40
BATCH     = 64

print("=" * 55)
print("  AgroSense — Market Price LSTM Training")
print("=" * 55)
print(f"   TensorFlow version: {tf.__version__}")

# ── Crop price profiles ────────────────────────────────────────────────
PROFILES = {
    'Rice':     {'base': 2100, 'trend': +0.5,  'vol': 60,  'seasonal': 80},
    'Wheat':    {'base': 2350, 'trend': +0.3,  'vol': 45,  'seasonal': 60},
    'Maize':    {'base': 1850, 'trend': +0.8,  'vol': 90,  'seasonal': 70},
    'Onion':    {'base': 1200, 'trend': -0.2,  'vol': 280, 'seasonal': 200},
    'Tomato':   {'base': 800,  'trend': -0.5,  'vol': 380, 'seasonal': 300},
    'Potato':   {'base': 950,  'trend': -0.3,  'vol': 130, 'seasonal': 100},
    'Soybean':  {'base': 4800, 'trend': +1.0,  'vol': 80,  'seasonal': 90},
    'Cotton':   {'base': 6200, 'trend': +0.2,  'vol': 150, 'seasonal': 120},
    'Groundnut':{'base': 5100, 'trend': +0.7,  'vol': 110, 'seasonal': 95},
}

# ── 1. Generate or load time-series data ──────────────────────────────
def generate_price_series(profile, n_days=730, seed=None):
    """Simulate realistic price time series with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n_days)
    trend    = profile['trend'] * t
    seasonal = profile['seasonal'] * np.sin(2 * np.pi * t / 365)
    noise    = rng.normal(0, profile['vol'] * 0.4, n_days)
    # Random spikes (market events)
    spikes   = rng.choice([0, 1], n_days, p=[0.97, 0.03]) * rng.uniform(
                    profile['vol'], profile['vol'] * 3, n_days)
    price    = profile['base'] + trend + seasonal + noise + spikes
    return np.clip(price, profile['base'] * 0.3, profile['base'] * 2.5)

if os.path.exists(DATA_PATH):
    df_all = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset loaded: {df_all.shape}")
else:
    print("\n⚠️  Generating synthetic price time-series (730 days × 9 crops)...")
    rows = []
    for crop, prof in PROFILES.items():
        prices = generate_price_series(prof, n_days=730, seed=hash(crop) % (2**32))
        for day, p in enumerate(prices):
            rows.append({'day': day, 'crop': crop, 'price': round(p, 2)})
    df_all = pd.DataFrame(rows)
    df_all.to_csv(DATA_PATH, index=False)
    print(f"   Generated {len(df_all)} rows → {DATA_PATH}")

# ── 2. Build sequences ─────────────────────────────────────────────────
print("\n⏳ Building LSTM sequences...")
crop_names = sorted(df_all['crop'].unique())
crop_idx   = {c: i for i, c in enumerate(crop_names)}
n_crops    = len(crop_names)

# Scale prices globally
scaler = MinMaxScaler(feature_range=(0, 1))
all_prices = df_all['price'].values.reshape(-1, 1)
scaler.fit(all_prices)

X_seq, y_seq, crop_codes = [], [], []

for crop in crop_names:
    prices = df_all[df_all['crop'] == crop].sort_values('day')['price'].values
    prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
    c_code = crop_idx[crop] / (n_crops - 1)   # normalised crop code [0,1]

    for i in range(len(prices_scaled) - SEQ_LEN - PRED_DAYS + 1):
        window = prices_scaled[i : i + SEQ_LEN]
        target = prices_scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_DAYS]
        # Feature: [price, crop_code_repeated]
        crop_col = np.full(SEQ_LEN, c_code)
        X_seq.append(np.stack([window, crop_col], axis=1))
        y_seq.append(target)

X_arr = np.array(X_seq, dtype=np.float32)   # (N, 30, 2)
y_arr = np.array(y_seq, dtype=np.float32)   # (N, 1)
print(f"   Sequences: {X_arr.shape}  Labels: {y_arr.shape}")

# Train/val split (time-aware: last 20% as val)
split = int(len(X_arr) * 0.8)
X_train, X_val = X_arr[:split], X_arr[split:]
y_train, y_val = y_arr[:split], y_arr[split:]

# ── 3. Build LSTM model ────────────────────────────────────────────────
print("\n⏳ Building LSTM model...")
tf.random.set_seed(42)

inputs = keras.Input(shape=(SEQ_LEN, 2), name='price_sequence')
x = layers.LSTM(64, return_sequences=True, name='lstm_1')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(32, name='lstm_2')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(PRED_DAYS, name='price_forecast')(x)

model = keras.Model(inputs, outputs, name='AgroSense_MarketLSTM')
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='huber',
    metrics=['mae']
)
model.summary()

# ── 4. Train ───────────────────────────────────────────────────────────
print(f"\n⏳ Training for {EPOCHS} epochs...")
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
lr_reduce  = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# ── 5. Evaluate ────────────────────────────────────────────────────────
val_pred   = model.predict(X_val, verbose=0)
# Inverse-transform both to original ₹ scale
val_pred_r = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
val_true_r = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
mae_rs     = np.mean(np.abs(val_pred_r - val_true_r))
print(f"\n✅ Val MAE (original scale): ₹{mae_rs:.2f}/quintal")
print(f"   Best val_loss : {min(history.history['val_loss']):.6f}")

# ── 6. Save ────────────────────────────────────────────────────────────
model_path  = os.path.join(SAVE_DIR, 'market_lstm.keras')
scaler_path = os.path.join(SAVE_DIR, 'market_scaler.pkl')
meta_path   = os.path.join(SAVE_DIR, 'market_meta.pkl')

model.save(model_path)
joblib.dump(scaler,     scaler_path)
joblib.dump({'crops': crop_names, 'seq_len': SEQ_LEN,
             'profiles': PROFILES}, meta_path)
print(f"\n💾 Saved: {model_path}")
print(f"💾 Saved: {scaler_path}")
print(f"💾 Saved: {meta_path}")

# ── 7. Sanity check ────────────────────────────────────────────────────
print("\n🧪 Sanity check — Rice price forecast:")
rice_prices = df_all[df_all['crop'] == 'Rice'].sort_values('day')['price'].values[-SEQ_LEN:]
rice_scaled = scaler.transform(rice_prices.reshape(-1, 1)).flatten()
c_code_rice = crop_idx['Rice'] / (n_crops - 1)
test_seq    = np.stack([rice_scaled, np.full(SEQ_LEN, c_code_rice)], axis=1)
test_seq    = test_seq[np.newaxis, :, :].astype(np.float32)
pred_scaled = model.predict(test_seq, verbose=0)
pred_price  = float(scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
print(f"   Last 30d avg: ₹{rice_prices.mean():.0f} | Forecast tomorrow: ₹{pred_price:.0f}")
print("\n✅ Training complete! Model ready for Flask API.")
