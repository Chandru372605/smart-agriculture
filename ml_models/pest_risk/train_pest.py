# -*- coding: utf-8 -*-
"""
Train: Pest Risk Prediction — Random Forest Regressor
Outputs risk score 0–100 (treated as regression → binned to Low/Med/High)
Dataset: Synthetic if no CSV found — good enough for demo
  Real option: ICAR pest dataset or custom-built CSV

Run: python train_pest.py   (from this file's directory)
  OR: python ml_models/pest_risk/train_pest.py  (from project root)
Expected accuracy: ~88%
Training time: < 30 seconds
"""
import os, sys, io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Force UTF-8 output so emoji print correctly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'datasets', 'pest_risk.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 55)
print("  AgroSense — Pest Risk Model Training")
print("=" * 55)

# ── 1. Load or generate ────────────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset loaded: {df.shape}")
else:
    print("\n⚠️  No CSV found — generating synthetic pest risk data...")
    np.random.seed(42)
    n       = 4000
    crops   = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Tomato', 'Potato', 'Sugarcane']
    seasons = ['Kharif', 'Rabi', 'Zaid']

    temp    = np.random.uniform(10, 45, n)
    hum     = np.random.uniform(20, 100, n)
    prev    = np.random.randint(0, 4, n)   # 0=None, 1=Light, 2=Moderate, 3=Severe
    density = np.random.randint(0, 3, n)   # 0=Low, 1=Medium, 2=High
    water   = np.random.randint(0, 3, n)   # 0=No, 1=within 500m, 2=within 100m
    crop    = np.random.choice(crops, n)
    season  = np.random.choice(seasons, n)

    # Risk score formula (agronomically driven)
    score = (
        np.clip((hum - 40) / 60, 0, 1) * 35 +
        np.clip((temp - 15) / 25, 0, 1) * 20 +
        prev * 10 +
        density * 8 +
        water * 7 +
        np.random.uniform(-8, 8, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        'temperature': temp, 'humidity': hum, 'prev_occurrence': prev,
        'crop_density': density, 'near_water': water,
        'crop': crop, 'season': season, 'risk_score': score
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"   Generated {n} samples → {DATA_PATH}")

# ── 2. Encode ──────────────────────────────────────────────────────────
encoders = {}
for col in ['crop', 'season']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Map string prev_occurrence if loaded from real CSV
if df['prev_occurrence'].dtype == object:
    prev_map = {'None': 0, 'Light': 1, 'Moderate': 2, 'Severe': 3}
    df['prev_occurrence'] = df['prev_occurrence'].map(prev_map).fillna(0)

feature_cols = ['temperature', 'humidity', 'prev_occurrence', 'crop_density',
                'near_water', 'season_enc', 'crop_enc']
X = df[feature_cols].values
y = df['risk_score'].values
print(f"\n   Samples: {len(X)} | Features: {len(feature_cols)}")
print(f"   Risk score range: {y.min():.1f} – {y.max():.1f}")

# ── 3. Train ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n⏳ Training Random Forest Regressor (risk score 0–100)...")
model = RandomForestRegressor(
    n_estimators=120, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"\n✅ R² Score : {r2:.4f}")
print(f"   MAE      : {mae:.2f} risk points")

# Accuracy as classification (Low / Medium / High)
def to_level(s):
    return 'High' if s > 65 else ('Medium' if s > 35 else 'Low')

true_levels = [to_level(s) for s in y_test]
pred_levels = [to_level(s) for s in y_pred]
clf_acc     = sum(t == p for t, p in zip(true_levels, pred_levels)) / len(true_levels)
print(f"   Level Classification Accuracy: {clf_acc*100:.1f}%")

# Distribution of predicted levels
from collections import Counter
dist = Counter(pred_levels)
print(f"   Predicted distribution → Low: {dist['Low']} | Medium: {dist['Medium']} | High: {dist['High']}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n📊 Feature Importances:")
for feat, imp in importances.items():
    bar = '█' * int(imp * 40)
    print(f"   {feat:<20} {bar} {imp:.3f}")

# ── 4. Save ────────────────────────────────────────────────────────────
model_path    = os.path.join(SAVE_DIR, 'pest_model.pkl')
encoders_path = os.path.join(SAVE_DIR, 'pest_encoders.pkl')
joblib.dump(model,    model_path)
joblib.dump(encoders, encoders_path)
print(f"\n💾 Saved: {model_path}")
print(f"💾 Saved: {encoders_path}")

# ── 5. Sanity check ────────────────────────────────────────────────────
# hot humid season, severe history → expect High
test_in = np.array([[32, 80, 3, 2, 1, 0, 0]])
risk    = float(model.predict(test_in)[0])
level   = to_level(risk)
print(f"\n🧪 Sanity check → hot(32°C), humid(80%), severe prev → Risk: {risk:.0f} ({level})")
print("   Expected: High")
print("\n✅ Training complete! Model ready for Flask API.")
