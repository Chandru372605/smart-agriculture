# -*- coding: utf-8 -*-
"""
Train: Smart Irrigation — Decision Tree Classifier
Dataset: https://www.kaggle.com/datasets/nelakurthisudheer/dataset-for-predicting-watering-the-plants
  OR uses synthetic data if CSV not found (still useful for demo)

Run: python train_irrigation.py   (from this file's directory)
  OR: python ml_models/irrigation/train_irrigation.py  (from project root)
Expected accuracy: ~92%
Training time: < 10 seconds
"""
import os, sys, io
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Force UTF-8 output so emoji print correctly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'datasets', 'irrigation.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

CROP_MAP  = {'Rice':0,'Wheat':1,'Maize':2,'Cotton':3,'Sugarcane':4,'Tomato':5,'Potato':6}
STAGE_MAP = {'Germination':0,'Vegetative':1,'Flowering':2,'Fruiting':3,'Maturity':4}

print("=" * 55)
print("  AgroSense — Smart Irrigation Model Training")
print("=" * 55)

# ── 1. Load or generate data ───────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset loaded: {df.shape}")
else:
    print("\n⚠️  No CSV found — generating synthetic training data...")
    print("   (Download real data from Kaggle for better accuracy)")
    np.random.seed(42)
    n = 3000
    soil_m  = np.random.uniform(10, 90, n)
    temp    = np.random.uniform(15, 45, n)
    hum     = np.random.uniform(20, 95, n)
    rain    = np.random.uniform(0, 80, n)
    crop    = np.random.randint(0, 7, n)
    stage   = np.random.randint(0, 5, n)

    # Rule: irrigate if low moisture, not enough rain, high temp
    label = ((soil_m < 50) & (rain < 12) & (temp > 25)).astype(int)
    # Add realistic noise
    flip  = np.random.random(n) < 0.05
    label = np.where(flip, 1 - label, label)

    df = pd.DataFrame({
        'soil_moisture': soil_m, 'temperature': temp, 'humidity': hum,
        'rainfall_forecast': rain, 'crop': crop, 'stage': stage, 'irrigate': label
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"   Generated {n} samples → saved to {DATA_PATH}")

# ── 2. Prepare features ────────────────────────────────────────────────
feature_cols = ['soil_moisture','temperature','humidity','rainfall_forecast','crop','stage']
target_col   = 'irrigate'

# Encode string columns if present
if df['crop'].dtype == object:
    df['crop']  = df['crop'].map(CROP_MAP).fillna(0).astype(int)
if df['stage'].dtype == object:
    df['stage'] = df['stage'].map(STAGE_MAP).fillna(1).astype(int)

X = df[feature_cols].values
y = df[target_col].values
print(f"\n   Positive (irrigate): {y.sum()} | Negative (skip): {len(y)-y.sum()}")

# ── 3. Train ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n⏳ Training Decision Tree...")
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy : {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Skip','Irrigate']))

# ── 4. Save ────────────────────────────────────────────────────────────
save_path = os.path.join(SAVE_DIR, 'irrigation_model.pkl')
joblib.dump(model, save_path)
print(f"\n💾 Saved: {save_path}")

# Sanity check
test_input = np.array([[30, 35, 60, 3, 0, 1]])  # Low moisture, hot, no rain → irrigate
pred = model.predict(test_input)[0]
print(f"\n🧪 Sanity check → soil_moisture=30%, rain=3mm → {'Irrigate ✅' if pred else 'Skip ❌'}")
print("   Expected: Irrigate")
print("\n✅ Training complete!")
