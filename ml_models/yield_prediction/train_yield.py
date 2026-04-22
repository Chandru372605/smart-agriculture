# -*- coding: utf-8 -*-
"""
Train: Yield Prediction — XGBoost / GradientBoosting Regressor
Dataset: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
  Place as: datasets/crop_production.csv
  Columns needed: Crop, Season, State_Name, Area, Production, Annual_Rainfall,
                  Fertilizer, Pesticide (some columns may need mapping)

Run: python train_yield.py   (from this file's directory)
  OR: python ml_models/yield_prediction/train_yield.py  (from project root)
Expected R² score: ~0.85+
Training time: 1–2 minutes
"""
import os, sys, io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Force UTF-8 output so emoji print correctly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Try to import XGBoost; fall back to GradientBoosting
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("⚠️  XGBoost not installed. Using GradientBoostingRegressor instead.")
    print("   Install: pip install xgboost")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'datasets', 'crop_production.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 55)
print("  AgroSense — Yield Prediction Model Training")
print("=" * 55)

# ── 1. Load or synthesise data ─────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset loaded: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # Standardise column names (Kaggle dataset uses these names)
    rename_map = {
        'Crop': 'crop', 'Season': 'season', 'State_Name': 'state',
        'Area': 'area', 'Production': 'production',
        'Annual_Rainfall': 'rainfall', 'Fertilizer': 'fertiliser',
        'Pesticide': 'pesticide'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.columns = [c.lower().strip() for c in df.columns]

    # Compute yield per hectare
    df = df.dropna(subset=['production', 'area'])
    df = df[df['area'] > 0]
    df['yield_per_ha'] = df['production'] / df['area']
    df = df[df['yield_per_ha'] < df['yield_per_ha'].quantile(0.99)]  # remove outliers
    df = df[df['yield_per_ha'] > 0]

    # Fill missing numerics
    for col in ['rainfall', 'fertiliser', 'pesticide']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 500 if col == 'rainfall' else 100

    print(f"   After cleaning: {df.shape[0]} rows")

else:
    print("\n⚠️  No CSV found — generating synthetic yield data...")
    print("   (Download real data from Kaggle for better accuracy)")
    np.random.seed(42)
    crops   = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 'Groundnut']
    seasons = ['Kharif', 'Rabi', 'Zaid', 'Whole Year']
    states  = ['Punjab', 'Uttar Pradesh', 'Maharashtra', 'Andhra Pradesh', 'West Bengal', 'Karnataka']
    BASE_YIELD = {'Rice': 3.8, 'Wheat': 4.2, 'Maize': 5.1, 'Cotton': 2.1,
                  'Sugarcane': 68.0, 'Soybean': 2.8, 'Groundnut': 2.2}
    rows = []
    for _ in range(5000):
        c  = np.random.choice(crops)
        s  = np.random.choice(seasons)
        st = np.random.choice(states)
        area = np.random.uniform(1, 200)
        rain = np.random.uniform(300, 2500)
        fert = np.random.uniform(20, 300)
        pest = np.random.uniform(0, 5)
        irr  = np.random.randint(0, 3)
        base = BASE_YIELD[c]
        y_ha = base * (0.8 + rain / 5000 + fert / 600 + irr * 0.15 + np.random.normal(0, 0.2))
        y_ha = max(0.1, y_ha)
        rows.append({'crop': c, 'season': s, 'state': st, 'area': area, 'rainfall': rain,
                     'fertiliser': fert, 'pesticide': pest, 'irrigation': irr, 'yield_per_ha': y_ha})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"   Generated {len(df)} samples → saved to {DATA_PATH}")

# ── 2. Encode categoricals ─────────────────────────────────────────────
encoders = {}
for col in ['crop', 'season', 'state']:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# ── 3. Feature matrix ──────────────────────────────────────────────────
feature_cols = ['crop_enc', 'season_enc', 'state_enc', 'area', 'rainfall', 'fertiliser']
if 'pesticide' in df.columns:
    feature_cols.append('pesticide')
if 'irrigation' in df.columns:
    feature_cols.append('irrigation')

feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].values
y = df['yield_per_ha'].values
print(f"\n   Features used: {feature_cols}")
print(f"   Samples: {len(X)}")

# ── 4. Train ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

algo_name = 'XGBoost' if USE_XGB else 'GradientBoosting'
print(f"\n⏳ Training {algo_name} Regressor...")

if USE_XGB:
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbosity=0
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"\n✅ R² Score : {r2:.4f}")
print(f"   MAE      : {mae:.4f} tonnes/ha")

# Feature importances (if available)
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n📊 Feature Importances:")
    for feat, imp in importances.items():
        bar = '█' * int(imp * 40)
        print(f"   {feat:<20} {bar} {imp:.3f}")

# ── 5. Save ────────────────────────────────────────────────────────────
model_path    = os.path.join(SAVE_DIR, 'yield_model.pkl')
encoders_path = os.path.join(SAVE_DIR, 'yield_encoders.pkl')
joblib.dump(model,    model_path)
joblib.dump(encoders, encoders_path)
print(f"\n💾 Saved: {model_path}")
print(f"💾 Saved: {encoders_path}")
print("\n✅ Training complete! Model ready for Flask API.")
