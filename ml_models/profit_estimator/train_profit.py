# -*- coding: utf-8 -*-
"""
Train: Farm Profit Estimator — Ridge Regression (yield optimisation)
The core profit calculation is formula-based (revenue - costs).
The ML component learns to predict adjusted yield based on inputs.

Dataset: Synthetic (cost-yield relationships from agricultural literature)
Run: python train_profit.py   (from this file's directory)
  OR: python ml_models/profit_estimator/train_profit.py  (from project root)
Training time: < 10 seconds
"""
import os, sys, io
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Force UTF-8 output so emoji print correctly on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'datasets', 'profit_data.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 55)
print("  AgroSense — Profit / Yield Optimiser Training")
print("=" * 55)

# ── 1. Load or generate ────────────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset loaded: {df.shape}")
else:
    print("\n⚠️  Generating synthetic cost-yield training data...")
    print("   (Based on diminishing-returns relationships from agri literature)")
    np.random.seed(42)
    n = 3000
    area     = np.random.uniform(1, 100, n)
    fert_ha  = np.random.uniform(1000, 12000, n)
    irr_ha   = np.random.uniform(500, 8000, n)
    pest_ha  = np.random.uniform(500, 5000, n)

    # Yield responds to fertiliser and irrigation (diminishing returns via log)
    yield_ha = (
        2.5
        + np.log1p(fert_ha / 1000) * 0.8
        + np.log1p(irr_ha  / 1000) * 0.5
        + np.log1p(pest_ha / 1000) * 0.3
        + np.random.normal(0, 0.3, n)
    )
    yield_ha = np.clip(yield_ha, 0.5, 12.0)

    df = pd.DataFrame({
        'area':             area,
        'fertiliser_cost':  fert_ha,
        'irrigation_cost':  irr_ha,
        'pesticide_cost':   pest_ha,
        'yield_per_ha':     yield_ha
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"   Generated {n} samples → {DATA_PATH}")

print(f"\n   Yield range: {df['yield_per_ha'].min():.2f} – {df['yield_per_ha'].max():.2f} t/ha")
print(f"   Mean yield : {df['yield_per_ha'].mean():.2f} t/ha")

# ── 2. Train ───────────────────────────────────────────────────────────
features = ['area', 'fertiliser_cost', 'irrigation_cost', 'pesticide_cost']
X = df[features].values
y = df['yield_per_ha'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n⏳ Training Ridge Regression (yield optimiser)...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge',  Ridge(alpha=1.0))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"\n✅ R² Score : {r2:.4f}")
print(f"   MAE      : {mae:.3f} t/ha")

# Coefficients (after scaling — directional insight only)
ridge_coef = model.named_steps['ridge'].coef_
print("\n📊 Ridge Coefficients (scaled features — direction matters):")
for feat, coef in zip(features, ridge_coef):
    direction = '↑' if coef > 0 else '↓'
    print(f"   {feat:<22} {direction}  {coef:+.4f}")

# ── 3. Save ────────────────────────────────────────────────────────────
save_path = os.path.join(SAVE_DIR, 'profit_model.pkl')
joblib.dump(model, save_path)
print(f"\n💾 Saved: {save_path}")

# ── 4. Sanity check ────────────────────────────────────────────────────
test_cases = [
    ([5.0, 5000, 2500, 2000],  "5ha, moderate inputs"),
    ([10.0, 10000, 6000, 3000], "10ha, high inputs"),
    ([2.0, 1500, 800, 600],    "2ha, low inputs"),
]
print("\n🧪 Sanity checks:")
for inputs, label in test_cases:
    pred = float(model.predict([inputs])[0])
    print(f"   {label:<26} → {pred:.2f} t/ha")

print("\n✅ Training complete! Model ready for Flask API.")
