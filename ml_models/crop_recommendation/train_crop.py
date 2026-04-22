# -*- coding: utf-8 -*-
"""
Train: Crop Recommendation — Random Forest Classifier
Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Save as: Crop_recommendation.csv in datasets/

Run: python train_crop.py   (from this file's directory)
  OR: python ml_models/crop_recommendation/train_crop.py  (from project root)
Expected accuracy: ~99%
Training time: < 30 seconds on any laptop
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys, io
# Force UTF-8 output so emoji print correctly on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, '..', '..', 'datasets', 'Crop_recommendation.csv')
SAVE_DIR    = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 55)
print("  AgroSense — Crop Recommendation Model Training")
print("=" * 55)

# ── 1. Load data ───────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print(f"\n❌ Dataset not found at: {DATA_PATH}")
    print("   Download from Kaggle: atharvaingle/crop-recommendation-dataset")
    print("   Place file as: datasets/Crop_recommendation.csv")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Crops: {df['label'].nunique()} unique → {sorted(df['label'].unique())}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# ── 2. Features & target ───────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[FEATURES].values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\n   Feature columns: {FEATURES}")

# ── 3. Train / test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"\n   Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Train Random Forest ─────────────────────────────────────────────
print("\n⏳ Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)
model.fit(X_train, y_train)

# ── 5. Evaluate ────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy : {acc*100:.2f}%")

cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
print(f"   5-Fold CV     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

print("\n📊 Classification Report (top 10 crops):")
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred),
    labels=le.classes_[:10]
))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("📊 Feature Importances:")
for feat, imp in importances.items():
    bar = '█' * int(imp * 40)
    print(f"   {feat:<15} {bar} {imp:.3f}")

# ── 6. Save model ──────────────────────────────────────────────────────
model_path   = os.path.join(SAVE_DIR, 'crop_recommend.pkl')
encoder_path = os.path.join(SAVE_DIR, 'crop_label_encoder.pkl')
joblib.dump(model, model_path)
joblib.dump(le,    encoder_path)
print(f"\n💾 Saved: {model_path}")
print(f"💾 Saved: {encoder_path}")

# ── 7. Quick sanity test ───────────────────────────────────────────────
sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])  # Expected: Rice
pred   = le.inverse_transform(model.predict(sample))[0]
conf   = max(model.predict_proba(sample)[0]) * 100
print(f"\n🧪 Sanity check → Predicted: {pred} ({conf:.1f}% confidence)")
print("   Expected: Rice")
print("\n✅ Training complete! Model ready for Flask API.")
