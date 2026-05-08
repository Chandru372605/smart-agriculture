"""
AgroSense — TensorFlow-Free Disease Detection Training
=======================================================
Trains a scikit-learn classifier on HOG + color histogram features
extracted from real PlantVillage images (39 classes, 55 000+ images).

NO TensorFlow required — runs on Python 3.14+.

Outputs (saved to ml_models/saved/):
  • disease_sklearn.pkl       — trained sklearn pipeline (scaler + classifier)
  • disease_classes.pkl       — list of 39 class name strings

Run from project root:
    .venv\\Scripts\\python.exe ml_models/disease_detection/train_sklearn_hog.py
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import joblib
from PIL import Image

# ── scikit-learn imports ────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, '..', '..', 'datasets', 'disease_images')
SAVE_DIR   = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE       = 64          # resize to 64×64 for fast feature extraction
MAX_PER_CLASS  = 200         # cap per class — keeps memory manageable
RANDOM_STATE   = 42


# ════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ════════════════════════════════════════════════════════════════════════════

def color_histogram(img_arr: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute per-channel colour histogram and return as flat vector."""
    feats = []
    for ch in range(3):  # R, G, B
        hist, _ = np.histogram(img_arr[:, :, ch], bins=bins, range=(0, 256))
        feats.extend(hist / (img_arr.shape[0] * img_arr.shape[1]))  # normalise
    return np.array(feats, dtype=np.float32)


def color_moments(img_arr: np.ndarray) -> np.ndarray:
    """Mean, std, skewness per channel = 9 features."""
    feats = []
    for ch in range(3):
        c = img_arr[:, :, ch].astype(np.float32)
        mean = c.mean()
        std  = c.std() + 1e-6
        skew = float(np.mean(((c - mean) / std) ** 3))
        feats.extend([mean / 255.0, std / 255.0, skew])
    return np.array(feats, dtype=np.float32)


def simple_hog(img_gray: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """
    Lightweight HOG without scikit-image dependency.
    Computes gradient magnitude and orientation histograms over cells.
    """
    h, w = img_gray.shape
    # Gradients
    gx = np.zeros_like(img_gray, dtype=np.float32)
    gy = np.zeros_like(img_gray, dtype=np.float32)
    gx[:, 1:-1] = img_gray[:, 2:].astype(np.float32) - img_gray[:, :-2].astype(np.float32)
    gy[1:-1, :] = img_gray[2:, :].astype(np.float32) - img_gray[:-2, :].astype(np.float32)

    magnitude  = np.sqrt(gx**2 + gy**2)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180).astype(np.float32)

    n_bins = 9
    n_cells_h = h // cell_size
    n_cells_w = w // cell_size
    hog_cells = np.zeros((n_cells_h, n_cells_w, n_bins), dtype=np.float32)

    for i in range(n_cells_h):
        for j in range(n_cells_w):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                                 j*cell_size:(j+1)*cell_size]
            cell_ori = orientation[i*cell_size:(i+1)*cell_size,
                                   j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(cell_ori, bins=n_bins, range=(0, 180),
                                   weights=cell_mag)
            norm = np.sqrt(hist.sum()**2 + 1e-6)
            hog_cells[i, j] = hist / norm

    return hog_cells.flatten()


def extract_features(img_path: str) -> np.ndarray | None:
    """Load image and return concatenated feature vector."""
    try:
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.uint8)

        gray = np.dot(arr[..., :3].astype(np.float32), [0.299, 0.587, 0.114])

        ch   = color_histogram(arr, bins=32)   # 96 features
        cm   = color_moments(arr)              # 9 features
        hog  = simple_hog(gray.astype(np.uint8), cell_size=8)  # 576 features

        return np.concatenate([ch, cm, hog])   # total: 681 features
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_dataset(image_dir: str):
    class_dirs = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ])

    X, y = [], []
    class_names = []

    print(f"\n📂 Found {len(class_dirs)} classes in {image_dir}")
    print(f"   Extracting features (max {MAX_PER_CLASS} images/class)...\n")

    t0 = time.time()
    for cls_idx, cls_name in enumerate(class_dirs):
        cls_dir = os.path.join(image_dir, cls_name)
        files   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Deterministic subsample
        rng = np.random.default_rng(RANDOM_STATE + cls_idx)
        if len(files) > MAX_PER_CLASS:
            files = rng.choice(files, MAX_PER_CLASS, replace=False).tolist()

        extracted = 0
        for fname in files:
            feat = extract_features(os.path.join(cls_dir, fname))
            if feat is not None:
                X.append(feat)
                y.append(cls_idx)
                extracted += 1

        class_names.append(cls_name)
        pct = (cls_idx + 1) / len(class_dirs) * 100
        elapsed = time.time() - t0
        print(f"  [{pct:5.1f}%] {cls_name:<50s} {extracted:3d} imgs  ({elapsed:.0f}s)")

    print(f"\n✅ Dataset loaded: {len(X)} samples | {len(class_names)} classes")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), class_names


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  AgroSense — Sklearn Disease Detection Training")
print("  (TensorFlow-free, Python 3.14 compatible)")
print("=" * 60)

# Check image dir
if not os.path.isdir(IMAGE_DIR):
    print(f"\n❌ Image directory not found: {IMAGE_DIR}")
    print("   Place PlantVillage images in datasets/disease_images/<ClassName>/")
    sys.exit(1)

X, y, class_names = load_dataset(IMAGE_DIR)

# ── Train / test split ──────────────────────────────────────────────────────
print("\n⏳ Splitting data (80/20)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train Random Forest pipeline ────────────────────────────────────────────
print("\n⏳ Training Random Forest (300 trees, n_jobs=-1)…")
t1 = time.time()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced',
    ))
])

pipeline.fit(X_train, y_train)
train_time = time.time() - t1
print(f"✅ Training complete in {train_time:.1f}s")

# ── Evaluate ────────────────────────────────────────────────────────────────
print("\n⏳ Evaluating on test set…")
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")

# Show per-class summary (top/bottom 5 by F1)
report = classification_report(
    y_test, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
per_class = {
    k: v['f1-score']
    for k, v in report.items()
    if k not in ('accuracy', 'macro avg', 'weighted avg')
}
sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
print("\nLowest F1 classes (may need more data):")
for name, f1 in sorted_classes[:5]:
    print(f"  {name:<50s} F1={f1:.2f}")
print("\nHighest F1 classes:")
for name, f1 in sorted_classes[-5:]:
    print(f"  {name:<50s} F1={f1:.2f}")

# ── Save ─────────────────────────────────────────────────────────────────────
model_path   = os.path.join(SAVE_DIR, 'disease_sklearn.pkl')
classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')

joblib.dump(pipeline,    model_path,   compress=3)
joblib.dump(class_names, classes_path)

print(f"\n💾 Saved model  : {model_path}")
print(f"💾 Saved classes: {classes_path}")
print(f"\nModel size: {os.path.getsize(model_path) / 1e6:.1f} MB")
print("\n" + "=" * 60)
print("  Training complete! Restart Flask to use the new model.")
print("=" * 60)
