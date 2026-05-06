"""
AgroSense — CNN Disease Detection Training (Synthetic Data Mode)
================================================================
Trains a real MobileNetV2 CNN on programmatically-generated leaf images.
Each class gets 200 synthetic images rendered with realistic colour profiles,
noise, and texture patterns matching real disease symptoms.

This produces a fully functional disease_cnn.keras model that works with
the existing /api/disease/predict endpoint — no Kaggle download needed.

If real PlantVillage images ARE present in datasets/disease_images/,
this script uses those instead (automatically detected).

Run from project root:
    venv/Scripts/python.exe ml_models/disease_detection/train_cnn_synthetic.py
"""
import os, sys, io, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, '..', '..', 'datasets', 'disease_images')
SAVE_DIR  = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Check for real PlantVillage images ──────────────────────────────
image_classes = []
if os.path.isdir(IMAGE_DIR):
    image_classes = [d for d in os.listdir(IMAGE_DIR)
                     if os.path.isdir(os.path.join(IMAGE_DIR, d))]

USE_REAL = len(image_classes) >= 5

print("=" * 60)
print("  AgroSense — Disease CNN Training")
print("=" * 60)

# ════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA DEFINITIONS
# ════════════════════════════════════════════════════════════════════
CLASSES = [
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Apple___Apple_scab',
    'Apple___healthy',
    'Potato___Late_blight',
    'Potato___healthy',
    'Corn___Common_rust',
    'Corn___healthy',
    'Rice___healthy',
]

# Realistic HSV colour profiles per class
# (base_hue, hue_std, sat_range, val_range, lesion_prob, lesion_color)
PROFILES = {
    'Tomato___Late_blight':  dict(hue=90,  hs=15, sat=(0.3,0.5), val=(0.25,0.45), lp=0.7, lc=(10,60,40)),
    'Tomato___Early_blight': dict(hue=85,  hs=18, sat=(0.3,0.5), val=(0.28,0.48), lp=0.6, lc=(25,70,50)),
    'Tomato___healthy':      dict(hue=115, hs=12, sat=(0.5,0.8), val=(0.45,0.70), lp=0.0, lc=None),
    'Apple___Apple_scab':    dict(hue=95,  hs=20, sat=(0.25,0.45),val=(0.20,0.40),lp=0.65,lc=(18,55,35)),
    'Apple___healthy':       dict(hue=118, hs=10, sat=(0.55,0.80),val=(0.50,0.72),lp=0.0, lc=None),
    'Potato___Late_blight':  dict(hue=88,  hs=16, sat=(0.30,0.50),val=(0.22,0.42),lp=0.72,lc=(12,58,38)),
    'Potato___healthy':      dict(hue=112, hs=12, sat=(0.48,0.75),val=(0.42,0.65),lp=0.0, lc=None),
    'Corn___Common_rust':    dict(hue=20,  hs=25, sat=(0.45,0.70),val=(0.40,0.65),lp=0.80,lc=(18,70,55)),
    'Corn___healthy':        dict(hue=110, hs=14, sat=(0.50,0.78),val=(0.44,0.68),lp=0.0, lc=None),
    'Rice___healthy':        dict(hue=108, hs=12, sat=(0.45,0.72),val=(0.45,0.68),lp=0.0, lc=None),
}

IMG_SIZE   = 128
N_PER_CLASS = 220    # images per class for training
N_VAL       = 40     # images per class for validation


def hsv_to_rgb(h, s, v):
    """Convert HSV (0-360, 0-1, 0-1) to RGB uint8 array pixel."""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if   h < 60:  r,g,b = c,x,0
    elif h < 120: r,g,b = x,c,0
    elif h < 180: r,g,b = 0,c,x
    elif h < 240: r,g,b = 0,x,c
    elif h < 300: r,g,b = x,0,c
    else:          r,g,b = c,0,x
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))


def generate_leaf_image(cls_name, rng):
    """Generate a synthetic 128×128 RGB leaf image for the given class."""
    p = PROFILES[cls_name]

    # Base leaf colour (green channel dominant for healthy, degraded for diseased)
    hue = rng.normal(p['hue'], p['hs'])
    sat = rng.uniform(*p['sat'])
    val = rng.uniform(*p['val'])

    # Build base RGB image with per-pixel hue variation
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    hue_map = rng.normal(hue, p['hs']*0.4, (IMG_SIZE, IMG_SIZE))
    sat_map = rng.uniform(p['sat'][0], p['sat'][1], (IMG_SIZE, IMG_SIZE))
    val_map = rng.uniform(p['val'][0], p['val'][1], (IMG_SIZE, IMG_SIZE))

    for i in range(IMG_SIZE):
        for j in range(0, IMG_SIZE, 4):  # vectorise a bit
            chunk = min(4, IMG_SIZE - j)
            for k in range(chunk):
                r, g, b = hsv_to_rgb(hue_map[i, j+k], sat_map[i, j+k], val_map[i, j+k])
                img[i, j+k] = [r/255, g/255, b/255]

    # Add Gaussian noise (texture)
    img += rng.normal(0, 0.025, img.shape)

    # Add lesion patches for diseased classes
    if p['lp'] > 0 and rng.random() < p['lp']:
        lc = np.array(p['lc'], dtype=np.float32) / 255.0
        n_lesions = rng.integers(2, 7)
        for _ in range(n_lesions):
            cx = rng.integers(10, IMG_SIZE - 10)
            cy = rng.integers(10, IMG_SIZE - 10)
            r_sz = rng.integers(4, 14)
            for di in range(-r_sz, r_sz+1):
                for dj in range(-r_sz, r_sz+1):
                    if di**2 + dj**2 <= r_sz**2:
                        ni, nj = cx+di, cy+dj
                        if 0 <= ni < IMG_SIZE and 0 <= nj < IMG_SIZE:
                            blend = rng.uniform(0.4, 0.9)
                            img[ni, nj] = img[ni, nj] * (1-blend) + lc * blend

    # Add elliptical leaf mask (edges darker/transparent)
    cy_c, cx_c = IMG_SIZE // 2, IMG_SIZE // 2
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = ((Y - cy_c)**2 / (cy_c*0.85)**2 + (X - cx_c)**2 / (cx_c*0.85)**2) <= 1
    img[~mask] = img[~mask] * 0.3 + 0.05  # darken background

    return np.clip(img, 0, 1)


def build_dataset(n_per_class, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        print(f"  Generating {n_per_class} images for: {cls}")
        for _ in range(n_per_class):
            img = generate_leaf_image(cls, rng)
            X.append(img)
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if USE_REAL:
    print(f"\n✅ Real PlantVillage images found: {len(image_classes)} classes")
    print("   Switching to real-data CNN training mode…")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    gen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                             width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, zoom_range=0.15,
                             validation_split=0.2)
    train_ds = gen.flow_from_directory(IMAGE_DIR, target_size=(IMG_SIZE,IMG_SIZE),
                                       batch_size=32, class_mode='categorical',
                                       subset='training', shuffle=True, seed=42)
    val_ds   = gen.flow_from_directory(IMAGE_DIR, target_size=(IMG_SIZE,IMG_SIZE),
                                       batch_size=32, class_mode='categorical',
                                       subset='validation', seed=42)
    class_names = list(train_ds.class_indices.keys())
    n_classes   = len(class_names)
else:
    print(f"\n⚡ Synthetic mode — generating {N_PER_CLASS} images × {len(CLASSES)} classes")
    print("   (Place real PlantVillage images in datasets/disease_images/ for full CNN)")

    print("\n⏳ Building training set…")
    X_train, y_train = build_dataset(N_PER_CLASS, rng_seed=42)
    print("⏳ Building validation set…")
    X_val,   y_val   = build_dataset(N_VAL, rng_seed=99)
    class_names = CLASSES
    n_classes   = len(CLASSES)
    print(f"\n   Train: {len(X_train)} images | Val: {len(X_val)} images")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, tf.keras.utils.to_categorical(y_train, n_classes)))
    train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   tf.keras.utils.to_categorical(y_val,   n_classes)))
    val_ds   = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

print(f"\n   TensorFlow: {tf.__version__}")
print(f"   Classes:    {n_classes}")

# ── Build MobileNetV2 model ──────────────────────────────────────────
print("\n🔧 Building MobileNetV2 transfer-learning model…")
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)
model   = keras.Model(inputs, outputs, name='AgroSense_DiseaseCNN')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary(print_fn=lambda x: None)  # suppress verbose summary

# ── Train ────────────────────────────────────────────────────────────
EPOCHS = 12
print(f"\n⏳ Training {EPOCHS} epochs (frozen MobileNetV2 base)…")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=2, verbose=0),
]
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks, verbose=1)

best_val_acc = max(history.history.get('val_accuracy', [0]))
print(f"\n✅ Best val accuracy: {best_val_acc*100:.1f}%")

# ── Fine-tune top layers ─────────────────────────────────────────────
print("\n🔧 Fine-tuning top 30 layers of MobileNetV2…")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=6, callbacks=callbacks, verbose=1)

best_val_acc2 = max(history2.history.get('val_accuracy', [0]))
print(f"\n✅ Fine-tuned val accuracy: {best_val_acc2*100:.1f}%")

# ── Save ─────────────────────────────────────────────────────────────
model_path   = os.path.join(SAVE_DIR, 'disease_cnn.keras')
classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')
model.save(model_path)
joblib.dump(class_names, classes_path)

print(f"\n💾 Saved: {model_path}")
print(f"💾 Saved: {classes_path}")
print("\n" + "=" * 60)
print("  CNN training complete!")
if not USE_REAL:
    print("  ⚠  Trained on synthetic data — accuracy will improve with")
    print("     real PlantVillage images (place in datasets/disease_images/).")
print("=" * 60)
