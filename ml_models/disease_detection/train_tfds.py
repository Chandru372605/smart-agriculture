"""
AgroSense — Plant Disease CNN Training via TensorFlow Datasets
==============================================================
Downloads PlantVillage automatically (no Kaggle login needed)
via tensorflow_datasets, then trains MobileNetV2.

Run from project root:
    venv\Scripts\python.exe ml_models/disease_detection/train_tfds.py

Dataset: ~800 MB, 54,306 images, 38 classes (plant___condition).
Training time: ~25-50 min on CPU, ~5-8 min on GPU.
"""
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR  = os.path.join(BASE_DIR, 'ml_models', 'saved')
DATA_DIR  = os.path.join(BASE_DIR, 'datasets', 'tfds_cache')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("  AgroSense — Disease CNN (TensorFlow Datasets)")
print("=" * 60)

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

print(f"\n   TensorFlow:         {tf.__version__}")
print(f"   TF-Datasets:        {tfds.__version__}")

IMG_SIZE   = 128
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# ─── STEP 1: Load PlantVillage dataset ───────────────────────────
print("\n⬇️  Loading PlantVillage dataset (downloads on first run ~800MB)...")
print("   This may take several minutes depending on your connection.\n")

(ds_train, ds_val), info = tfds.load(
    'plant_village',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True,
    data_dir=DATA_DIR,
)

n_classes   = info.features['label'].num_classes
class_names = info.features['label'].names
total_train = info.splits['train'].num_examples
n_train     = int(total_train * 0.8)
n_val       = total_train - n_train

print(f"   Classes:      {n_classes}")
print(f"   Train images: {n_train:,}")
print(f"   Val images:   {n_val:,}")
print(f"   Sample classes: {class_names[:4]}...")

# ─── STEP 2: Preprocess & augment ────────────────────────────────
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, n_classes)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

ds_train = (ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .map(augment,    num_parallel_calls=AUTOTUNE)
    .shuffle(2000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

ds_val = (ds_val
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# ─── STEP 3: Build MobileNetV2 model ─────────────────────────────
print("\n🔧 Building MobileNetV2 transfer-learning model...")
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)
model   = keras.Model(inputs, outputs, name='AgroSense_DiseaseCNN')

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"   Parameters: {model.count_params():,} ({model.count_params()//1e6:.1f}M)")

# ─── STEP 4: Phase 1 — Feature extraction ────────────────────────
EPOCHS_P1 = 12
print(f"\n⏳ Phase 1 — Feature extraction ({EPOCHS_P1} epochs, frozen base)...")
ckpt_path = os.path.join(SAVE_DIR, 'disease_cnn_ckpt.keras')

callbacks_p1 = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=2, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                                    save_best_only=True, verbose=0),
]

h1 = model.fit(ds_train, validation_data=ds_val,
               epochs=EPOCHS_P1, callbacks=callbacks_p1)
best_p1 = max(h1.history['val_accuracy'])
print(f"\n✅ Phase 1 best val accuracy: {best_p1*100:.1f}%")

# ─── STEP 5: Phase 2 — Fine-tune top layers ──────────────────────
EPOCHS_P2 = 8
print(f"\n🔧 Phase 2 — Fine-tuning top 40 MobileNetV2 layers ({EPOCHS_P2} epochs)...")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=2, min_lr=1e-8, verbose=1),
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                                    save_best_only=True, verbose=0),
]

h2 = model.fit(ds_train, validation_data=ds_val,
               epochs=EPOCHS_P2, callbacks=callbacks_p2)
best_p2 = max(h2.history['val_accuracy'])
print(f"\n✅ Phase 2 best val accuracy: {best_p2*100:.1f}%")

# ─── STEP 6: Save model + classes ────────────────────────────────
model_path   = os.path.join(SAVE_DIR, 'disease_cnn.keras')
classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')

model.save(model_path)
joblib.dump(class_names, classes_path)

# Clean up checkpoint
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

final_acc = max(best_p1, best_p2)
model_mb  = os.path.getsize(model_path) // (1024*1024)

print(f"\n💾 Saved: {model_path} ({model_mb} MB)")
print(f"💾 Saved: {classes_path}")
print("\n" + "=" * 60)
print(f"  CNN Training Complete!")
print(f"  Final accuracy : {final_acc*100:.1f}%")
print(f"  Classes        : {n_classes}")
print(f"  Model size     : {model_mb} MB")
print("=" * 60)
print("\nThe /api/disease/predict endpoint now uses this CNN model.")
