"""
AgroSense — Download PlantVillage dataset from Kaggle + Train CNN
Run from project root:
    venv\Scripts\python.exe ml_models/disease_detection/download_and_train.py
Requires: kaggle.json at C:\Users\<you>\.kaggle\kaggle.json
"""
import os, sys, subprocess, zipfile, shutil

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, 'datasets', 'disease_images')
SAVE_DIR  = os.path.join(BASE_DIR, 'ml_models', 'saved')
ZIP_PATH  = os.path.join(BASE_DIR, 'datasets', 'plantdisease.zip')

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("=" * 60)
print("  AgroSense — PlantVillage Download + CNN Training")
print("=" * 60)

# ─── STEP 1: Check Kaggle credentials ────────────────────────────
kaggle_json = os.path.expanduser(r'~\.kaggle\kaggle.json')
if not os.path.exists(kaggle_json):
    print(f"\n❌ Kaggle credentials not found at:\n   {kaggle_json}")
    print("\nTo fix:")
    print("  1. Go to https://www.kaggle.com → Settings → API → Create New Token")
    print("  2. Move downloaded kaggle.json to:", kaggle_json)
    sys.exit(1)
print(f"\n✅ Kaggle credentials found")

# ─── STEP 2: Check if already extracted ──────────────────────────
existing_classes = [d for d in os.listdir(IMAGE_DIR)
                    if os.path.isdir(os.path.join(IMAGE_DIR, d))]
if len(existing_classes) >= 20:
    print(f"✅ Dataset already extracted: {len(existing_classes)} classes found")
else:
    # ─── STEP 2a: Download ──────────────────────────────────────
    print("\n⬇️  Downloading PlantVillage dataset from Kaggle...")
    print("    Dataset: emmarex/plantdisease (~1.5 GB) — please wait...")
    result = subprocess.run(
        [sys.executable, '-m', 'kaggle', 'datasets', 'download',
         'emmarex/plantdisease',
         '--path', os.path.join(BASE_DIR, 'datasets'),
         '--unzip'],
        capture_output=False
    )
    if result.returncode != 0:
        # Try alternative dataset
        print("\n⚠  Trying alternative dataset: abdallahalidev/plantvillage-dataset...")
        result = subprocess.run(
            [sys.executable, '-m', 'kaggle', 'datasets', 'download',
             'abdallahalidev/plantvillage-dataset',
             '--path', os.path.join(BASE_DIR, 'datasets'),
             '--unzip'],
            capture_output=False
        )

    # ─── STEP 2b: Organise folder structure ─────────────────────
    print("\n🗂️  Organising extracted files...")
    datasets_dir = os.path.join(BASE_DIR, 'datasets')

    # Find any folder containing disease-named subdirs and move them to IMAGE_DIR
    for root, dirs, files in os.walk(datasets_dir):
        for d in dirs:
            if '___' in d:   # PlantVillage class naming convention
                src = os.path.join(root, d)
                dst = os.path.join(IMAGE_DIR, d)
                if not os.path.exists(dst):
                    print(f"  Moving: {d}")
                    shutil.move(src, dst)

    existing_classes = [d for d in os.listdir(IMAGE_DIR)
                        if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    print(f"\n✅ Extracted: {len(existing_classes)} disease classes")

# ─── STEP 3: Train CNN ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting MobileNetV2 CNN Training")
print("=" * 60)

import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 15

print(f"   TensorFlow: {tf.__version__}")
print(f"   Classes:    {len(existing_classes)}")
print(f"   Image dir:  {IMAGE_DIR}")

# Count total images
total_imgs = sum(
    len([f for f in os.listdir(os.path.join(IMAGE_DIR, c))
         if f.lower().endswith(('.jpg','.jpeg','.png'))])
    for c in existing_classes
)
print(f"   Total images: {total_imgs:,}")

# Data generators with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)
train_ds = train_gen.flow_from_directory(
    IMAGE_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training', shuffle=True, seed=42
)
val_ds = train_gen.flow_from_directory(
    IMAGE_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', seed=42
)
class_names = list(train_ds.class_indices.keys())
n_classes   = len(class_names)
print(f"   Train batches: {len(train_ds)} | Val batches: {len(val_ds)}")

# ── Phase 1: Frozen base ─────────────────────────────────────────
print("\n🔧 Phase 1 — Frozen MobileNetV2 base (feature extraction)")
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)
model   = keras.Model(inputs, outputs, name='AgroSense_DiseaseCNN')
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR, 'disease_cnn_checkpoint.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=0),
]

phase1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS, callbacks=callbacks, verbose=1)
p1_best = max(phase1.history.get('val_accuracy', [0]))
print(f"\n✅ Phase 1 best val accuracy: {p1_best*100:.1f}%")

# ── Phase 2: Fine-tune top layers ────────────────────────────────
print("\n🔧 Phase 2 — Fine-tuning top 40 layers")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(5e-6),
              loss='categorical_crossentropy', metrics=['accuracy'])
phase2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=8, callbacks=callbacks, verbose=1)
p2_best = max(phase2.history.get('val_accuracy', [0]))
print(f"\n✅ Phase 2 best val accuracy: {p2_best*100:.1f}%")

# ── Save ─────────────────────────────────────────────────────────
model_path   = os.path.join(SAVE_DIR, 'disease_cnn.keras')
classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')
model.save(model_path)
joblib.dump(class_names, classes_path)
print(f"\n💾 Saved model:   {model_path}")
print(f"💾 Saved classes: {classes_path}")

final_acc = max(p1_best, p2_best)
print("\n" + "=" * 60)
print(f"  ✅ CNN Training Complete!")
print(f"     Final accuracy: {final_acc*100:.1f}%")
print(f"     Classes trained: {n_classes}")
print(f"     Model size: {os.path.getsize(model_path)//1024//1024} MB")
print("=" * 60)
print("\nThe /api/disease/predict endpoint will now use this CNN automatically.")
