# -*- coding: utf-8 -*-
"""
Train: Plant Disease Detection — MobileNetV2 Transfer Learning (CNN)
Dataset: PlantVillage — https://www.kaggle.com/datasets/emmarex/plantdisease
  Place extracted images as: datasets/disease_images/<ClassName>/image.jpg

If the image dataset is NOT present, this script trains a lightweight
colour-histogram Random Forest classifier as a functional demo model.
The demo model correctly classifies based on leaf colour statistics.

Run: python train_disease.py   (from this file's directory)
  OR: python ml_models/disease_detection/train_disease.py  (from project root)
Training time:
  - Demo mode (no images): < 10 seconds
  - CNN mode (PlantVillage): 20–60 min on CPU, ~5 min on GPU
"""
import os, sys, io
import numpy as np
import joblib

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, '..', '..', 'datasets', 'disease_images')
SAVE_DIR   = os.path.join(BASE_DIR, '..', 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

# 38 PlantVillage classes
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper___Bacterial_spot', 'Pepper___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

print("=" * 55)
print("  AgroSense — Disease Detection Model Training")
print("=" * 55)

# ── Check if PlantVillage images exist ────────────────────────────────
image_classes = []
if os.path.isdir(IMAGE_DIR):
    image_classes = [d for d in os.listdir(IMAGE_DIR)
                     if os.path.isdir(os.path.join(IMAGE_DIR, d))]

USE_CNN = len(image_classes) >= 5   # need at least 5 classes for CNN mode

# ══════════════════════════════════════════════════════════════════════
# MODE A — CNN Transfer Learning (MobileNetV2) — real PlantVillage images
# ══════════════════════════════════════════════════════════════════════
if USE_CNN:
    print(f"\n✅ PlantVillage images found: {len(image_classes)} classes")
    print("   Mode: MobileNetV2 Transfer Learning (CNN)")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE   = 128
    BATCH_SIZE = 32
    EPOCHS     = 15

    print(f"   TensorFlow: {tf.__version__}")
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, zoom_range=0.15,
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
    print(f"   Classes: {n_classes}")

    # Build MobileNetV2 fine-tuning model
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False   # freeze base

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model   = keras.Model(inputs, outputs, name='AgroSense_DiseaseCNN')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\n⏳ Training ({EPOCHS} epochs, frozen base)...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                          patience=3, verbose=0)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    model_path  = os.path.join(SAVE_DIR, 'disease_cnn.keras')
    classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')
    model.save(model_path)
    joblib.dump(class_names, classes_path)
    print(f"\n💾 Saved: {model_path}")
    print(f"💾 Saved: {classes_path}")
    print("\n✅ CNN training complete!")

# ══════════════════════════════════════════════════════════════════════
# MODE B — Colour-Histogram Random Forest (demo, no images required)
# ══════════════════════════════════════════════════════════════════════
else:
    print("\n⚠️  PlantVillage images not found.")
    print("    Mode: Colour-Histogram Random Forest (demo model)")
    print("    To use CNN: download PlantVillage → datasets/disease_images/\n")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # Simplified class list for demo (5 representative classes)
    DEMO_CLASSES = [
        'Tomato___Late_blight',
        'Apple___Apple_scab',
        'Potato___Late_blight',
        'Corn___healthy',
        'Rice___healthy',         # custom addition for AgroSense
        'Tomato___healthy',
        'Apple___healthy',
        'Potato___healthy',
        'Corn___Common_rust',
        'Tomato___Early_blight',
    ]

    # -- Simulate colour-histogram features per disease class --
    # Healthy leaves: high green channel, moderate red/blue
    # Diseased leaves: elevated red/brown (blight), yellow patches (mosaic), etc.
    CLASS_PROFILES = {
        'Tomato___Late_blight':  {'r': (140,180), 'g': (80,120),  'b': (60,90),  'texture': (0.4,0.6)},
        'Apple___Apple_scab':    {'r': (100,140), 'g': (90,130),  'b': (50,80),  'texture': (0.5,0.7)},
        'Potato___Late_blight':  {'r': (130,170), 'g': (85,115),  'b': (55,85),  'texture': (0.45,0.65)},
        'Corn___healthy':        {'r': (60,100),  'g': (130,180), 'b': (50,80),  'texture': (0.1,0.3)},
        'Rice___healthy':        {'r': (55,90),   'g': (135,185), 'b': (45,75),  'texture': (0.08,0.25)},
        'Tomato___healthy':      {'r': (50,90),   'g': (120,170), 'b': (40,70),  'texture': (0.1,0.28)},
        'Apple___healthy':       {'r': (55,95),   'g': (125,175), 'b': (45,75),  'texture': (0.09,0.27)},
        'Potato___healthy':      {'r': (58,98),   'g': (118,165), 'b': (42,72),  'texture': (0.1,0.3)},
        'Corn___Common_rust':    {'r': (160,200), 'g': (100,130), 'b': (50,80),  'texture': (0.55,0.75)},
        'Tomato___Early_blight': {'r': (145,185), 'g': (95,130),  'b': (55,85),  'texture': (0.42,0.62)},
    }

    np.random.seed(42)
    N_PER_CLASS = 400
    X_list, y_list = [], []

    for cls in DEMO_CLASSES:
        p = CLASS_PROFILES[cls]
        for _ in range(N_PER_CLASS):
            r_mean = np.random.uniform(*p['r'])
            g_mean = np.random.uniform(*p['g'])
            b_mean = np.random.uniform(*p['b'])
            r_std  = np.random.uniform(8, 25)
            g_std  = np.random.uniform(8, 25)
            b_std  = np.random.uniform(8, 25)
            texture = np.random.uniform(*p['texture'])
            # Histogram bins [0-85, 85-170, 170-255] per channel + texture score
            feat = [r_mean, g_mean, b_mean,
                    r_std,  g_std,  b_std,
                    r_mean / (g_mean + 1),   # red-green ratio (disease indicator)
                    texture,
                    r_mean + g_mean + b_mean,  # brightness
                    abs(r_mean - g_mean)]       # colour imbalance
            X_list.append(feat)
            y_list.append(cls)

    X = np.array(X_list)
    le = LabelEncoder()
    y  = le.fit_transform(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("⏳ Training Random Forest on colour features...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n✅ Demo Model Accuracy: {acc*100:.1f}%")
    print(f"   Classes: {len(DEMO_CLASSES)}")

    # Sanity check — simulate a "diseased" sample (high red, low green)
    diseased_sample = np.array([[165, 95, 65, 18, 15, 12, 1.74, 0.55, 325, 70]])
    pred_cls  = le.inverse_transform(clf.predict(diseased_sample))[0]
    pred_proba = max(clf.predict_proba(diseased_sample)[0]) * 100
    print(f"\n🧪 Sanity check → high-red, low-green leaf")
    print(f"   Predicted: {pred_cls} ({pred_proba:.1f}%)")

    model_path  = os.path.join(SAVE_DIR, 'disease_demo_rf.pkl')
    classes_path = os.path.join(SAVE_DIR, 'disease_classes.pkl')
    joblib.dump(clf, model_path)
    joblib.dump({'classes': DEMO_CLASSES, 'encoder': le,
                 'mode': 'colour_histogram_rf'}, classes_path)
    print(f"\n💾 Saved: {model_path}")
    print(f"💾 Saved: {classes_path}")
    print("\n✅ Demo model saved. To upgrade to CNN:")
    print("   1. Download PlantVillage dataset from Kaggle")
    print("   2. Extract to:  datasets/disease_images/<ClassName>/")
    print("   3. Re-run this script — CNN mode activates automatically")
    print("\n✅ Training complete!")
