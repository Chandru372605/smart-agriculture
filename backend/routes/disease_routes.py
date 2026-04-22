"""
AgroSense — Disease Detection API
POST /api/disease/predict          (multipart image upload)
POST /api/disease/predict-sample   (JSON — sample button path)
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err
from backend.config import Config

disease_bp = Blueprint('disease', __name__)

TREATMENT_DB = {
    'Tomato___Late_blight':   {
        'treatment':  'Apply Mancozeb (0.25%) or Metalaxyl + Mancozeb (Ridomil Gold) at 7-day intervals. Remove and destroy infected plant parts immediately.',
        'parts':      ['Leaves', 'Stems', 'Fruits'],
        'preventive': ['Use certified disease-free seeds', 'Avoid overhead irrigation', 'Maintain proper plant spacing for air circulation', 'Apply preventive copper-based fungicide at first sign of rain'],
    },
    'Apple___Apple_scab':     {
        'treatment':  'Spray Captan (0.3%) or Mancozeb (0.25%) during pink bud to petal fall stage. Repeat at 10-day intervals.',
        'parts':      ['Leaves', 'Fruits'],
        'preventive': ['Plant resistant varieties', 'Rake and destroy fallen leaves', 'Apply fungicide before rains', 'Prune for good canopy airflow'],
    },
    'Potato___Late_blight':   {
        'treatment':  'Apply Cymoxanil + Mancozeb or Chlorothalonil at first sign. Haulm destruction before harvest prevents tuber blight.',
        'parts':      ['Leaves', 'Stems', 'Tubers'],
        'preventive': ['Use certified seed potatoes', 'Plant in well-drained soil', 'Avoid excessive nitrogen', 'Monitor weather — blight favours cool moist conditions'],
    },
    'Corn___healthy':         {'treatment': 'Plant appears healthy. Maintain current agronomic practices.', 'parts': [], 'preventive': ['Regular field scouting', 'Balanced fertilisation', 'Timely irrigation']},
    'Rice___healthy':         {'treatment': 'Plant appears healthy. No intervention needed.', 'parts': [], 'preventive': ['Monitor for brown planthopper', 'Maintain proper water depth', 'Apply balanced NPK']},
    'Tomato___healthy':       {'treatment': 'Plant is healthy. Continue current care practices.', 'parts': [], 'preventive': ['Regular inspection', 'Stake plants for support', 'Consistent watering schedule']},
    'Apple___healthy':        {'treatment': 'Tree is healthy. Maintain current orchard management.', 'parts': [], 'preventive': ['Annual pruning', 'Soil pH monitoring', 'Integrated pest management']},
    'Potato___healthy':       {'treatment': 'Plant appears healthy.', 'parts': [], 'preventive': ['Crop rotation (avoid planting after tomato/pepper)', 'Hill up soil around stems', 'Reduce irrigation before harvest']},
    'Corn___Common_rust':     {
        'treatment':  'Apply Propiconazole (0.1%) or Tebuconazole at disease onset. Most field corn is tolerant — economic threshold determines spray timing.',
        'parts':      ['Leaves'],
        'preventive': ['Plant resistant hybrids', 'Early planting avoids peak rust season', 'Scout from V5 stage'],
    },
    'Tomato___Early_blight':  {
        'treatment':  'Apply Chlorothalonil (0.2%) or copper oxychloride. Remove lower infected leaves. Ensure good drainage.',
        'parts':      ['Lower leaves', 'Stems', 'Fruits'],
        'preventive': ['Mulch to prevent soil splash', 'Avoid wetting foliage', 'Remove plant debris after harvest', 'Crop rotation of 2–3 years'],
    },
}

HEALTHY_CLASSES = {'Corn___healthy', 'Rice___healthy', 'Tomato___healthy', 'Apple___healthy', 'Potato___healthy'}


def _get_response(class_name: str, confidence: float) -> dict:
    db    = TREATMENT_DB.get(class_name, {})
    is_h  = class_name in HEALTHY_CLASSES
    label = class_name.replace('___', ' — ').replace('_', ' ')
    return {
        'name':       label,
        'type':       'healthy' if is_h else 'diseased',
        'confidence': round(confidence, 1),
        'treatment':  db.get('treatment', 'Consult a local agricultural extension officer for diagnosis.'),
        'parts':      db.get('parts', []),
        'preventive': db.get('preventive', []),
    }


@disease_bp.route('/disease/predict', methods=['POST'])
def predict_image():
    """Real image inference — colour-feature RF (demo) or CNN if available."""
    if 'image' not in request.files:
        return err('No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return err('Empty filename')

    try:
        import io
        from PIL import Image

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((128, 128))

        # ── Try CNN first ──────────────────────────────────────────────
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        from backend.utils.helpers import load_keras_model
        cnn   = load_keras_model(Config.DISEASE_CNN_PATH, 'disease_cnn')
        if cnn is not None:
            from backend.utils.helpers import load_model as lm
            meta  = lm(Config.DISEASE_META_PATH, 'disease_meta')
            arr   = np.array(img, dtype=np.float32) / 255.0
            arr   = arr[np.newaxis, ...]
            preds = cnn.predict(arr, verbose=0)[0]
            idx   = int(preds.argmax())
            conf  = float(preds[idx]) * 100
            cls   = (meta[idx] if isinstance(meta, list) else meta.get('classes', [])[idx])
            return jsonify(_get_response(cls, conf))

        # ── Fall back to colour-histogram RF ──────────────────────────
        meta_obj = load_model(Config.DISEASE_META_PATH, 'disease_meta')
        rf_model = load_model(Config.DISEASE_RF_PATH, 'disease_rf')
        if rf_model is None:
            return err('Disease model not found. Run train_disease.py first.')

        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        feat = np.array([[
            r.mean(), g.mean(), b.mean(),
            r.std(),  g.std(),  b.std(),
            r.mean() / (g.mean() + 1),
            (arr.std() / 128),
            arr.mean(),
            abs(r.mean() - g.mean()),
        ]])
        enc   = meta_obj['encoder']
        proba = rf_model.predict_proba(feat)[0]
        idx   = int(proba.argmax())
        conf  = float(proba[idx]) * 100
        cls   = enc.inverse_transform([idx])[0]
        return jsonify(_get_response(cls, conf))

    except Exception as e:
        return err(f'Image analysis error: {e}', 500)


@disease_bp.route('/disease/predict-sample', methods=['POST'])
def predict_sample():
    """Handle pre-defined sample button clicks — returns curated response."""
    data = request.get_json(force=True) or {}
    name = data.get('name', 'Unknown')
    conf = float(data.get('conf', 85))
    dtype = data.get('type', 'diseased')

    # Map sample name to DB key
    SAMPLE_MAP = {
        'Tomato Leaf Blight': 'Tomato___Late_blight',
        'Healthy Corn':       'Corn___healthy',
        'Apple Scab':         'Apple___Apple_scab',
        'Potato Late Blight': 'Potato___Late_blight',
        'Healthy Rice':       'Rice___healthy',
    }
    cls = SAMPLE_MAP.get(name, 'Tomato___healthy')
    return jsonify(_get_response(cls, conf))
