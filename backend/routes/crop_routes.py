"""
AgroSense — Crop Recommendation API
POST /api/crop/recommend
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err, safe_float
from backend.config import Config

crop_bp = Blueprint('crop', __name__)

CROP_TIPS = {
    'rice':        'Maintain standing water 5cm deep during vegetative stage. Use split nitrogen application.',
    'wheat':       'Sow at 100–125 kg/ha seed rate. First irrigation at crown root initiation (21 days).',
    'maize':       'Ensure proper spacing of 60×20 cm. Apply zinc sulphate if soil pH > 7.5.',
    'cotton':      'Adopt square planting at 90×60 cm. Monitor for bollworm from 45 DAS.',
    'sugarcane':   'Plant in paired rows 90+30 cm. Trash mulching conserves moisture and suppresses weeds.',
    'mango':       'Prune after harvest. Apply NPK 1:0.5:1 kg/tree/year for mature trees.',
    'banana':      'Maintain 1.8×1.8 m spacing. Earthing-up at 3 and 5 months after planting.',
    'grapes':      'Train on bower/trellis. Apply 60g N + 30g P + 60g K per vine per year.',
    'apple':       'Requires chilling hours. Thin fruits to one per spur at marble stage.',
    'orange':      'Irrigate at 10-day intervals in dry season. Apply 600g N/tree/year.',
    'papaya':      'Space at 1.8×1.8 m. Remove male plants — keep one per 10 females.',
    'coconut':     'Apply 200g N + 320g P + 1200g K per palm/year. Mulch basin in summer.',
    'coffee':      'Shade-grow under silver oak. Harvest only red-ripe cherries.',
    'jute':        'Sow broadcast at 7–8 kg/ha. Thin to 7 cm spacing after 15 days.',
    'chickpea':    'No irrigation for rainfed crop. Spray 2% urea at flowering for better pod set.',
    'kidneybeans': 'Stake at 30 cm. Apply phosphorus (60 kg/ha) at sowing for better nodulation.',
    'lentil':      'Pre-sow with Rhizobium inoculant. Reduce nitrogen to 20 kg/ha.',
    'blackgram':   'Short-duration crop (60–65 days). Apply 20:40:20 NPK kg/ha.',
    'mungbean':    'Sow at 25 kg/ha. Harvest in 60–70 days when 80% pods turn black.',
    'mothbeans':   'Drought-tolerant. Suits arid regions. Apply only 10 kg N/ha at sowing.',
    'pigeonpeas':  'Intercrop with sorghum or groundnut 2:1. Avoid waterlogging.',
    'watermelon':  'Requires well-drained sandy loam. Apply 60:40:40 NPK at planting.',
    'muskmelon':   'Train on trellis in raised beds. Stop irrigation 7 days before harvest.',
    'pomegranate': 'Prune 3–4 primary shoots. Bahar treatment controls flowering time.',
}

SOIL_INDICATORS = [
    {'key': 'ph',       'label': 'Soil pH',       'good': (6.0, 7.5), 'unit': ''},
    {'key': 'N',        'label': 'Nitrogen',       'good': (40, 120),  'unit': ' kg/ha'},
    {'key': 'P',        'label': 'Phosphorus',     'good': (20, 100),  'unit': ' kg/ha'},
    {'key': 'K',        'label': 'Potassium',      'good': (20, 110),  'unit': ' kg/ha'},
    {'key': 'humidity', 'label': 'Humidity',       'good': (40, 85),   'unit': '%'},
    {'key': 'rainfall', 'label': 'Rainfall',       'good': (80, 250),  'unit': ' mm'},
]


@crop_bp.route('/crop/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True) or {}

    # Load models  (lazy-cached)
    model = load_model(Config.CROP_MODEL_PATH, 'crop_model')
    le    = load_model(Config.CROP_ENCODER_PATH, 'crop_le')
    if model is None or le is None:
        return err('Crop recommendation model not found. Run train_crop.py first.')

    try:
        N    = safe_float(data, 'N',           60)
        P    = safe_float(data, 'P',           45)
        K    = safe_float(data, 'K',           45)
        temp = safe_float(data, 'temperature', 26)
        hum  = safe_float(data, 'humidity',    70)
        ph   = safe_float(data, 'ph',          6.5)
        rain = safe_float(data, 'rainfall',    120)

        X       = np.array([[N, P, K, temp, hum, ph, rain]])
        enc_idx = model.predict(X)[0]
        probas  = model.predict_proba(X)[0]
        conf    = round(float(probas.max()) * 100, 1)

        # Top 3 alternatives
        top_idx  = probas.argsort()[::-1][:4]
        all_crops = le.classes_
        best_crop = all_crops[enc_idx]
        alts      = [all_crops[i].title() for i in top_idx if all_crops[i] != best_crop][:3]

        # Soil health indicators
        input_vals = {'N': N, 'P': P, 'K': K, 'ph': ph, 'humidity': hum, 'rainfall': rain}
        indicators = []
        for ind in SOIL_INDICATORS:
            val  = input_vals[ind['key']]
            lo, hi = ind['good']
            status = '✅' if lo <= val <= hi else ('⚠️ Low' if val < lo else '⚠️ High')
            indicators.append({'v': f"{val}{ind['unit']} {status}", 'l': ind['label']})

        crop_key = best_crop.lower()
        tips = CROP_TIPS.get(crop_key, f"Follow standard agronomic practices for {best_crop.title()} cultivation.")

        return jsonify({
            'crop':         best_crop.title(),
            'confidence':   conf,
            'alternatives': alts,
            'indicators':   indicators,
            'tips':         tips,
        })
    except Exception as e:
        return err(f'Prediction error: {e}', 500)
