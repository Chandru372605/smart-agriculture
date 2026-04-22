"""
AgroSense — Pest Risk Assessment API
POST /api/pest/assess
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err, safe_float
from backend.config import Config

pest_bp = Blueprint('pest', __name__)

PREV_MAP    = {'None': 0, 'Light': 1, 'Moderate': 2, 'Severe': 3}
DENSITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
WATER_MAP   = {'No': 0, 'Yes — within 500m': 1, 'Yes — within 100m': 2}

PEST_CATALOG = {
    'Rice':     ['Brown Planthopper', 'Stem Borer', 'Leaf Folder', 'Blast'],
    'Wheat':    ['Aphid', 'Rust (Yellow/Brown)', 'Termite', 'Powdery Mildew'],
    'Maize':    ['Fall Armyworm', 'Stem Borer', 'Aphid', 'Northern Leaf Blight'],
    'Cotton':   ['Pink Bollworm', 'Whitefly', 'Thrips', 'Mealybug'],
    'Tomato':   ['Tomato Leaf Miner', 'Whitefly', 'Helicoverpa', 'Late Blight'],
    'Potato':   ['Potato Tuber Moth', 'Aphid', 'Colorado Beetle', 'Late Blight'],
    'Sugarcane':['Pyrilla', 'Early Shoot Borer', 'Woolly Aphid', 'Red Rot'],
}

ACTIONS_BY_LEVEL = {
    'Low': [
        'Continue regular field monitoring (twice weekly)',
        'Apply neem-based pesticides as a preventive measure',
        'Maintain field sanitation — remove crop residues',
        'Set up yellow sticky traps for early detection',
    ],
    'Medium': [
        'Increase scouting frequency to every 3 days',
        'Spray systemic insecticide (e.g., Imidacloprid 0.05%) at economic threshold',
        'Release natural predators — Chrysoperla, Trichogramma cards',
        'Apply pheromone traps (2/acre) to monitor adult population',
        'Record pest counts and track population trends',
    ],
    'High': [
        '⚠️ Immediate chemical intervention required — assess economic threshold',
        'Spray broad-spectrum insecticide (e.g., Chlorpyrifos 20 EC @ 2.5 ml/L)',
        'Consult agronomist or KVK for crop-specific emergency protocol',
        'Notify neighbouring farmers for area-wide coordinated management',
        'Document severity and report to district agriculture office',
        'Consider crop-loss insurance claim if >25% damage',
    ],
}


@pest_bp.route('/pest/assess', methods=['POST'])
def assess():
    data = request.get_json(force=True) or {}

    model    = load_model(Config.PEST_MODEL_PATH, 'pest_model')
    encoders = load_model(Config.PEST_ENC_PATH,   'pest_enc')
    if model is None or encoders is None:
        return err('Pest model not found. Run train_pest.py first.')

    try:
        crop      = data.get('crop',   'Rice')
        season    = data.get('season', 'Kharif (Jun–Oct)')
        temp      = safe_float(data, 'temperature', 30)
        hum       = safe_float(data, 'humidity',    75)
        prev_raw  = data.get('prev_occurrence', 'None')
        density_r = data.get('crop_density',    'Medium')
        water_r   = data.get('near_water',      'No')

        prev    = PREV_MAP.get(str(prev_raw), int(prev_raw) if str(prev_raw).isdigit() else 0)
        density = DENSITY_MAP.get(str(density_r), 1)
        water   = WATER_MAP.get(str(water_r), 0)

        # Encode crop / season
        def enc_s(le, val):
            classes = list(le.classes_)
            return classes.index(val) if val in classes else 0

        season_simple = season.split(' (')[0]  # strip season descriptor
        c_enc = enc_s(encoders['crop'],   crop)
        s_enc = enc_s(encoders['season'], season_simple)

        X    = np.array([[temp, hum, prev, density, water, s_enc, c_enc]])
        risk = float(model.predict(X)[0])
        risk = max(0.0, min(100.0, risk))

        level = 'High' if risk > 65 else ('Medium' if risk > 35 else 'Low')

        threats = PEST_CATALOG.get(crop, ['General pest pressure detected'])
        # Surface only top threats proportional to risk
        n_show  = 1 if level == 'Low' else (3 if level == 'Medium' else len(threats))
        threats = threats[:n_show]

        return jsonify({
            'score':   round(risk, 1),
            'level':   level,
            'threats': threats,
            'actions': ACTIONS_BY_LEVEL[level],
        })

    except Exception as e:
        return err(f'Pest risk error: {e}', 500)
