"""
AgroSense — Crop Rotation API
POST /api/rotation/recommend
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err
from backend.config import Config

rotation_bp = Blueprint('rotation', __name__)

CROP_ICONS = {
    'Rice':'🌾','Wheat':'🌾','Maize':'🌽','Cotton':'🌿','Sugarcane':'🎋',
    'Soybean':'🌱','Groundnut':'🥜','Chickpea':'🫘','Mustard':'🌼',
    'Onion':'🧅','Potato':'🥔','Sorghum':'🌾','Lentil':'🫘',
}

SEASON_SEQ = ['Kharif (Jun–Oct)', 'Rabi (Nov–Mar)', 'Zaid (Mar–Jun)']

ROTATION_NOTES_DB = {
    'Chickpea': 'Legume — fixes atmospheric nitrogen (50–80 kg N/ha), greatly reducing fertiliser cost for the next season.',
    'Wheat':    'Heavy feeder — apply split nitrogen. Residue incorporation improves organic matter.',
    'Maize':    'Deep-rooted — breaks hardpan. Plant at 60×20 cm, apply 150 kg N/ha.',
    'Groundnut':'Nitrogen fixer and soil brightener. Ideal post-cotton to break root rot cycle.',
    'Soybean':  'Fix 100+ kg N/ha. Shallow roots — leave more subsoil moisture for following crop.',
    'Mustard':  'Allelopathic to weeds. Incorporated biomass adds organic matter.',
    'Rice':     'Flooded conditions suppress soil-borne pathogens. Ensure good drainage before next crop.',
    'Onion':    'Biofumigant properties suppress nematodes. High value crop improves farm income.',
    'Potato':   'Intensive feeder. Ensure 3-year gap before returning to same field.',
    'Sorghum':  'Drought-tolerant. Excellent biomass for mulch and soil organic matter.',
    'Cotton':   'Deep-rooted. Requires high NPK — plan fertiliser accordingly.',
    'Lentil':   'Short-duration legume (90 days). Fixes nitrogen and leaves soil friable.',
    'Sugarcane':'Long-duration crop — ratoon for 2–3 years to amortise establishment cost.',
}

BENEFIT_TAGS = {
    'Chickpea': ['🌿 Nitrogen Fixation', '💰 Reduced Fertiliser Cost'],
    'Soybean':  ['🌿 Nitrogen Fixation', '🌱 Soil Improvement'],
    'Groundnut':['🌿 Nitrogen Fixation', '🐛 Pest Cycle Break'],
    'Mustard':  ['🌿 Weed Suppression',  '🌱 Organic Matter'],
    'default':  ['🔄 Pest Cycle Break', '🌱 Soil Health', '💧 Moisture Management'],
}


@rotation_bp.route('/rotation/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True) or {}

    model    = load_model(Config.ROT_MODEL_PATH, 'rot_model')
    encoders = load_model(Config.ROT_ENC_PATH,   'rot_enc')
    if model is None or encoders is None:
        return err('Rotation model not found. Run train_rotation.py first.')

    try:
        curr_crop    = data.get('current_crop', 'Rice')
        soil_type    = data.get('soil_type',    'Loamy')
        region       = data.get('region',       'Indo-Gangetic Plain')
        n_level      = data.get('n_level',      'Medium')
        pest_history = data.get('pest_history', 'None')

        def enc_safe(enc, col, val):
            classes = list(enc.classes_)
            return classes.index(val) if val in classes else 0

        c_enc  = enc_safe(encoders['current_crop'], 'current_crop', curr_crop)
        s_enc  = enc_safe(encoders['soil_type'],    'soil_type',    soil_type)
        r_enc  = enc_safe(encoders['region'],       'region',       region)
        n_enc  = enc_safe(encoders['n_level'],      'n_level',      n_level)
        p_enc  = enc_safe(encoders['pest_history'], 'pest_history', pest_history)

        X          = np.array([[c_enc, s_enc, r_enc, n_enc, p_enc]])
        next_enc   = model.predict(X)[0]
        probas     = model.predict_proba(X)[0]
        top_idx    = probas.argsort()[::-1][:3]
        le_target  = encoders['next_crop']
        crops_seq  = [le_target.classes_[i] for i in top_idx]

        # Build a 3-season plan: predicted → alt → original (closing rotation)
        season1 = crops_seq[0]
        season2 = crops_seq[1] if len(crops_seq) > 1 else 'Wheat'
        season3 = curr_crop if curr_crop not in (season1, season2) else (
            crops_seq[2] if len(crops_seq) > 2 else 'Chickpea')

        plan = [
            {'crop': season1, 'icon': CROP_ICONS.get(season1, '🌱'), 'season': SEASON_SEQ[0]},
            {'crop': season2, 'icon': CROP_ICONS.get(season2, '🌱'), 'season': SEASON_SEQ[1]},
            {'crop': season3, 'icon': CROP_ICONS.get(season3, '🌱'), 'season': SEASON_SEQ[2]},
        ]

        notes = [
            {'season': SEASON_SEQ[i], 'crop': p['crop'],
             'note': ROTATION_NOTES_DB.get(p['crop'], f'Follow standard agronomic practices for {p["crop"]}.')}
            for i, p in enumerate(plan)
        ]

        # Collect benefit tags from the sequence
        benefits_set = set()
        for p in [season1, season2]:
            for tag in BENEFIT_TAGS.get(p, BENEFIT_TAGS['default']):
                benefits_set.add(tag)
        benefits_set.update(BENEFIT_TAGS['default'])
        benefits = list(benefits_set)[:6]

        return jsonify({
            'plan':     plan,
            'benefits': benefits,
            'notes':    notes,
        })

    except Exception as e:
        return err(f'Rotation prediction error: {e}', 500)
