"""
AgroSense — Yield Prediction API
POST /api/yield/predict
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err, safe_float, fmt_inr
from backend.config import Config
from backend.models.db_models import log_prediction

yield_bp = Blueprint('yield', __name__)

PRICE_PER_TONNE = {
    'Rice': 18000, 'Wheat': 21000, 'Maize': 16000,
    'Cotton': 55000, 'Sugarcane': 3500, 'Soybean': 38000, 'Groundnut': 46000,
}
IRR_MAP  = {'Rainfed': 0, 'Partially Irrigated': 1, 'Fully Irrigated': 2}
PEST_MAP = {'Organic': 0, 'Moderate': 1, 'High': 2}
REGIONAL_AVG = {
    'Rice': 3.8, 'Wheat': 4.2, 'Maize': 5.1, 'Cotton': 2.1,
    'Sugarcane': 68.0, 'Soybean': 2.8, 'Groundnut': 2.2,
}

YIELD_TIPS = {
    'Rice':     ['Use SRI method — transplant single seedlings at wider spacing',
                 'Apply zinc sulphate (25 kg/ha) if deficiency observed',
                 'Drain field 10 days before harvest for easier mechanisation'],
    'Wheat':    ['Seed treatment with carbendazim prevents loose smut',
                 'Apply second dose of nitrogen at first node stage',
                 'Timely harvesting prevents shattering losses'],
    'Maize':    ['Intercrop with soybean for 20% higher land use efficiency',
                 'Apply boron (500g/ha) at tasseling stage',
                 'Use drip irrigation to save 40% water vs flood'],
    'Cotton':   ['Plant on raised beds in vertisols to prevent waterlogging',
                 'Monitor pink bollworm pheromone traps from 45 DAS',
                 'Apply plant growth regulator at 75 DAS to reduce vegetative growth'],
    'Sugarcane':['Ratoon crop saves replanting cost — maintain for 2–3 ratoons',
                 'Trash mulching conserves moisture and suppresses weeds',
                 'Harvest at optimal sucrose content — test before cutting'],
    'Soybean':  ['Inoculate seeds with Bradyrhizobium japonicum for free nitrogen',
                 'Apply molybdenum (500g/ha) for better nodulation',
                 'Harvest when 95% pods turn brown to minimise shattering'],
    'Groundnut':['Gypsum (500 kg/ha) applied at flowering prevents peg rot',
                 'Spray chlorothalonil for early leaf spot control',
                 'Harvest before first rains to prevent aflatoxin contamination'],
}


@yield_bp.route('/yield/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) or {}

    model    = load_model(Config.YIELD_MODEL_PATH, 'yield_model')
    encoders = load_model(Config.YIELD_ENC_PATH,   'yield_enc')
    if model is None or encoders is None:
        return err('Yield model not found. Run train_yield.py first.')

    try:
        crop   = data.get('crop',   'Rice')
        season = data.get('season', 'Kharif')
        state  = data.get('state',  'Punjab')
        area   = safe_float(data, 'area',       5)
        rain   = safe_float(data, 'rainfall',   900)
        fert   = safe_float(data, 'fertiliser', 120)
        irr    = safe_float(data, 'irrigation', 1)
        pest   = safe_float(data, 'pesticide',  1)

        def enc_safe(le, val):
            classes = list(le.classes_)
            return classes.index(val) if val in classes else 0

        c_enc  = enc_safe(encoders['crop'],   crop)
        s_enc  = enc_safe(encoders['season'], season)
        st_enc = enc_safe(encoders['state'],  state)
        irr_v  = IRR_MAP.get(str(irr),  int(irr))  if isinstance(irr,  str) else int(irr)
        pest_v = PEST_MAP.get(str(pest), 1)         if isinstance(pest, str) else int(pest)

        X      = np.array([[c_enc, s_enc, st_enc, area, rain, fert, pest_v, irr_v]])
        y_pred = float(model.predict(X)[0])
        y_pred = max(0.1, round(y_pred, 2))

        regional_avg = REGIONAL_AVG.get(crop, 3.5)
        pct_diff     = round((y_pred - regional_avg) / regional_avg * 100, 1)
        total_prod   = round(y_pred * area, 2)
        price        = PRICE_PER_TONNE.get(crop, 18000)
        revenue      = round(total_prod * price)
        conf         = min(98, max(55, 80 + (pct_diff * 0.2)))

        summary = (f'Expected {total_prod:.1f}t total — {pct_diff:+.1f}% '
                   f'vs regional average of {regional_avg}t/ha')

        comparison = [
            {'lbl': 'Your Yield',    'val': str(y_pred),                        'color': 'var(--green-soft)'},
            {'lbl': 'Region Avg',    'val': str(regional_avg),                  'color': 'var(--amber)'},
            {'lbl': 'National Best', 'val': str(round(regional_avg * 1.35, 1)), 'color': 'var(--green-mid)'},
        ]
        metrics = [
            {'val': f'{y_pred} t/ha',      'lbl': 'Predicted Yield'},
            {'val': f'{total_prod} t',      'lbl': 'Total Production'},
            {'val': fmt_inr(revenue),       'lbl': 'Est. Revenue'},
            {'val': f'{pct_diff:+.1f}%',   'lbl': 'vs Regional Avg'},
        ]

        log_prediction('yield', f"{crop} | {area}ha | {data.get('irrigation', 'N/A')}",
                       f'{y_pred} t/ha', inputs=data, confidence=round(conf, 1))
        return jsonify({
            'yield_per_ha': y_pred,
            'summary':      summary,
            'confidence':   round(conf, 1),
            'metrics':      metrics,
            'comparison':   comparison,
            'tips':         YIELD_TIPS.get(crop, ['Follow best agronomic practices for your region.']),
        })

    except Exception as e:
        return err(f'Yield prediction error: {e}', 500)
