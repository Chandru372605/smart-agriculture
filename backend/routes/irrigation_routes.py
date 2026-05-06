"""
AgroSense — Smart Irrigation API
POST /api/irrigation/plan
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err, safe_float
from backend.config import Config
from backend.models.db_models import log_prediction

irrigation_bp = Blueprint('irrigation', __name__)

CROP_MAP  = {'Rice': 0, 'Wheat': 1, 'Maize': 2, 'Cotton': 3,
             'Sugarcane': 4, 'Tomato': 5, 'Potato': 6}
STAGE_MAP = {'Germination': 0, 'Vegetative': 1, 'Flowering': 2,
             'Fruiting': 3, 'Maturity': 4}

# Water requirements (litres/acre/day) by crop + growth stage
WATER_REQ = {
    'Rice':      [1800, 2200, 2500, 2200, 1500],
    'Wheat':     [800,  1200, 1500, 1200, 700],
    'Maize':     [900,  1400, 1800, 1500, 900],
    'Cotton':    [700,  1100, 1600, 1400, 800],
    'Sugarcane': [1200, 1800, 2200, 2000, 1600],
    'Tomato':    [600,  1000, 1400, 1600, 900],
    'Potato':    [700,  1100, 1500, 1800, 1000],
}


@irrigation_bp.route('/irrigation/plan', methods=['POST'])
def plan():
    data = request.get_json(force=True) or {}

    model = load_model(Config.IRRIG_MODEL_PATH, 'irrig_model')
    if model is None:
        return err('Irrigation model not found. Run train_irrigation.py first.')

    try:
        crop   = data.get('crop',  'Rice')
        stage  = data.get('stage', 'Vegetative')
        soil_m = safe_float(data, 'soil_moisture',     35)
        temp   = safe_float(data, 'temperature',       32)
        hum    = safe_float(data, 'humidity',          55)
        rain   = safe_float(data, 'rainfall_forecast',  5)

        c_enc = CROP_MAP.get(crop, 0)
        s_enc = STAGE_MAP.get(stage, 1)
        X     = np.array([[soil_m, temp, hum, rain, c_enc, s_enc]])
        pred  = int(model.predict(X)[0])

        decision = '💧 Irrigate Now' if pred else '⏸ Skip — Adequate Moisture'
        summary  = (
            f'Soil moisture ({soil_m}%) is below threshold with only {rain}mm forecast rain.'
            if pred else
            f'Soil moisture ({soil_m}%) and forecast rain ({rain}mm) are sufficient.'
        )

        # 7-day schedule
        stage_idx   = STAGE_MAP.get(stage, 1)
        base_vol    = WATER_REQ.get(crop, WATER_REQ['Wheat'])[stage_idx]
        today_moist = soil_m
        schedule    = []
        rain_days   = [0, rain, 0, 0, 0, rain * 0.5, 0]
        day_labels  = ['Today', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']

        for i in range(7):
            effective_rain = rain_days[i]
            need_irr = (today_moist < 45) and (effective_rain < 12)
            amount   = round(base_vol * (1 - effective_rain / 80), 0) if need_irr else 0
            schedule.append({'day': day_labels[i], 'irrigate': bool(need_irr), 'amount': int(amount)})
            today_moist = min(80, today_moist + effective_rain * 0.5 + (5 if need_irr else -3))

        metrics = [
            {'val': f'{soil_m}%',   'lbl': 'Soil Moisture'},
            {'val': f'{rain}mm',    'lbl': 'Rain Forecast'},
            {'val': f'{temp}°C',    'lbl': 'Temperature'},
            {'val': f'{base_vol}L', 'lbl': 'Daily Req/acre'},
        ]

        log_prediction('irrigation', f'{crop} | {stage} | Soil={soil_m}% Rain={rain}mm',
                       decision, inputs=data)
        return jsonify({
            'decision': decision,
            'summary':  summary,
            'metrics':  metrics,
            'schedule': schedule,
        })

    except Exception as e:
        return err(f'Irrigation prediction error: {e}', 500)
