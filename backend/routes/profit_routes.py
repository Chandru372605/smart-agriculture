"""
AgroSense — Farm Profit Estimator API
POST /api/profit/calculate
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err, safe_float, fmt_inr
from backend.config import Config

profit_bp = Blueprint('profit', __name__)

COST_COLORS = ['#4c9c6e','#2475a8','#e8a030','#7a5c3e','#c0392b','#6c5ce7']


@profit_bp.route('/profit/calculate', methods=['POST'])
def calculate():
    data = request.get_json(force=True) or {}

    model = load_model(Config.PROFIT_MODEL_PATH, 'profit_model')
    if model is None:
        return err('Profit model not found. Run train_profit.py first.')

    try:
        crop     = data.get('crop', 'Rice')
        area     = safe_float(data, 'area',            5)
        yph      = safe_float(data, 'yield_per_ha',    4)
        price    = safe_float(data, 'selling_price',18000)
        seed     = safe_float(data, 'seed_cost',     3500)
        fert     = safe_float(data, 'fertiliser_cost',5000)
        labour   = safe_float(data, 'labour_cost',  8000)
        irr      = safe_float(data, 'irrigation_cost',2500)
        pest     = safe_float(data, 'pesticide_cost',2000)
        misc     = safe_float(data, 'misc_cost',    1000)

        # ML-adjusted yield
        X      = np.array([[area, fert, irr, pest]])
        adj_y  = float(model.predict(X)[0])        # ML predicted yield/ha
        # Blend user's expected yield with ML adjustment (70/30)
        final_y = round(0.7 * yph + 0.3 * adj_y, 2)

        total_prod   = round(final_y * area, 2)
        revenue      = round(total_prod * price)

        cost_per_ha  = seed + fert + labour + irr + pest + misc
        total_cost   = round(cost_per_ha * area)
        net_profit   = revenue - total_cost
        roi          = round((net_profit / total_cost) * 100, 1) if total_cost > 0 else 0
        bep_yield    = round(total_cost / (price * area), 2) if price * area > 0 else 0

        profitable = net_profit > 0
        kpis = [
            {'val': fmt_inr(revenue),    'lbl': 'Gross Revenue',  'highlight': False, 'trend': None},
            {'val': fmt_inr(total_cost), 'lbl': 'Total Cost',     'highlight': False, 'trend': None},
            {'val': fmt_inr(net_profit), 'lbl': 'Net Profit',     'highlight': True,
             'trend': 'up' if profitable else 'down'},
            {'val': f'{roi:.1f}%',       'lbl': 'ROI',            'highlight': False, 'trend': None},
            {'val': f'{bep_yield} t/ha','lbl': 'Break-even Yield','highlight': False, 'trend': None},
            {'val': f'{final_y} t/ha',  'lbl': 'ML-Adj. Yield',  'highlight': False, 'trend': None},
        ]

        cost_items = [
            ('Seed',        seed   * area),
            ('Fertiliser',  fert   * area),
            ('Labour',      labour * area),
            ('Irrigation',  irr    * area),
            ('Pesticide',   pest   * area),
            ('Misc.',       misc   * area),
        ]
        costs_out = [
            {'lbl': c[0], 'val': c[1], 'display': fmt_inr(c[1]),
             'color': COST_COLORS[i % len(COST_COLORS)]}
            for i, c in enumerate(cost_items) if c[1] > 0
        ]

        if profitable:
            analysis = (f'Your {crop} farm of {area}ha is projected to earn {fmt_inr(net_profit)} '
                        f'net profit with an ROI of {roi}%. The ML yield adjustment based on your '
                        f'input costs estimates {final_y}t/ha. You need at least {bep_yield}t/ha to break even.')
        else:
            analysis = (f'At current costs and prices, the farm shows a loss of {fmt_inr(abs(net_profit))}. '
                        f'Consider increasing yield (target: {bep_yield}t/ha to break even), '
                        f'reducing input costs, or exploring better market channels.')

        return jsonify({
            'kpis':     kpis,
            'costs':    costs_out,
            'analysis': analysis,
        })

    except Exception as e:
        return err(f'Profit calculation error: {e}', 500)
