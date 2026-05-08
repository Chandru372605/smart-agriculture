"""
AgroSense — Market Price Prediction API
POST /api/market/forecast
GET  /api/mandi/price?crop=Rice&state=Karnataka  → live mandi price
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, load_keras_model, err, safe_float, fmt_inr
from backend.config import Config
from backend.models.db_models import log_prediction
from backend.services.mandi_service import get_mandi_price, get_available_commodities

market_bp = Blueprint('market', __name__)

MARKET_PREMIUM = {
    'Delhi (Azadpur)':    1.08,
    'Mumbai (Vashi)':     1.12,
    'Bengaluru (APMC)':   1.05,
    'Chennai (Koyambedu)':1.03,
    'Hyderabad':          1.04,
    'Pune':               1.06,
    'Kolkata':            1.02,
}

INSIGHTS_DB = {
    'Rice':     'Rice prices typically peak during lean season (Apr–Jun). Kharif harvest (Oct–Nov) brings seasonal dip.',
    'Wheat':    'Wheat prices are MSP-supported. Market prices may soften after Rabi procurement season.',
    'Maize':    'Maize prices correlate with poultry feed demand. Prices rise in winter as feed demand increases.',
    'Onion':    'Onion is highly volatile — monitor rainfall in Maharashtra and Karnataka. Export bans can cause sharp dips.',
    'Tomato':   'Tomato prices fluctuate weekly. Summer prices often 3–4x higher than winter glut prices.',
    'Potato':   'Cold storage availability drives inter-seasonal price variation. Best selling window: May–Aug.',
    'Soybean':  'Soybean follows global oilseed markets. MSP acts as floor price. Watch US crop reports.',
    'Cotton':   'Cotton prices influenced by arrivals and spinning mill demand. Peak prices in Jan–Mar.',
    'Groundnut':'High-value crop — prices stable. Watch for diversion to oil crushing vs direct consumption.',
}


# ── Live Mandi Price endpoint ────────────────────────────────────────────────
@market_bp.route('/mandi/price', methods=['GET'])
def mandi_price():
    """
    GET /api/mandi/price?crop=Rice+%28Common%29&state=Karnataka
    Returns today's live mandi modal price from data.gov.in.
    Falls back to baseline estimates when API is unavailable.
    """
    crop  = request.args.get('crop', 'Rice (Common)').strip()
    state = request.args.get('state', '').strip()
    data  = get_mandi_price(crop, state)
    return jsonify(data)


@market_bp.route('/mandi/commodities', methods=['GET'])
def mandi_commodities():
    """List of crops supported by the mandi price service."""
    return jsonify({'commodities': get_available_commodities()})


# ── Forecast endpoint ────────────────────────────────────────────────────────
@market_bp.route('/market/forecast', methods=['POST'])
def forecast():
    data = request.get_json(force=True) or {}

    scaler = load_model(Config.MARKET_SCALER_PATH, 'mkt_scaler')
    meta   = load_model(Config.MARKET_META_PATH,   'mkt_meta')
    lstm   = load_keras_model(Config.MARKET_MODEL_PATH, 'mkt_lstm')

    try:
        crop_raw = data.get('crop',   'Rice (Common)')
        market   = data.get('market', 'Delhi (Azadpur)')
        n_days   = safe_float(data, 'forecast_days', 14)
        season   = data.get('season', 'Kharif (harvest)')
        crop     = crop_raw.split(' (')[0]   # strip parenthetical
        premium  = MARKET_PREMIUM.get(market, 1.0)
        n_days   = int(max(7, min(30, n_days)))

        # ── Auto-fill current price from live mandi data ──────────────────
        mandi    = get_mandi_price(crop_raw)
        curr_p   = safe_float(data, 'current_price', 0)
        if curr_p <= 0:
            curr_p = mandi['modal_price']
        price_live = mandi['live']

        if lstm is not None and scaler is not None and meta is not None:
            # ── Real LSTM inference ──────────────────────────────────────
            crop_names = meta.get('crops', [])
            seq_len    = meta.get('seq_len', 30)
            n_crops    = len(crop_names)

            crop_idx_map = {c.split(' (')[0].lower(): i for i, c in enumerate(crop_names)}
            c_i    = crop_idx_map.get(crop.lower(), 0)
            c_code = c_i / max(n_crops - 1, 1)

            rng  = np.random.default_rng(hash(crop + market) % (2**31))
            hist = curr_p + rng.normal(0, curr_p * 0.02, seq_len).cumsum() * 0.3
            hist = np.clip(hist, curr_p * 0.7, curr_p * 1.3)
            hist_s = scaler.transform(hist.reshape(-1, 1)).flatten()

            prices_pred = [float(curr_p)]
            window = hist_s.copy()
            for _ in range(n_days):
                seq  = np.stack([window[-seq_len:], np.full(seq_len, c_code)], axis=1)
                seq  = seq[np.newaxis, :, :].astype(np.float32)
                nxt  = float(scaler.inverse_transform(
                    lstm.predict(seq, verbose=0).reshape(-1, 1)
                )[0][0])
                nxt  = nxt * premium
                prices_pred.append(round(nxt, 0))
                window = np.append(window, scaler.transform([[nxt]])[0][0])

        else:
            # ── Fallback: simple trend simulation ───────────────────────
            rng    = np.random.default_rng(hash(crop + market) % (2**31))
            trend  = rng.uniform(-0.3, 0.8)
            vol    = curr_p * 0.012
            prices_pred = [curr_p]
            p = curr_p
            for i in range(n_days):
                seasonal = 1 + 0.005 * np.sin(2 * np.pi * i / 14)
                p = round((p + trend + rng.normal(0, vol)) * seasonal * premium, 0)
                p = max(curr_p * 0.6, min(curr_p * 1.6, p))
                prices_pred.append(p)

        final_p  = prices_pred[-1]
        change   = final_p - curr_p
        pct_chg  = round(change / curr_p * 100, 1)
        min_p    = round(min(prices_pred))
        max_p    = round(max(prices_pred))
        best_day = int(np.argmax(prices_pred))

        metrics = [
            {'val': fmt_inr(int(curr_p)),  'lbl': 'Current Price'},
            {'val': fmt_inr(int(final_p)), 'lbl': f'Day {n_days} Forecast',
             'color': 'var(--green-soft)' if change >= 0 else 'var(--danger)'},
            {'val': f'{pct_chg:+.1f}%',   'lbl': 'Expected Change',
             'color': 'var(--green-soft)' if change >= 0 else 'var(--danger)'},
            {'val': fmt_inr(max_p),        'lbl': f'Peak (Day {best_day})'},
        ]

        if change > 0:
            sell_tags = [
                {'text': f'📅 Best window: Day {best_day}',    'cls': 'green'},
                {'text': f'💰 Target price: {fmt_inr(max_p)}', 'cls': 'green'},
                {'text': '⏳ Hold if possible — prices rising', 'cls': ''},
            ]
        else:
            sell_tags = [
                {'text': '⚡ Sell soon — prices softening', 'cls': 'red'},
                {'text': f'🎯 Sell at: {fmt_inr(int(curr_p))} or above', 'cls': ''},
            ]

        insight = INSIGHTS_DB.get(crop, f'{crop} prices depend on seasonal demand and arrivals in key markets.')

        log_prediction('market', f'{crop} | {n_days}d forecast',
                       f'Forecast {fmt_inr(int(prices_pred[-1]))}', inputs=data)
        return jsonify({
            'prices':      [int(p) for p in prices_pred],
            'metrics':     metrics,
            'insights':    insight,
            'sell_tags':   sell_tags,
            'mandi_price': mandi['modal_price'],
            'mandi_live':  price_live,
            'mandi_market':mandi['market'],
            'mandi_date':  mandi['arrival_date'],
            'mandi_source':mandi['source'],
        })

    except Exception as e:
        return err(f'Market forecast error: {e}', 500)
