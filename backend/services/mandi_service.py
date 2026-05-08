"""
AgroSense — Mandi Price Service
=================================
Fetches daily commodity prices from data.gov.in's APMC mandi dataset.
Dataset: 9ef84268-d588-465a-a308-a864a43d0070
  (Current Daily Price of Various Commodities from Various Markets)

Falls back gracefully when API is unavailable (slow gov servers are common).
Results cached in-process for 3 hours to avoid hammering the API.
"""

import time
import threading
import json
import os

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID = '9ef84268-d588-465a-a308-a864a43d0070'
BASE_URL    = f'https://api.data.gov.in/resource/{DATASET_ID}'
CACHE_TTL   = 10800   # 3 hours (mandi prices update once daily)
TIMEOUT     = 12      # seconds

# ── In-process cache ──────────────────────────────────────────────────────────
_cache: dict = {}
_lock  = threading.Lock()

# ── Commodity name mapping (AgroSense → data.gov.in names) ───────────────────
CROP_MAP = {
    'Rice (Common)':    'Rice',
    'Rice (Basmati)':   'Rice',
    'Wheat':            'Wheat',
    'Maize':            'Maize',
    'Cotton':           'Cotton',
    'Soybean':          'Soybean',
    'Sugarcane':        'Sugarcane',
    'Groundnut':        'Groundnut',
    'Bajra':            'Bajra',
    'Jowar':            'Jowar',
    # add more as needed
}

# Fallback base prices (₹/quintal) when API is unavailable
FALLBACK_PRICES = {
    'Rice':       2150,
    'Wheat':      2275,
    'Maize':      1850,
    'Cotton':     6500,
    'Soybean':    4200,
    'Sugarcane':  315,
    'Groundnut':  5600,
    'Bajra':      2350,
    'Jowar':      2800,
}


def _get_api_key() -> str:
    return os.getenv('DATAGOV_API_KEY', '')


def _fetch_mandi_price(commodity: str, state: str = '') -> dict | None:
    """
    Fetch modal price for a commodity from data.gov.in.
    Returns dict with price info or None on failure.
    """
    api_key = _get_api_key()
    if not api_key or not _HAS_REQUESTS:
        return None

    params = {
        'api-key': api_key,
        'format':  'json',
        'limit':   10,
        'filters[commodity]': commodity,
    }
    if state:
        params['filters[state]'] = state

    try:
        r = _requests.get(BASE_URL, params=params, timeout=TIMEOUT, verify=False)
        if r.status_code != 200:
            return None
        d = r.json()
        records = d.get('records', [])
        if not records:
            return None

        # Pick the most recent record with a valid modal price
        best = None
        for rec in records:
            try:
                price = float(rec.get('modal_price', 0))
                if price > 0:
                    best = rec
                    break
            except (ValueError, TypeError):
                continue

        if not best:
            return None

        return {
            'commodity':    best.get('commodity', commodity),
            'market':       best.get('market', '—'),
            'state':        best.get('state', '—'),
            'district':     best.get('district', '—'),
            'min_price':    float(best.get('min_price', 0)),
            'max_price':    float(best.get('max_price', 0)),
            'modal_price':  float(best.get('modal_price', 0)),
            'arrival_date': best.get('arrival_date', '—'),
            'unit':         'per quintal (100 kg)',
            'source':       'data.gov.in (APMC)',
            'live':         True,
        }

    except Exception:
        return None


def get_mandi_price(agrosense_crop: str, state: str = '') -> dict:
    """
    Public API — returns current mandi price for a crop.
    Falls back to baseline prices when data.gov.in is unavailable.

    Args:
        agrosense_crop: crop name as used in AgroSense UI
        state: optional state filter (e.g. 'Karnataka')

    Returns:
        {
          modal_price: float (₹/quintal),
          live: bool,
          source: str,
          ...
        }
    """
    commodity = CROP_MAP.get(agrosense_crop, agrosense_crop.split('(')[0].strip())
    cache_key  = f"{commodity}_{state}"

    # Check cache
    with _lock:
        entry = _cache.get(cache_key)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['data']

    # Fetch live
    result = _fetch_mandi_price(commodity, state)

    # Fallback if fetch failed
    if result is None:
        fallback_price = FALLBACK_PRICES.get(commodity, 2000)
        result = {
            'commodity':   commodity,
            'market':      'Baseline estimate',
            'state':       state or 'India (avg)',
            'district':    '—',
            'min_price':   round(fallback_price * 0.9),
            'max_price':   round(fallback_price * 1.1),
            'modal_price': fallback_price,
            'arrival_date':'—',
            'unit':        'per quintal (100 kg)',
            'source':      'AgroSense baseline (data.gov.in unavailable)',
            'live':        False,
        }

    # Cache the result
    with _lock:
        _cache[cache_key] = {'data': result, 'ts': time.time()}

    return result


def get_mandi_price_per_kg(agrosense_crop: str, state: str = '') -> float:
    """Returns modal price in ₹/kg (quintal / 100)."""
    d = get_mandi_price(agrosense_crop, state)
    return round(d['modal_price'] / 100, 2)


def get_available_commodities() -> list[str]:
    """Return list of supported crop names."""
    return list(CROP_MAP.keys())
