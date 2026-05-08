"""
AgroSense — Open-Meteo Service
================================
Free weather + soil data from open-meteo.com — no API key required.
Provides better data than OpenWeatherMap for agriculture:
  - Soil moisture (0-1cm, 1-3cm, 3-9cm depth)
  - Evapotranspiration (ET₀)
  - Precipitation sum (hourly + daily)
  - UV index, wind gusts
"""
import os, threading, time
try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

BASE_URL   = 'https://api.open-meteo.com/v1/forecast'
CACHE_TTL  = 1800   # 30 minutes
TIMEOUT    = 10

_cache = {}
_lock  = threading.Lock()


def get_agro_data(lat: float, lon: float) -> dict:
    """
    Fetch agriculture-relevant weather data from Open-Meteo.
    Returns soil moisture, ET0, rainfall forecast for given coordinates.
    """
    cache_key = f"{round(lat,3)}_{round(lon,3)}"
    with _lock:
        entry = _cache.get(cache_key)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['data']

    if not _HAS_REQUESTS:
        return _fallback()

    params = {
        'latitude':   lat,
        'longitude':  lon,
        'timezone':   'Asia/Kolkata',
        # Hourly
        'hourly': ','.join([
            'soil_moisture_0_to_1cm',
            'soil_moisture_1_to_3cm',
            'evapotranspiration',
            'precipitation',
        ]),
        # Daily
        'daily': ','.join([
            'precipitation_sum',
            'et0_fao_evapotranspiration',
            'uv_index_max',
            'windspeed_10m_max',
        ]),
        'forecast_days': 7,
        'current_weather': 'true',
    }

    try:
        r = _req.get(BASE_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        raw = r.json()

        # Extract most relevant values
        daily   = raw.get('daily',   {})
        hourly  = raw.get('hourly',  {})
        current = raw.get('current_weather', {})

        # 7-day precip sum
        precip_7d = sum(v for v in (daily.get('precipitation_sum') or []) if v is not None)
        # 3-day precip sum
        precip_3d = sum((daily.get('precipitation_sum') or [0,0,0])[:3] if daily.get('precipitation_sum') else [0])
        # Current soil moisture (most recent hourly value)
        sm_vals   = [v for v in (hourly.get('soil_moisture_0_to_1cm') or []) if v is not None]
        soil_moist= round(sm_vals[0] * 100, 1) if sm_vals else None   # fraction → %
        # ET0 today
        et0_vals  = daily.get('et0_fao_evapotranspiration') or []
        et0_today = round(et0_vals[0], 2) if et0_vals else None
        # UV index max today
        uv_vals   = daily.get('uv_index_max') or []
        uv_max    = uv_vals[0] if uv_vals else None

        result = {
            'source':         'open-meteo.com',
            'lat':            lat,
            'lon':            lon,
            'temperature':    current.get('temperature'),
            'windspeed':      current.get('windspeed'),
            'precip_3d_mm':   round(precip_3d, 1),
            'precip_7d_mm':   round(precip_7d, 1),
            'soil_moisture_pct': soil_moist,       # 0–100%
            'et0_mm_day':     et0_today,           # mm/day
            'uv_index_max':   uv_max,
            'daily_precip':   (daily.get('precipitation_sum') or [])[:7],
            'daily_et0':      (daily.get('et0_fao_evapotranspiration') or [])[:7],
            'success':        True,
        }

        with _lock:
            _cache[cache_key] = {'data': result, 'ts': time.time()}
        return result

    except Exception as e:
        return _fallback(str(e))


def _fallback(err=''):
    return {
        'source':           'open-meteo.com',
        'success':          False,
        'error':            err or 'Open-Meteo unavailable',
        'precip_3d_mm':     None,
        'precip_7d_mm':     None,
        'soil_moisture_pct': None,
        'et0_mm_day':       None,
        'uv_index_max':     None,
    }
