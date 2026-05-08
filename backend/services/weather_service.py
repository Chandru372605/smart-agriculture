"""
AgroSense — Weather Service
============================
Wraps the OpenWeatherMap free API (Current + 5-day forecast).
Results are cached in-process for CACHE_TTL seconds to stay within the
free-tier rate limit (60 calls/min, 1000 calls/day).

API key is read from the WEATHER_API_KEY environment variable (or config.py).
If the key is missing or the call fails, all functions return structured
"unavailable" dicts — no exceptions bubble up to the routes.

Endpoints used:
  • https://api.openweathermap.org/data/2.5/weather     (current)
  • https://api.openweathermap.org/data/2.5/forecast    (5-day / 3-hr)
  • https://api.openweathermap.org/geo/1.0/direct       (city → lat/lon)
"""

import time
import urllib.request
import urllib.parse
import json
import threading
from datetime import datetime

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL = 600          # seconds (10 minutes)
_cache: dict = {}
_lock  = threading.Lock()

# ── OWM base URL ─────────────────────────────────────────────────────────────
_BASE  = "https://api.openweathermap.org"


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(url: str) -> dict | None:
    """HTTP GET → parsed JSON, or None on any error."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _cached(key: str, fetcher):
    """Return cached value if fresh, else call fetcher() and cache result."""
    with _lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['data']
    data = fetcher()
    with _lock:
        _cache[key] = {'data': data, 'ts': time.time()}
    return data


def _wind_dir(deg: float) -> str:
    dirs = ['N','NE','E','SE','S','SW','W','NW']
    return dirs[round(deg / 45) % 8]


def _uv_label(uvi: float) -> str:
    if uvi < 3:   return 'Low'
    if uvi < 6:   return 'Moderate'
    if uvi < 8:   return 'High'
    if uvi < 11:  return 'Very High'
    return 'Extreme'


def _weather_icon(code: int) -> str:
    """Map OWM condition code to a single emoji."""
    if   code == 800:              return '☀️'
    elif code in (801, 802):       return '⛅'
    elif code in (803, 804):       return '☁️'
    elif 200 <= code < 300:        return '⛈️'
    elif 300 <= code < 400:        return '🌧️'
    elif 500 <= code < 600:        return '🌦️'
    elif 600 <= code < 700:        return '🌨️'
    elif 700 <= code < 800:        return '🌫️'
    return '🌡️'


# ─────────────────────────────────────────────────────────────────────────────
#  Geo-coding
# ─────────────────────────────────────────────────────────────────────────────

def geocode(city: str, api_key: str) -> tuple[float, float] | None:
    """Resolve city name → (lat, lon).  Returns None if not found."""
    q   = urllib.parse.quote(city)
    url = f"{_BASE}/geo/1.0/direct?q={q}&limit=1&appid={api_key}"
    res = _fetch(url)
    if res and isinstance(res, list) and len(res) > 0:
        return res[0]['lat'], res[0]['lon']
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Current weather
# ─────────────────────────────────────────────────────────────────────────────

def get_current(lat: float, lon: float, api_key: str) -> dict:
    """
    Returns a normalised current-weather dict.
    Keys guaranteed to be present (use sensible defaults when data missing).
    """
    cache_key = f"current_{lat:.3f}_{lon:.3f}"

    def _fetch_current():
        url = (f"{_BASE}/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={api_key}&units=metric")
        return _fetch(url)

    raw = _cached(cache_key, _fetch_current)

    if raw is None or 'main' not in raw:
        return _unavailable_current()

    main    = raw.get('main', {})
    wind    = raw.get('wind', {})
    rain    = raw.get('rain', {})
    clouds  = raw.get('clouds', {})
    weather = raw.get('weather', [{}])[0]
    sys     = raw.get('sys', {})
    code    = weather.get('id', 800)

    return {
        'available':    True,
        'city':         raw.get('name', '—'),
        'country':      sys.get('country', ''),
        'temp':         round(main.get('temp', 0), 1),
        'feels_like':   round(main.get('feels_like', 0), 1),
        'temp_min':     round(main.get('temp_min', 0), 1),
        'temp_max':     round(main.get('temp_max', 0), 1),
        'humidity':     main.get('humidity', 0),
        'pressure':     main.get('pressure', 1013),
        'description':  weather.get('description', '—').capitalize(),
        'icon':         _weather_icon(code),
        'icon_code':    weather.get('icon', '01d'),
        'wind_speed':   round(wind.get('speed', 0) * 3.6, 1),   # m/s → km/h
        'wind_dir':     _wind_dir(wind.get('deg', 0)),
        'clouds':       clouds.get('all', 0),
        'rain_1h':      round(rain.get('1h', 0), 1),
        'rain_3h':      round(rain.get('3h', 0), 1),
        'visibility':   round(raw.get('visibility', 10000) / 1000, 1),  # m → km
        'updated_at':   datetime.utcnow().strftime('%H:%M UTC'),
    }


def _unavailable_current() -> dict:
    return {
        'available':   False,
        'city':        'Unavailable',
        'country':     '',
        'temp':        0, 'feels_like': 0, 'temp_min': 0, 'temp_max': 0,
        'humidity':    0, 'pressure':   1013,
        'description': 'Weather data unavailable',
        'icon':        '❓', 'icon_code': '01d',
        'wind_speed':  0,   'wind_dir':  'N',
        'clouds':      0,   'rain_1h':   0,   'rain_3h': 0,
        'visibility':  0,   'updated_at': '—',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5-day / 3-hour forecast  →  condensed daily summary
# ─────────────────────────────────────────────────────────────────────────────

def get_forecast(lat: float, lon: float, api_key: str, days: int = 5) -> list[dict]:
    """
    Returns a list of daily forecast dicts (up to `days` days).
    Each entry has keys: date, day_name, temp_min, temp_max, humidity,
    rain_mm, description, icon, wind_speed.
    """
    cache_key = f"forecast_{lat:.3f}_{lon:.3f}"

    def _fetch_forecast():
        url = (f"{_BASE}/data/2.5/forecast"
               f"?lat={lat}&lon={lon}&appid={api_key}&units=metric&cnt=40")
        return _fetch(url)

    raw = _cached(cache_key, _fetch_forecast)

    if raw is None or 'list' not in raw:
        return []

    # Aggregate 3-hour slots → daily buckets
    daily: dict[str, dict] = {}
    for slot in raw['list']:
        dt_txt = slot.get('dt_txt', '')
        date   = dt_txt[:10]          # 'YYYY-MM-DD'
        if date not in daily:
            dt_obj = datetime.strptime(date, '%Y-%m-%d')
            daily[date] = {
                'date':      date,
                'day_name':  dt_obj.strftime('%a'),
                'temps':     [],
                'humidities':[],
                'rain_mm':   0.0,
                'codes':     [],
                'descs':     [],
                'wind_speeds':[],
            }
        d = daily[date]
        m = slot.get('main', {})
        d['temps'].append(m.get('temp', 0))
        d['humidities'].append(m.get('humidity', 0))
        d['rain_mm'] += slot.get('rain', {}).get('3h', 0)
        w = slot.get('weather', [{}])[0]
        d['codes'].append(w.get('id', 800))
        d['descs'].append(w.get('description', ''))
        d['wind_speeds'].append(slot.get('wind', {}).get('speed', 0) * 3.6)

    result = []
    for date, d in sorted(daily.items())[:days]:
        temps  = d['temps']
        code   = max(set(d['codes']), key=d['codes'].count)  # modal code
        desc   = max(set(d['descs']), key=d['descs'].count)
        result.append({
            'date':       date,
            'day_name':   d['day_name'],
            'temp_min':   round(min(temps), 1),
            'temp_max':   round(max(temps), 1),
            'humidity':   round(sum(d['humidities']) / len(d['humidities'])),
            'rain_mm':    round(d['rain_mm'], 1),
            'description':desc.capitalize(),
            'icon':       _weather_icon(code),
            'wind_speed': round(max(d['wind_speeds']), 1),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Agro-context helper — fields ready to pre-fill irrigation/pest forms
# ─────────────────────────────────────────────────────────────────────────────

def agro_context(lat: float, lon: float, api_key: str) -> dict:
    """
    Combines current + forecast into a single dict that the frontend can use
    to auto-fill form fields.
    """
    curr     = get_current(lat, lon, api_key)
    forecast = get_forecast(lat, lon, api_key, days=3)

    rain_3day = sum(f['rain_mm'] for f in forecast) if forecast else curr['rain_3h']
    avg_hum   = (sum(f['humidity'] for f in forecast) / len(forecast)
                 if forecast else curr['humidity'])

    return {
        'temperature':        curr['temp'],
        'humidity':           round(avg_hum),
        'rainfall_forecast':  round(rain_3day, 1),
        'wind_speed':         curr['wind_speed'],
        'description':        curr['description'],
        'icon':               curr['icon'],
        'city':               curr['city'],
        'available':          curr['available'],
    }
