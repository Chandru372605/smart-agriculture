"""
AgroSense — Weather API Routes
================================
GET  /api/weather/current?city=<name>         → current conditions
GET  /api/weather/current?lat=<f>&lon=<f>     → by coordinates (browser geolocation)
GET  /api/weather/forecast?city=<name>        → 5-day daily forecast
GET  /api/weather/forecast?lat=<f>&lon=<f>
GET  /api/weather/agro-context?city=<name>    → pre-filled values for irrigation/pest forms
GET  /api/weather/agro-context?lat=<f>&lon=<f>
"""
from flask import Blueprint, request, jsonify
from backend.config import Config
from backend.services import weather_service as ws

weather_bp = Blueprint('weather', __name__)


def _resolve_location():
    """
    Return (lat, lon) from query params.
    Accepts ?lat=&lon=  OR  ?city=<name>.
    Returns (None, None) + error string on failure.
    """
    api_key = Config.WEATHER_API_KEY
    if not api_key:
        return None, None, 'WEATHER_API_KEY not configured. Add it to your .env file.'

    lat = request.args.get('lat')
    lon = request.args.get('lon')
    city = request.args.get('city', '').strip()

    if lat and lon:
        try:
            return float(lat), float(lon), None
        except ValueError:
            return None, None, 'Invalid lat/lon values.'

    if city:
        coords = ws.geocode(city, api_key)
        if coords:
            return coords[0], coords[1], None
        return None, None, f'City "{city}" not found.'

    return None, None, 'Provide ?city=<name> or ?lat=<f>&lon=<f>.'


@weather_bp.route('/weather/current', methods=['GET'])
def current():
    lat, lon, loc_err = _resolve_location()
    if loc_err:
        return jsonify({'error': loc_err, 'available': False}), 400
    data = ws.get_current(lat, lon, Config.WEATHER_API_KEY)
    return jsonify(data)


@weather_bp.route('/weather/forecast', methods=['GET'])
def forecast():
    lat, lon, loc_err = _resolve_location()
    if loc_err:
        return jsonify({'error': loc_err, 'available': False}), 400
    days = min(int(request.args.get('days', 5)), 5)
    data = ws.get_forecast(lat, lon, Config.WEATHER_API_KEY, days=days)
    return jsonify({'forecast': data, 'available': len(data) > 0})


@weather_bp.route('/weather/agro-context', methods=['GET'])
def agro_context():
    """Returns form-ready weather values for auto-filling irrigation/pest inputs."""
    lat, lon, loc_err = _resolve_location()
    if loc_err:
        return jsonify({'error': loc_err, 'available': False}), 400
    data = ws.agro_context(lat, lon, Config.WEATHER_API_KEY)
    return jsonify(data)


# ── Open-Meteo endpoint (free, no key required) ───────────────────────────────
from backend.services.openmeteo_service import get_agro_data as _om_get

@weather_bp.route('/openmeteo/agro', methods=['GET'])
def openmeteo_agro():
    """
    GET /api/openmeteo/agro?lat=<f>&lon=<f>
    Returns soil moisture, ET0, rainfall from Open-Meteo (free, no API key).
    Used to enhance Irrigation, Crop Recommendation, and Yield pages.
    """
    try:
        lat = float(request.args.get('lat', ''))
        lon = float(request.args.get('lon', ''))
    except (TypeError, ValueError):
        return jsonify({'error': 'Provide ?lat=<f>&lon=<f>'}), 400
    data = _om_get(lat, lon)
    return jsonify(data)
