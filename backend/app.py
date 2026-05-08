"""
AgroSense — Flask Application Entry Point
Serves 9 page routes + 8 /api/* blueprint routes for live ML predictions
+ prediction history API + SQLite persistence.
"""
import sys, os, logging

# Make sure 'backend' package is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS

from backend.config import Config

# ─────────────────────────────────────────
#  Application factory
# ─────────────────────────────────────────
app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static',
)
app.config.from_object(Config)
CORS(app)

# ─────────────────────────────────────────
#  Database initialisation
# ─────────────────────────────────────────
from backend.models.db_models import init_db
init_db()

# ─────────────────────────────────────────
#  Import & register API blueprints
# ─────────────────────────────────────────
from backend.routes.crop_routes       import crop_bp
from backend.routes.disease_routes    import disease_bp
from backend.routes.irrigation_routes import irrigation_bp
from backend.routes.yield_routes      import yield_bp
from backend.routes.rotation_routes   import rotation_bp
from backend.routes.pest_routes       import pest_bp
from backend.routes.profit_routes     import profit_bp
from backend.routes.market_routes     import market_bp
from backend.routes.history_routes    import history_bp
from backend.routes.weather_routes    import weather_bp

API_PREFIX = '/api'
app.register_blueprint(crop_bp,       url_prefix=API_PREFIX)
app.register_blueprint(disease_bp,    url_prefix=API_PREFIX)
app.register_blueprint(irrigation_bp, url_prefix=API_PREFIX)
app.register_blueprint(yield_bp,      url_prefix=API_PREFIX)
app.register_blueprint(rotation_bp,   url_prefix=API_PREFIX)
app.register_blueprint(pest_bp,       url_prefix=API_PREFIX)
app.register_blueprint(profit_bp,     url_prefix=API_PREFIX)
app.register_blueprint(market_bp,     url_prefix=API_PREFIX)
app.register_blueprint(history_bp,    url_prefix=API_PREFIX)
app.register_blueprint(weather_bp,    url_prefix=API_PREFIX)


# ─────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/disease-detection')
def disease_detection():
    return render_template('disease_detection.html')

@app.route('/irrigation')
def irrigation():
    return render_template('irrigation.html')

@app.route('/yield-prediction')
def yield_prediction():
    return render_template('yield_prediction.html')

@app.route('/crop-rotation')
def crop_rotation():
    return render_template('crop_rotation.html')

@app.route('/pest-risk')
def pest_risk():
    return render_template('pest_risk.html')

@app.route('/profit-estimator')
def profit_estimator():
    return render_template('profit_estimator.html')

@app.route('/market-price')
def market_price():
    return render_template('market_price.html')

@app.route('/history')
def history():
    return render_template('history.html')


# ─────────────────────────────────────────
#  Health-check endpoint
# ─────────────────────────────────────────
from flask import jsonify

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0', 'modules': 9})


# ─────────────────────────────────────────
#  Favicon route (prevents 404 log spam)
# ─────────────────────────────────────────
from flask import send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.static_folder, 'img'),
        'favicon.png', mimetype='image/png'
    )


# ─────────────────────────────────────────
#  Error handlers
# ─────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'details': str(e)}), 500



# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
