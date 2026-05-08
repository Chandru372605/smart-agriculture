"""
AgroSense — Application Configuration
"""
import os
from dotenv import load_dotenv

# Load .env file from project root (silently ignored if file doesn't exist)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'ml_models', 'saved')

class Config:
    SECRET_KEY       = os.getenv('SECRET_KEY', 'agrosense-dev-secret-2024')
    DEBUG            = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB max upload

    # ── OpenWeatherMap API key ────────────────────────────────
    WEATHER_API_KEY  = os.getenv('WEATHER_API_KEY', '')

    # ── data.gov.in Mandi Prices API key ─────────────────────
    # Get free key at https://data.gov.in/user/me/api-keys
    DATAGOV_API_KEY  = os.getenv('DATAGOV_API_KEY', '')

    # ── Saved model paths ────────────────────────────────
    CROP_MODEL_PATH    = os.path.join(MODELS_DIR, 'crop_recommend.pkl')
    CROP_ENCODER_PATH  = os.path.join(MODELS_DIR, 'crop_label_encoder.pkl')

    DISEASE_CNN_PATH     = os.path.join(MODELS_DIR, 'disease_cnn.keras')
    DISEASE_RF_PATH      = os.path.join(MODELS_DIR, 'disease_demo_rf.pkl')
    DISEASE_META_PATH    = os.path.join(MODELS_DIR, 'disease_classes.pkl')
    DISEASE_SKLEARN_PATH = os.path.join(MODELS_DIR, 'disease_sklearn.pkl')

    IRRIG_MODEL_PATH   = os.path.join(MODELS_DIR, 'irrigation_model.pkl')

    YIELD_MODEL_PATH   = os.path.join(MODELS_DIR, 'yield_model.pkl')
    YIELD_ENC_PATH     = os.path.join(MODELS_DIR, 'yield_encoders.pkl')

    ROT_MODEL_PATH     = os.path.join(MODELS_DIR, 'rotation_model.pkl')
    ROT_ENC_PATH       = os.path.join(MODELS_DIR, 'rotation_encoders.pkl')

    PEST_MODEL_PATH    = os.path.join(MODELS_DIR, 'pest_model.pkl')
    PEST_ENC_PATH      = os.path.join(MODELS_DIR, 'pest_encoders.pkl')

    PROFIT_MODEL_PATH  = os.path.join(MODELS_DIR, 'profit_model.pkl')

    MARKET_MODEL_PATH  = os.path.join(MODELS_DIR, 'market_lstm.keras')
    MARKET_SCALER_PATH = os.path.join(MODELS_DIR, 'market_scaler.pkl')
    MARKET_META_PATH   = os.path.join(MODELS_DIR, 'market_meta.pkl')
