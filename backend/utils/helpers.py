"""
AgroSense — Shared helpers used across API route blueprints.
"""
import os
import joblib
import numpy as np
from flask import jsonify

# ── Lazy model cache ──────────────────────────────────────────────────
_cache = {}

def load_model(path: str, key: str = None):
    """
    Load a joblib .pkl artifact with a simple in-process cache.
    Returns None if the file does not exist (route will fall back gracefully).
    """
    k = key or path
    if k not in _cache:
        if os.path.exists(path):
            try:
                _cache[k] = joblib.load(path)
            except Exception:
                _cache[k] = None
        else:
            _cache[k] = None
    return _cache[k]


def load_keras_model(path: str, key: str = None):
    """Load a Keras .keras / .h5 model with caching.
    Returns None if TensorFlow is not installed or the file is missing.
    """
    k = key or path
    if k not in _cache:
        if not os.path.exists(path):
            _cache[k] = None
        else:
            try:
                os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
                os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
                from tensorflow import keras
                _cache[k] = keras.models.load_model(path)
            except Exception:
                # TensorFlow not installed or model incompatible — use RF fallback
                _cache[k] = None
    return _cache[k]


def err(msg: str, code: int = 400):
    """Return a JSON error response."""
    return jsonify({'error': msg}), code


def fmt_inr(value: float) -> str:
    """Format a number as Indian Rupees with commas (handles negatives)."""
    try:
        v = int(round(value))
        negative = v < 0
        s = f"{abs(v):,}"
        return f"-₹{s}" if negative else f"₹{s}"
    except Exception:
        return str(value)


def safe_float(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def safe_int(d: dict, key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default
