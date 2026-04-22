# AgroSense — Smart Agriculture AI Platform

> 8 integrated AI/ML modules to help farmers make data-driven decisions — from soil analysis to commodity price forecasting.

---

## 📋 Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Training the ML Models](#training-the-ml-models)
5. [Running the Application](#running-the-application)
6. [Connecting the Real API](#connecting-the-real-api)
7. [Disease Detection (CNN Upgrade)](#disease-detection-cnn-upgrade)
8. [Module Reference](#module-reference)
9. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
smart-agriculture/
├── backend/
│   ├── app.py                  # Flask entry point — 9 page routes
│   ├── config.py               # App configuration
│   ├── models/                 # SQLAlchemy DB models
│   ├── routes/                 # Per-module API blueprint routes
│   │   ├── crop_routes.py
│   │   ├── disease_routes.py
│   │   ├── irrigation_routes.py
│   │   ├── market_routes.py
│   │   ├── pest_routes.py
│   │   ├── profit_routes.py
│   │   ├── rotation_routes.py
│   │   └── yield_routes.py
│   └── utils/
│       └── helpers.py
├── frontend/
│   ├── static/
│   │   ├── css/style.css       # Global stylesheet
│   │   └── js/
│   │       ├── app.js          # Simulation mode (no backend needed)
│   │       └── api.js          # Live API mode (real Flask predictions)
│   └── templates/
│       ├── base.html           # Jinja2 base layout + navbar
│       ├── index.html          # Home — module grid
│       ├── crop_recommendation.html
│       ├── disease_detection.html
│       ├── irrigation.html
│       ├── yield_prediction.html
│       ├── crop_rotation.html
│       ├── pest_risk.html
│       ├── profit_estimator.html
│       └── market_price.html
├── ml_models/
│   ├── crop_recommendation/    train_crop.py
│   ├── disease_detection/      train_disease.py
│   ├── irrigation/             train_irrigation.py
│   ├── market_price/           train_market.py
│   ├── pest_risk/              train_pest.py
│   ├── profit_estimator/       train_profit.py
│   ├── crop_rotation/          train_rotation.py
│   ├── yield_prediction/       train_yield.py
│   └── saved/                  # ← All .pkl / .keras model files go here
├── datasets/                   # CSV datasets (auto-generated or Kaggle downloads)
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| pip | ≥ 23.0 |
| Git | any |
| RAM | ≥ 4 GB recommended |

---

## Quick Start

### 1. Clone / open the project
```bash
cd smart-agriculture
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow on Windows ≥ 2.11 runs on CPU only (no native CUDA).  
> Use WSL2 or the `tensorflow-directml` plugin for GPU support.

---

## Training the ML Models

All training scripts are self-contained. Run them once to generate  
model files in `ml_models/saved/`. Synthetic data is auto-generated  
if the real Kaggle dataset is not present.

```bash
# Crop Recommendation  (~10 sec, 99.5% accuracy)
python -X utf8 ml_models/crop_recommendation/train_crop.py

# Smart Irrigation  (~2 sec, 93.8% accuracy)
python -X utf8 ml_models/irrigation/train_irrigation.py

# Yield Prediction  (~5 sec, R²=0.97)
python -X utf8 ml_models/yield_prediction/train_yield.py

# Pest Risk  (~3 sec, R²=0.93)
python -X utf8 ml_models/pest_risk/train_pest.py

# Profit Estimator  (~2 sec, R²=0.67)
python -X utf8 ml_models/profit_estimator/train_profit.py

# Crop Rotation  (~10 sec, 85.8% accuracy)
python -X utf8 ml_models/crop_rotation/train_rotation.py

# Disease Detection  (~5 sec demo, or 20-60 min CNN)
python -X utf8 ml_models/disease_detection/train_disease.py

# Market Price LSTM  (~60 sec, MAE ₹87/quintal)
python -X utf8 ml_models/market_price/train_market.py
```

### Using real Kaggle datasets (optional, improves accuracy)

| Module | Dataset | Place at |
|---|---|---|
| Crop Recommendation | [atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | `datasets/Crop_recommendation.csv` |
| Yield Prediction | [abhinand05/crop-production-in-india](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india) | `datasets/crop_production.csv` |
| Disease Detection | [emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) | `datasets/disease_images/<ClassName>/` |
| Irrigation | [nelakurthisudheer/dataset-for-predicting-watering-the-plants](https://www.kaggle.com/datasets/nelakurthisudheer/dataset-for-predicting-watering-the-plants) | `datasets/irrigation.csv` |

Re-run the corresponding training script after placing the dataset — it auto-detects and uses the real data.

---

## Running the Application

```bash
# Start Flask development server
python backend/app.py
```

Open your browser at: **http://127.0.0.1:5000**

The app works in **simulation mode** by default (`app.js`) — no trained  
models are required. All 8 modules return realistic synthetic predictions.

---

## Connecting the Real API

To serve live predictions from trained models:

### Step 1 — Register API blueprints in `backend/app.py`

Add these lines after the page routes:
```python
from flask_cors import CORS
from backend.routes.crop_routes import crop_bp
from backend.routes.irrigation_routes import irrigation_bp
# ... import all other blueprints

CORS(app)
app.register_blueprint(crop_bp,       url_prefix='/api')
app.register_blueprint(irrigation_bp, url_prefix='/api')
# ... register all others
```

### Step 2 — Enable `api.js` in `base.html`

Uncomment (or add) this line **after** the `app.js` script tag:
```html
<script src="{{ url_for('static', filename='js/api.js') }}"></script>
```

`api.js` overrides all 8 simulation functions with real `fetch()` calls to  
`/api/*` endpoints. The DOM manipulation remains identical — only the data source changes.

### API Endpoint Summary

| Module | Method | Endpoint | Key inputs |
|---|---|---|---|
| Crop Recommendation | POST | `/api/crop/recommend` | N, P, K, temperature, humidity, ph, rainfall |
| Disease Detection | POST | `/api/disease/predict` | image (multipart) |
| Smart Irrigation | POST | `/api/irrigation/plan` | crop, soil_moisture, temperature, rainfall_forecast |
| Yield Prediction | POST | `/api/yield/predict` | crop, season, state, area, rainfall, fertiliser |
| Crop Rotation | POST | `/api/rotation/recommend` | current_crop, soil_type, region, n_level |
| Pest Risk | POST | `/api/pest/assess` | crop, temperature, humidity, prev_occurrence |
| Profit Estimator | POST | `/api/profit/calculate` | area, yield_per_ha, selling_price, cost inputs |
| Market Price | POST | `/api/market/forecast` | crop, current_price, forecast_days |

All endpoints return JSON. See individual `backend/routes/*.py` files for full request/response schemas.

---

## Disease Detection (CNN Upgrade)

The default disease model uses a colour-histogram Random Forest (demo).  
To upgrade to the full **MobileNetV2 CNN** (38 classes, ~95% accuracy):

```bash
# 1. Download PlantVillage from Kaggle
#    https://www.kaggle.com/datasets/emmarex/plantdisease

# 2. Extract so the structure looks like:
#    datasets/disease_images/Apple___Apple_scab/image001.jpg
#    datasets/disease_images/Tomato___healthy/image042.jpg
#    ...

# 3. Re-run training — CNN mode activates automatically
python -X utf8 ml_models/disease_detection/train_disease.py
```

Expected training time: **~20 min on CPU / ~5 min on GPU**.

---

## Module Reference

| # | Module | Algorithm | Saved files |
|---|---|---|---|
| 1 | Crop Recommendation | Random Forest (sklearn) | `crop_recommend.pkl`, `crop_label_encoder.pkl` |
| 2 | Disease Detection | RF demo / MobileNetV2 CNN | `disease_demo_rf.pkl` or `disease_cnn.keras`, `disease_classes.pkl` |
| 3 | Smart Irrigation | Decision Tree (sklearn) | `irrigation_model.pkl` |
| 4 | Yield Prediction | XGBoost Regressor | `yield_model.pkl`, `yield_encoders.pkl` |
| 5 | Crop Rotation | Random Forest (sklearn) | `rotation_model.pkl`, `rotation_encoders.pkl` |
| 6 | Pest Risk | RF Regressor (0–100 score) | `pest_model.pkl`, `pest_encoders.pkl` |
| 7 | Profit Estimator | Ridge Regression pipeline | `profit_model.pkl` |
| 8 | Market Price | LSTM (TensorFlow/Keras) | `market_lstm.keras`, `market_scaler.pkl`, `market_meta.pkl` |

---

## Troubleshooting

### `UnicodeEncodeError` on Windows (emoji in terminal)
```bash
# Always run training scripts with:
python -X utf8 ml_models/.../train_*.py
```

### `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install xgboost
```

### TensorFlow GPU warning on Windows
```
WARNING: TensorFlow GPU support is not available on native Windows for TF >= 2.11
```
This is expected. TF runs on CPU. To use GPU: install WSL2 or `tensorflow-directml`.

### Flask `url_for` errors on startup
Ensure you run from the **project root** (`smart-agriculture/`), not from inside `backend/`:
```bash
# Correct
python backend/app.py

# Wrong
cd backend && python app.py
```

### Port 5000 already in use
```bash
# Windows — find and kill the process
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

## License
MIT © AgroSense Team
