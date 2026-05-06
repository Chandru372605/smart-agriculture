"""
AgroSense — Disease Detection API
POST /api/disease/predict          (multipart image upload)
POST /api/disease/predict-sample   (JSON — sample button path)
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.helpers import load_model, err
from backend.config import Config
from backend.models.db_models import log_prediction

disease_bp = Blueprint('disease', __name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Treatment database — covers all 39 PlantVillage classes
#  Keys match the exact folder names used by the Kaggle dataset (and therefore
#  the class names saved inside disease_classes.pkl by train_disease.py).
# ──────────────────────────────────────────────────────────────────────────────
TREATMENT_DB = {

    # ── APPLE ──────────────────────────────────────────────────────────────
    'Apple___Apple_scab': {
        'treatment':  'Spray Captan (0.3%) or Mancozeb (0.25%) during pink bud to petal fall stage. Repeat at 10-day intervals until terminal bud set.',
        'parts':      ['Leaves', 'Fruits', 'Young shoots'],
        'preventive': ['Plant resistant varieties (e.g., Liberty, Enterprise)',
                       'Rake and destroy fallen infected leaves',
                       'Apply fungicide before and after rain events',
                       'Prune canopy for good air circulation'],
    },
    'Apple___Black_rot': {
        'treatment':  'Apply Captan (0.2%) or Thiophanate-methyl. Remove mummified fruits and infected wood. Prune 15 cm beyond visible canker margin.',
        'parts':      ['Fruits', 'Leaves', 'Bark / Cankers'],
        'preventive': ['Eliminate all mummified fruits before bud break',
                       'Avoid wounding bark during cultivation',
                       'Apply protective copper sprays at dormant stage',
                       'Ensure proper orchard sanitation after harvest'],
    },
    'Apple___Cedar_apple_rust': {
        'treatment':  'Spray Myclobutanil (0.1%) or Triadimefon at pink stage and petal fall. Repeat every 7–10 days for 3 applications.',
        'parts':      ['Leaves', 'Fruits', 'Leaf petioles'],
        'preventive': ['Remove nearby eastern red cedar (Juniperus virginiana) trees',
                       'Plant rust-resistant apple varieties',
                       'Apply protective fungicide starting at half-inch green stage'],
    },
    'Apple___healthy': {
        'treatment':  'Tree is healthy. Maintain current orchard management practices.',
        'parts':      [],
        'preventive': ['Annual dormant pruning to maintain open canopy',
                       'Monitor soil pH (6.0–6.5) and correct with lime if needed',
                       'Apply balanced NPK fertiliser annually',
                       'Integrated pest management scouting weekly'],
    },

    # ── BLUEBERRY ──────────────────────────────────────────────────────────
    'Blueberry___healthy': {
        'treatment':  'Blueberry plant is healthy. Continue standard care.',
        'parts':      [],
        'preventive': ['Maintain soil pH 4.5–5.5 using sulfur amendments',
                       'Mulch with pine bark (10 cm depth) to conserve moisture',
                       'Monitor for mummy berry and apply fungicide at bud swell',
                       'Net plants to protect ripening fruit from birds'],
    },

    # ── CHERRY ─────────────────────────────────────────────────────────────
    'Cherry___Powdery_mildew': {
        'treatment':  'Spray Sulphur (0.3%) or Myclobutanil (0.1%). Apply when first colonies appear. Repeat every 10–14 days. Avoid high-nitrogen fertilisation.',
        'parts':      ['Leaves', 'Young shoots', 'Fruit surface'],
        'preventive': ['Prune to improve air circulation inside canopy',
                       'Avoid excessive nitrogen which promotes succulent growth',
                       'Apply preventive sulphur spray at bud break',
                       'Remove and destroy infected shoot tips'],
    },
    'Cherry___healthy': {
        'treatment':  'Cherry tree is healthy. No intervention required.',
        'parts':      [],
        'preventive': ['Apply dormant oil spray to control scale insects',
                       'Monitor for cherry leaf spot; apply fungicide post-bloom',
                       'Thin fruits to 5 cm spacing for larger fruit size',
                       'Protect blossoms from late frost with overhead irrigation'],
    },

    # ── CORN (MAIZE) ───────────────────────────────────────────────────────
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'treatment':  'Apply Propiconazole (0.1%) or Azoxystrobin at first sign of lesions. Scout from V5 stage and spray if lesions exceed threshold on lower leaves.',
        'parts':      ['Leaves', 'Leaf sheaths'],
        'preventive': ['Plant tolerant hybrids (most commercial hybrids have partial resistance)',
                       'Rotate crops — avoid continuous maize',
                       'Reduce residue with tillage in high-disease seasons',
                       'Avoid excessively dense planting for better air flow'],
    },
    'Corn___Common_rust': {
        'treatment':  'Apply Propiconazole (0.1%) or Tebuconazole at disease onset. Most field corn is tolerant — economic threshold determines spray timing.',
        'parts':      ['Leaves'],
        'preventive': ['Plant resistant hybrids — primary management strategy',
                       'Early planting avoids peak rust season',
                       'Scout fields from V5 stage onwards',
                       'Chemical control is rarely economical for field corn'],
    },
    'Corn___Northern_Leaf_Blight': {
        'treatment':  'Spray Azoxystrobin + Propiconazole or Pyraclostrobin at tasseling if disease detected on ear leaf or above. Apply at VT-R1 growth stage.',
        'parts':      ['Leaves', 'Leaf sheaths'],
        'preventive': ['Use resistant hybrids (rated ≤ 4 on disease scale)',
                       'Crop rotation to reduce residue inoculum',
                       'Scout from V6 stage — lesion count drives spray decision',
                       'Avoid planting in fields with history of severe NLB'],
    },
    'Corn___healthy': {
        'treatment':  'Plant appears healthy. Maintain current agronomic practices.',
        'parts':      [],
        'preventive': ['Regular field scouting for fall armyworm',
                       'Balanced NPK fertilisation — apply split nitrogen',
                       'Timely irrigation at silking stage is critical',
                       'Monitor for stalk rot at maturity (push test)'],
    },

    # ── GRAPE ──────────────────────────────────────────────────────────────
    'Grape___Black_rot': {
        'treatment':  'Apply Mancozeb (0.25%) or Myclobutanil starting at bud break. Repeat every 7–10 days through berry touch. Remove mummified berries immediately.',
        'parts':      ['Berries', 'Leaves', 'Shoots', 'Tendrils'],
        'preventive': ['Remove all mummified berries before new growth begins',
                       'Ensure good canopy airflow through proper shoot positioning',
                       'Apply protectant fungicide before each rain event',
                       'Use resistant varieties where available'],
    },
    'Grape___Esca_(Black_Measles)': {
        'treatment':  'No curative chemical treatment available. Remove and burn infected vines. Apply sodium arsenite wound paste on pruning cuts (where legally permitted).',
        'parts':      ['Wood (vascular)', 'Leaves', 'Berries'],
        'preventive': ['Delay pruning to dry weather — reduce wound infection risk',
                       'Apply wound sealant (e.g., Trichoderma-based paste) immediately after pruning',
                       'Avoid excessive vine stress (drought, over-cropping)',
                       'Disinfect pruning tools with 10% bleach solution between vines'],
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'treatment':  'Apply Copper oxychloride (0.3%) or Mancozeb at first sign. Repeat every 10 days during humid conditions. Remove severely infected leaves.',
        'parts':      ['Leaves', 'Petioles'],
        'preventive': ['Maintain good canopy management for air flow',
                       'Avoid overhead irrigation which prolongs leaf wetness',
                       'Apply protective copper spray pre-monsoon',
                       'Collect and destroy fallen infected leaves'],
    },
    'Grape___healthy': {
        'treatment':  'Vine is healthy. Continue standard vineyard management.',
        'parts':      [],
        'preventive': ['Apply dormant copper + oil spray before bud break',
                       'Manage canopy for 30–40% light penetration',
                       'Monitor for downy mildew after first rains',
                       'Soil test annually and adjust nutrition accordingly'],
    },

    # ── ORANGE ─────────────────────────────────────────────────────────────
    'Orange___Haunglongbing_(Citrus_greening)': {
        'treatment':  'No cure exists. Remove and destroy infected trees immediately to prevent spread. Inject thermotherapy (45°C hot water treatment) on early-stage budwood.',
        'parts':      ['Phloem (systemic)', 'Leaves', 'Fruits', 'Roots'],
        'preventive': ['Control Asian citrus psyllid (Diaphorina citri) — the insect vector — with Imidacloprid',
                       'Use certified disease-free nursery budwood only',
                       'Inspect new trees for blotchy mottle symptoms before planting',
                       'Establish buffer zones and report to agriculture department immediately'],
    },

    # ── PEACH ──────────────────────────────────────────────────────────────
    'Peach___Bacterial_spot': {
        'treatment':  'Apply copper-based bactericides (Copper hydroxide 0.3%) from shuck split through harvest. Oxytetracycline (200 ppm) can be used at petal fall.',
        'parts':      ['Leaves', 'Fruits', 'Young twigs'],
        'preventive': ['Plant resistant varieties (e.g., Contender, Redhaven)',
                       'Avoid overhead sprinkler irrigation',
                       'Remove and burn infected twigs during dormant pruning',
                       'Apply copper spray at leaf fall and dormant stage'],
    },
    'Peach___healthy': {
        'treatment':  'Tree appears healthy. No intervention needed.',
        'parts':      [],
        'preventive': ['Thin fruits to 15–20 cm spacing after June drop',
                       'Apply dormant oil + copper spray before bud swell',
                       'Monitor for peach leaf curl starting at bud break',
                       'Irrigate deeply every 10–14 days in dry periods'],
    },

    # ── PEPPER (BELL) ──────────────────────────────────────────────────────
    'Pepper,_bell___Bacterial_spot': {
        'treatment':  'Apply copper hydroxide (0.3%) + Mancozeb tank mix. Remove and destroy infected leaves. Avoid overhead irrigation.',
        'parts':      ['Leaves', 'Stems', 'Fruits'],
        'preventive': ['Use certified disease-free transplants',
                       'Rotate crops — avoid planting after tomato/eggplant',
                       'Stake plants to improve air circulation',
                       'Apply copper spray preventively in humid weather'],
    },
    'Pepper,_bell___healthy': {
        'treatment':  'Plant is healthy. Maintain current cultivation practices.',
        'parts':      [],
        'preventive': ['Soil solarisation before transplanting reduces soilborne pathogens',
                       'Apply 20:20:20 NPK at 5g/L foliar spray at flowering',
                       'Monitor for thrips and aphids — vectors of pepper viruses',
                       'Maintain consistent irrigation to avoid blossom drop'],
    },

    # ── POTATO ─────────────────────────────────────────────────────────────
    'Potato___Early_blight': {
        'treatment':  'Apply Chlorothalonil (0.2%) or Mancozeb at disease onset. Repeat every 7–10 days. Remove lower infected leaves. Ensure drainage.',
        'parts':      ['Lower leaves (oldest first)', 'Stems', 'Tubers'],
        'preventive': ['Avoid over-watering and nitrogen excess',
                       'Mulch to prevent soil splash onto lower leaves',
                       'Remove plant debris after harvest',
                       'Ensure 2–3 year crop rotation away from solanaceous crops'],
    },
    'Potato___Late_blight': {
        'treatment':  'Apply Cymoxanil + Mancozeb (Curzate) or Chlorothalonil immediately. Haulm destruction 10 days before harvest prevents tuber blight.',
        'parts':      ['Leaves', 'Stems', 'Tubers'],
        'preventive': ['Use certified blight-free seed potatoes',
                       'Plant in well-drained soil on raised beds',
                       'Avoid excessive nitrogen fertilisation',
                       'Monitor weather — blight favours cool (10–25°C), wet conditions'],
    },
    'Potato___healthy': {
        'treatment':  'Plant appears healthy. No intervention needed.',
        'parts':      [],
        'preventive': ['Crop rotation — avoid planting after tomato/pepper for 3 years',
                       'Hill up soil around stems at 3 and 6 weeks after emergence',
                       'Reduce irrigation 2 weeks before harvest to harden tubers',
                       'Monitor for Colorado potato beetle from plant emergence'],
    },

    # ── RASPBERRY ──────────────────────────────────────────────────────────
    'Raspberry___healthy': {
        'treatment':  'Canes are healthy. Continue standard raspberry management.',
        'parts':      [],
        'preventive': ['Prune out and destroy all floricanes after harvest',
                       'Apply pre-emergent herbicide in early spring',
                       'Trellis canes for support and air circulation',
                       'Apply Botrytis fungicide at 10% and 50% bloom'],
    },

    # ── SOYBEAN ────────────────────────────────────────────────────────────
    'Soybean___healthy': {
        'treatment':  'Plant is healthy. No treatment required.',
        'parts':      [],
        'preventive': ['Inoculate seeds with Bradyrhizobium japonicum before planting',
                       'Apply molybdenum (500g/ha) for better nitrogen fixation',
                       'Scout for soybean aphid from V3 stage',
                       'Harvest when 95% of pods turn brown to minimise shattering'],
    },

    # ── SQUASH ─────────────────────────────────────────────────────────────
    'Squash___Powdery_mildew': {
        'treatment':  'Spray Sulphur (0.3%) or Azoxystrobin at first sign. Apply in morning (avoid afternoon heat). Repeat every 7–10 days. Remove severely infected leaves.',
        'parts':      ['Leaves (upper surface)', 'Stems', 'Petioles'],
        'preventive': ['Plant resistant varieties where available',
                       'Avoid overhead irrigation — water at base',
                       'Space plants for good air circulation',
                       'Apply potassium bicarbonate (1%) as organic option'],
    },

    # ── STRAWBERRY ─────────────────────────────────────────────────────────
    'Strawberry___Leaf_scorch': {
        'treatment':  'Apply Captan or Copper hydroxide (0.3%) at first lesion appearance. Remove old, infected leaves. Avoid overhead irrigation.',
        'parts':      ['Leaves', 'Petioles', 'Calyxes'],
        'preventive': ['Renovate planting annually — remove old foliage after last harvest',
                       'Avoid dense planting — maintain 30 cm row spacing',
                       'Apply mulch to reduce soil splash',
                       'Use drip irrigation instead of overhead sprinklers'],
    },
    'Strawberry___healthy': {
        'treatment':  'Plant is healthy. Continue current strawberry management.',
        'parts':      [],
        'preventive': ['Apply Botrytis fungicide at 10% bloom for grey mould prevention',
                       'Renovate beds annually — mow to 10 cm and thin runners',
                       'Maintain soil pH 5.5–6.5 with sulfur or lime',
                       'Monitor for two-spotted spider mite in dry conditions'],
    },

    # ── TOMATO ─────────────────────────────────────────────────────────────
    'Tomato___Bacterial_spot': {
        'treatment':  'Apply copper hydroxide + Mancozeb tank mix (0.3%). Spray every 5–7 days during wet conditions. Remove heavily infected leaves.',
        'parts':      ['Leaves', 'Stems', 'Fruits'],
        'preventive': ['Use certified disease-free transplants from reputable nurseries',
                       'Avoid overhead irrigation — use drip systems',
                       'Rotate tomatoes away from peppers and eggplant for 2 years',
                       'Apply copper spray at transplanting as a protective measure'],
    },
    'Tomato___Early_blight': {
        'treatment':  'Apply Chlorothalonil (0.2%) or Mancozeb at disease onset. Remove lower infected leaves. Ensure good drainage and air circulation.',
        'parts':      ['Lower leaves (oldest first)', 'Stems', 'Fruits'],
        'preventive': ['Mulch soil surface to prevent splash dispersal of spores',
                       'Avoid wetting foliage when irrigating',
                       'Remove all plant debris after harvest',
                       '2–3 year crop rotation away from solanaceous crops'],
    },
    'Tomato___Late_blight': {
        'treatment':  'Apply Mancozeb (0.25%) or Metalaxyl + Mancozeb (Ridomil Gold) at 7-day intervals. Remove and destroy infected plant parts immediately.',
        'parts':      ['Leaves', 'Stems', 'Fruits'],
        'preventive': ['Use certified disease-free seeds and transplants',
                       'Avoid overhead irrigation — keep foliage dry',
                       'Maintain proper plant spacing for air circulation',
                       'Apply preventive copper-based fungicide at first sign of rain'],
    },
    'Tomato___Leaf_Mold': {
        'treatment':  'Apply Chlorothalonil or Copper oxychloride (0.3%). Improve ventilation in greenhouse/polyhouse. Reduce humidity below 85%.',
        'parts':      ['Leaf undersides (primary)', 'Upper leaf surfaces'],
        'preventive': ['Maintain relative humidity below 85% in protected cultivation',
                       'Install side ventilation in polyhouses',
                       'Avoid dense planting — remove suckers regularly',
                       'Remove and destroy fallen infected leaves promptly'],
    },
    'Tomato___Septoria_leaf_spot': {
        'treatment':  'Apply Mancozeb (0.25%) or Chlorothalonil at first sign. Remove infected lower leaves. Repeat spray every 7–10 days.',
        'parts':      ['Leaves (lower then upper)', 'Stems (severe cases)'],
        'preventive': ['Avoid working in the crop when foliage is wet',
                       'Stake or cage plants to reduce soil contact',
                       'Remove infected leaves and destroy — do not compost',
                       '2–3 year rotation away from tomato family plants'],
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'treatment':  'Apply Abamectin (0.5 ml/L) or Spiromesifen. Spray undersides of leaves thoroughly. Repeat after 7 days. Avoid broad-spectrum insecticides that kill predators.',
        'parts':      ['Leaf undersides', 'Young shoots'],
        'preventive': ['Release predatory mite Phytoseiulus persimilis (biological control)',
                       'Maintain adequate soil moisture — drought stress worsens mite outbreaks',
                       'Avoid excess nitrogen which promotes succulent growth',
                       'Use reflective mulch to deter migrating mite populations'],
    },
    'Tomato___Target_Spot': {
        'treatment':  'Apply Azoxystrobin or Chlorothalonil at first lesion appearance. Remove and destroy infected foliage. Repeat every 7–14 days.',
        'parts':      ['Leaves', 'Stems', 'Fruits'],
        'preventive': ['Avoid dense planting — ensure good canopy airflow',
                       'Stake or trellis plants to reduce leaf contact with soil',
                       'Apply mulch to reduce soil-borne spore splash',
                       'Monitor lower leaves weekly from transplanting'],
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'treatment':  'No cure — remove and destroy infected plants immediately. Control whitefly vector with Imidacloprid (0.3 ml/L) or yellow sticky traps.',
        'parts':      ['Leaves (systemic — whole plant affected)', 'Growing tips'],
        'preventive': ['Use TYLCV-resistant varieties (e.g., TY-1, TY-2 gene hybrids)',
                       'Control Bemisia tabaci (whitefly) vector aggressively',
                       'Install 40-mesh insect-proof netting in nurseries',
                       'Remove and bury infected plant material; never compost'],
    },
    'Tomato___Tomato_mosaic_virus': {
        'treatment':  'No chemical cure. Remove infected plants. Sanitize tools with 1% sodium hypochlorite. Wash hands between plants.',
        'parts':      ['Leaves (mosaic pattern)', 'Fruits (discolouration)', 'Whole plant (stunting)'],
        'preventive': ['Use virus-indexed certified seed',
                       'Control aphid vectors with reflective mulch',
                       'Do not smoke near tomato plants (tobacco is a reservoir)',
                       'Rogue out and destroy infected plants at first symptom'],
    },
    'Tomato___healthy': {
        'treatment':  'Plant is healthy. Continue current care practices.',
        'parts':      [],
        'preventive': ['Regular scouting for whitefly, aphid, and mite populations',
                       'Stake plants at transplanting for support and air circulation',
                       'Maintain consistent irrigation — avoid water stress at fruit set',
                       'Apply calcium nitrate (0.5%) foliar spray to prevent blossom end rot'],
    },

    # ── BACKGROUND (no leaf) ───────────────────────────────────────────────
    'Background_without_leaves': {
        'treatment':  'No plant material detected in the image. Please upload a clear photo of a leaf.',
        'parts':      [],
        'preventive': ['Ensure the leaf fills most of the image frame',
                       'Use natural light for better image quality',
                       'Avoid blurry or over-exposed images'],
    },
}

# Classes that represent healthy plants
HEALTHY_CLASSES = {
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy',
    'Corn___healthy', 'Grape___healthy', 'Peach___healthy',
    'Pepper,_bell___healthy', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Strawberry___healthy', 'Tomato___healthy',
}


def _prettify_class(cls: str) -> str:
    """Convert a PlantVillage folder name into a human-readable label."""
    # Handle special cases with brackets / commas
    cls = cls.replace('Pepper,_bell', 'Bell Pepper')
    cls = cls.replace('Grape___Esca_(Black_Measles)', 'Grape — Esca (Black Measles)')
    cls = cls.replace('Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape — Isariopsis Leaf Spot')
    cls = cls.replace('Orange___Haunglongbing_(Citrus_greening)', 'Orange — Citrus Greening (HLB)')
    cls = cls.replace('Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato — Two-spotted Spider Mite')
    cls = cls.replace('Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn — Gray Leaf Spot (Cercospora)')
    # Standard triple-underscore pattern
    cls = cls.replace('___', ' — ')
    cls = cls.replace('_', ' ')
    return cls


def _get_response(class_name: str, confidence: float) -> dict:
    db   = TREATMENT_DB.get(class_name, {})
    is_h = class_name in HEALTHY_CLASSES
    return {
        'name':       _prettify_class(class_name),
        'type':       'healthy' if is_h else 'diseased',
        'confidence': round(confidence, 1),
        'treatment':  db.get('treatment', 'Consult a local agricultural extension officer for diagnosis.'),
        'parts':      db.get('parts', []),
        'preventive': db.get('preventive', []),
    }


@disease_bp.route('/disease/predict', methods=['POST'])
def predict_image():
    """Real image inference — CNN (MobileNetV2) if available, else colour-histogram RF."""
    if 'image' not in request.files:
        return err('No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return err('Empty filename')

    try:
        import io
        from PIL import Image

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((128, 128))

        # ── Try CNN first ───────────────────────────────────────────────
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        from backend.utils.helpers import load_keras_model
        cnn = load_keras_model(Config.DISEASE_CNN_PATH, 'disease_cnn')
        if cnn is not None:
            from backend.utils.helpers import load_model as lm
            meta  = lm(Config.DISEASE_META_PATH, 'disease_meta')
            arr   = np.array(img, dtype=np.float32) / 255.0
            arr   = arr[np.newaxis, ...]
            preds = cnn.predict(arr, verbose=0)[0]
            idx   = int(preds.argmax())
            conf  = float(preds[idx]) * 100
            # disease_classes.pkl saved by train_disease.py is a plain list of class names
            if isinstance(meta, list):
                cls = meta[idx]
            else:
                cls = meta.get('classes', [])[idx]
            log_prediction('disease', f'CNN | {cls}', cls, confidence=round(conf, 1))
            return jsonify(_get_response(cls, conf))

        # ── Fallback: colour-histogram Random Forest ────────────────────
        meta_obj = load_model(Config.DISEASE_META_PATH, 'disease_meta')
        rf_model = load_model(Config.DISEASE_RF_PATH,   'disease_rf')
        if rf_model is None:
            return err('Disease model not found. Run train_disease.py first.')

        arr   = np.array(img, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        feat  = np.array([[
            r.mean(), g.mean(), b.mean(),
            r.std(),  g.std(),  b.std(),
            r.mean() / (g.mean() + 1),
            arr.std() / 128,
            arr.mean(),
            abs(r.mean() - g.mean()),
        ]])
        enc   = meta_obj['encoder']
        proba = rf_model.predict_proba(feat)[0]
        idx   = int(proba.argmax())
        conf  = float(proba[idx]) * 100
        cls   = enc.inverse_transform([idx])[0]
        log_prediction('disease', f'RF | {cls}', cls, confidence=round(conf, 1))
        return jsonify(_get_response(cls, conf))

    except Exception as e:
        return err(f'Image analysis error: {e}', 500)


@disease_bp.route('/disease/predict-sample', methods=['POST'])
def predict_sample():
    """Handle pre-defined sample button clicks — returns curated response."""
    data  = request.get_json(force=True) or {}
    name  = data.get('name', 'Unknown')
    conf  = float(data.get('conf', 85))

    SAMPLE_MAP = {
        'Tomato Leaf Blight': 'Tomato___Late_blight',
        'Healthy Corn':       'Corn___healthy',
        'Apple Scab':         'Apple___Apple_scab',
        'Potato Late Blight': 'Potato___Late_blight',
        'Healthy Rice':       'Soybean___healthy',   # closest healthy proxy
        'Tomato Mosaic Virus':'Tomato___Tomato_mosaic_virus',
        'Grape Black Rot':    'Grape___Black_rot',
        'Corn Gray Leaf Spot':'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    }
    cls = SAMPLE_MAP.get(name, 'Tomato___healthy')
    log_prediction('disease', f'Sample | {name}', cls, confidence=conf)
    return jsonify(_get_response(cls, conf))
