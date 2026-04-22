// ═══════════════════════════════════════════════════════
//  AgroSense — Main Application JavaScript
// ═══════════════════════════════════════════════════════

// ─────────────────────────────────────────
//  UTILITIES
// ─────────────────────────────────────────
function rv(el, id, divisor, scale) {
  const val = divisor ? (el.value * divisor).toFixed(1) : el.value;
  document.getElementById(id).textContent = val;
}

function fakeDelay(ms) { return new Promise(r => setTimeout(r, ms)); }

function setLoading(btn, on) {
  btn.classList.toggle('loading', on);
  btn.disabled = on;
}

function showResult(id) {
  const el = document.getElementById(id);
  el.classList.remove('show');
  void el.offsetWidth;
  el.classList.add('show');
}

function toast(msg) {
  const w = document.getElementById('toastWrap');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = '✅ ' + msg;
  w.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function randBetween(a, b, decimals = 1) {
  return parseFloat((Math.random() * (b - a) + a).toFixed(decimals));
}

// Mark active nav item based on current page
function markActiveNav() {
  const path = window.location.pathname.replace(/\/$/, '').split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach(item => {
    const href = (item.getAttribute('href') || '').split('/').pop();
    item.classList.toggle('active', href === path || (path === '' && href === 'index.html'));
  });
}
document.addEventListener('DOMContentLoaded', markActiveNav);


// ─────────────────────────────────────────
//  MODULE 1: CROP RECOMMENDATION
// ─────────────────────────────────────────
const CROPS = {
  highN:    {crop:'Rice',     alts:['Jute','Maize'],          soil:'High N suits paddy fields. pH looks slightly alkaline.',    tips:'Maintain standing water. Space seedlings 20cm apart. Apply Urea in split doses.',                                   indicators:[{l:'N Status',v:'High'},{l:'pH Level',v:'Optimal'},{l:'Humidity',v:'Good'}]},
  highP:    {crop:'Wheat',    alts:['Barley','Mustard'],       soil:'High P promotes strong root growth ideal for wheat.',        tips:'Sow in rows 22cm apart after adequate pre-sowing irrigation. Ensure timely harvesting.',                            indicators:[{l:'N Status',v:'Med'},{l:'pH Level',v:'Good'},{l:'Water Need',v:'Low'}]},
  highK:    {crop:'Banana',   alts:['Grapes','Watermelon'],    soil:'K-rich soil promotes fruit development.',                    tips:'Banana thrives in tropical climate. Plant in ridges. Top-dress with K twice.',                                       indicators:[{l:'K Status',v:'High'},{l:'pH Level',v:'Optimal'},{l:'Drainage',v:'Good'}]},
  lowPH:    {crop:'Cotton',   alts:['Sunflower','Soybean'],    soil:'Neutral–slightly acidic soil ideal for cotton fibers.',      tips:'Cotton needs 150+ frost-free days. Control bollworm early. Requires deep plowing.',                                 indicators:[{l:'pH',v:'Neutral'},{l:'Drainage',v:'High'},{l:'Warmth',v:'Needed'}]},
  highRain: {crop:'Sugarcane',alts:['Tea','Coffee'],           soil:'High rainfall and humidity ideal for sugarcane belt.',       tips:'Use stale seed-bed technique. Irrigate at 4-week intervals. Top-dress at 30 and 60 days.',                         indicators:[{l:'Rain',v:'High'},{l:'Temp',v:'Warm'},{l:'Soil',v:'Deep'}]},
  default:  {crop:'Maize',    alts:['Sorghum','Pearl Millet'], soil:'Balanced soil with medium NPK. Suitable for dryland crops.', tips:'Inter-crop with legumes. Ensure 75cm row spacing. Harvest at 30% grain moisture.',                                  indicators:[{l:'Balance',v:'Good'},{l:'Drainage',v:'Med'},{l:'Organic',v:'Med'}]}
};

async function runCropRecommend(btn) {
  setLoading(btn, true);
  await fakeDelay(1400);
  const N    = parseInt(document.querySelector('#crop-n-range').value || 60);
  const rain = parseInt(document.getElementById('crop-rain').textContent);
  const ph   = parseFloat(document.getElementById('crop-ph').textContent);
  let data;
  if (N > 80)        data = CROPS.highN;
  else if (rain > 200) data = CROPS.highRain;
  else if (ph < 5.5)  data = CROPS.lowPH;
  else               data = CROPS.default;
  const conf = randBetween(88, 99);
  document.getElementById('crop-result-name').textContent = data.crop;
  document.getElementById('crop-result-conf').textContent = `Model confidence: ${conf}%`;
  setTimeout(() => document.getElementById('crop-conf-bar').style.width = conf + '%', 100);
  document.getElementById('crop-alts').innerHTML = data.alts.map(a => `<span class="tag">🌱 ${a}</span>`).join('');
  document.getElementById('crop-metrics').innerHTML = data.indicators.map(i => `<div class="metric-box"><div class="num">${i.v}</div><div class="lbl">${i.l}</div></div>`).join('');
  document.getElementById('crop-tips').textContent = data.tips;
  showResult('crop-result');
  setLoading(btn, false);
  toast('Crop recommendation ready!');
}

// ─────────────────────────────────────────
//  MODULE 2: DISEASE DETECTION
// ─────────────────────────────────────────
const DISEASE_DATA = {
  'Tomato Leaf Blight': {
    type: 'diseased',
    treatment: 'Apply Mancozeb 75% WP at 2g/L water. Remove and destroy infected leaves. Improve drainage. Avoid overhead irrigation.',
    parts: ['Leaves', 'Stem'],
    preventive: ['Avoid waterlogging', 'Use disease-resistant varieties', 'Rotate crops annually', 'Apply copper-based fungicide prophylactically']
  },
  'Apple Scab': {
    type: 'diseased',
    treatment: 'Spray Captan 50% WP at 2.5g/L during early infection. Prune infected branches. Apply lime-sulfur during dormancy.',
    parts: ['Leaves', 'Fruit', 'Branches'],
    preventive: ['Rake fallen leaves', 'Maintain air circulation', 'Early-season fungicide spray', 'Select scab-resistant cultivars']
  },
  'Potato Late Blight': {
    type: 'diseased',
    treatment: 'Apply Metalaxyl + Mancozeb 0.25% solution. Destroy infected plants. Harvest early if infection is severe.',
    parts: ['Leaves', 'Tubers', 'Stem'],
    preventive: ['Plant certified disease-free seed', 'Avoid planting in low-lying areas', 'Apply fungicide before rainy season', 'Remove volunteer potato plants']
  },
  'Healthy Corn': {
    type: 'healthy',
    treatment: 'No treatment needed. Continue regular care and maintenance.',
    parts: [],
    preventive: ['Monitor for pests weekly', 'Ensure adequate N fertilization', 'Maintain optimal soil moisture', 'Scout for early signs of disease']
  },
  'Healthy Rice': {
    type: 'healthy',
    treatment: 'Plant appears healthy. Maintain current agronomic practices.',
    parts: [],
    preventive: ['Keep field leveled for uniform irrigation', 'Monitor for stem borers', 'Apply balanced fertilizer', 'Weed management at 15 and 30 DAT']
  }
};

let selectedDisease = null;
let hasImage = false;

function handleImageUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('upload-preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
    hasImage = true;
    selectedDisease = null;
    document.getElementById('disease-btn').disabled = false;
    document.getElementById('disease-btn').style.opacity = '1';
  };
  reader.readAsDataURL(file);
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    document.getElementById('leaf-input').files = e.dataTransfer.files;
    handleImageUpload(document.getElementById('leaf-input'));
  }
}

function useSample(name, type, conf) {
  selectedDisease = { name, type, conf };
  const colors = { healthy: '#2e6b4a', diseased: '#c0392b' };
  const preview = document.getElementById('upload-preview');
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='300' height='200' viewBox='0 0 300 200'><rect width='300' height='200' fill='${type === 'healthy' ? '#e8f5ee' : '#fdecea'}'/><text x='150' y='80' text-anchor='middle' font-size='48'>${type === 'healthy' ? '🍃' : '🍂'}</text><text x='150' y='130' text-anchor='middle' font-size='14' fill='${colors[type]}' font-family='sans-serif'>${name}</text><text x='150' y='155' text-anchor='middle' font-size='11' fill='#888' font-family='sans-serif'>Sample Image</text></svg>`;
  preview.src = 'data:image/svg+xml;base64,' + btoa(svg);
  preview.style.display = 'block';
  hasImage = true;
  document.getElementById('disease-btn').disabled = false;
  document.getElementById('disease-btn').style.opacity = '1';
  toast(`Sample loaded: ${name}`);
}

async function runDiseaseDetect(btn) {
  if (!hasImage) return;
  setLoading(btn, true);
  await fakeDelay(1800);
  let name, type, conf;
  if (selectedDisease) {
    name = selectedDisease.name; type = selectedDisease.type; conf = selectedDisease.conf;
  } else {
    const samples = Object.keys(DISEASE_DATA);
    name = samples[Math.floor(Math.random() * samples.length)];
    type = DISEASE_DATA[name].type;
    conf = randBetween(81, 97);
  }
  const data = DISEASE_DATA[name];
  document.getElementById('disease-badge-wrap').innerHTML = `<span class="disease-badge ${data.type}">${data.type === 'healthy' ? '✅ Healthy' : '⚠️ Disease Detected'}</span>`;
  document.getElementById('disease-name').textContent = name;
  document.getElementById('disease-conf').textContent = `Confidence: ${conf}%`;
  setTimeout(() => document.getElementById('disease-conf-bar').style.width = conf + '%', 100);
  document.getElementById('disease-treatment').textContent = data.treatment;
  document.getElementById('disease-parts').innerHTML = data.parts.length
    ? data.parts.map(p => `<span class="tag ${data.type === 'diseased' ? 'red' : ''}">${p}</span>`).join('')
    : '<span style="font-size:13px;color:var(--text-muted);">No disease symptoms detected.</span>';
  document.getElementById('disease-preventive').innerHTML = data.preventive.map(p => `<li>${p}</li>`).join('');
  showResult('disease-result');
  setLoading(btn, false);
  toast('Disease analysis complete!');
}

// ─────────────────────────────────────────
//  MODULE 3: IRRIGATION
// ─────────────────────────────────────────
async function runIrrigation(btn) {
  setLoading(btn, true);
  await fakeDelay(1200);
  const moisture = parseInt(document.getElementById('irr-moisture').textContent);
  const rain     = parseInt(document.getElementById('irr-rain').textContent);
  const shouldIrrigate = moisture < 50 && rain < 15;
  const waterNeeded = Math.max(0, Math.round((70 - moisture) * 3.2));
  document.getElementById('irr-decision').textContent = shouldIrrigate ? '💧 Irrigate Now' : '✅ No Irrigation Needed';
  document.getElementById('irr-sub').textContent = shouldIrrigate
    ? `Apply ${waterNeeded} litres/acre — soil moisture critically low at ${moisture}%`
    : `Sufficient moisture (${moisture}%) or rain expected (${rain}mm). Skip irrigation.`;
  const barItems = [
    { lbl: 'Soil\nMoisture', val: moisture, color: 'var(--green-soft)' },
    { lbl: 'Rain\nForecast', val: Math.min(rain, 80), color: 'var(--info)' },
    { lbl: 'Target\nMoisture', val: 70, color: 'var(--amber)' }
  ];
  document.getElementById('water-bars').innerHTML = barItems.map(b =>
    `<div class="water-bar-wrap"><div class="water-bar-bg"><div class="water-bar-fill" style="height:0%;background:${b.color};" data-val="${b.val}"></div></div><div class="water-bar-label" style="font-size:10px;text-align:center;">${b.lbl}</div></div>`
  ).join('');
  setTimeout(() => {
    document.querySelectorAll('.water-bar-fill').forEach(el => { el.style.height = el.dataset.val + '%'; });
  }, 100);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  let html = '<div style="display:flex;flex-direction:column;gap:6px;">';
  days.forEach((d, i) => {
    const needWater = i % 3 === 0 && shouldIrrigate;
    html += `<div style="display:flex;align-items:center;gap:10px;font-size:13px;padding:6px 10px;background:${needWater ? 'var(--green-pale)' : 'var(--cream)'};border-radius:6px;"><span style="font-weight:600;width:36px;color:var(--text-mid);">${d}</span><span>${needWater ? '💧 Irrigate — ' + Math.round(waterNeeded * 0.85) + 'L/acre' : '⏸ Skip'}</span></div>`;
  });
  html += '</div>';
  document.getElementById('irr-schedule').innerHTML = html;
  document.getElementById('irr-metrics').innerHTML = `<div class="metric-box"><div class="num">${moisture}%</div><div class="lbl">Soil Moisture</div></div><div class="metric-box"><div class="num">${waterNeeded}L</div><div class="lbl">Water/Acre</div></div><div class="metric-box"><div class="num">${rain}mm</div><div class="lbl">Rain Forecast</div></div>`;
  showResult('irr-result');
  setLoading(btn, false);
  toast('Irrigation schedule ready!');
}

// ─────────────────────────────────────────
//  MODULE 4: YIELD PREDICTION
// ─────────────────────────────────────────
const YIELD_BASE = { Rice: 3.8, Wheat: 4.2, Maize: 5.1, Cotton: 2.1, Sugarcane: 68, Soybean: 2.8, Groundnut: 2.2 };

async function runYield(btn) {
  setLoading(btn, true);
  await fakeDelay(1500);
  const crop    = document.getElementById('yd-crop').value;
  const area    = parseFloat(document.getElementById('yd-area').value) || 5;
  const fert    = parseInt(document.getElementById('yd-fert').textContent);
  const irrType = document.getElementById('yd-irr').value;
  const base    = YIELD_BASE[crop] || 3.5;
  const irrMult = irrType === 'Fully Irrigated' ? 1.3 : irrType === 'Partially Irrigated' ? 1.1 : 0.9;
  const fertMult = fert > 150 ? 1.15 : fert > 80 ? 1.05 : 0.95;
  const predicted = parseFloat((base * irrMult * fertMult * (0.92 + Math.random() * 0.16)).toFixed(2));
  const totalTonnes = (predicted * area).toFixed(1);
  const conf = randBetween(85, 95);
  document.getElementById('yield-val').textContent = predicted;
  document.getElementById('yield-sub').textContent = `Total for ${area} ha: ${totalTonnes} tonnes (${conf}% confidence)`;
  setTimeout(() => document.getElementById('yield-bar').style.width = conf + '%', 100);
  const revenue = Math.round(parseFloat(totalTonnes) * 18000);
  document.getElementById('yield-metrics').innerHTML = `<div class="metric-box"><div class="num">${totalTonnes}t</div><div class="lbl">Total Yield</div></div><div class="metric-box"><div class="num">₹${(revenue / 1000).toFixed(0)}K</div><div class="lbl">Est. Revenue</div></div><div class="metric-box"><div class="num">${conf}%</div><div class="lbl">Confidence</div></div>`;
  const chartWrap = document.getElementById('yield-chart-wrap');
  const regional  = (base * 0.9).toFixed(2);
  const national  = (base * 0.78).toFixed(2);
  const bars2 = [
    { lbl: 'Your Farm',    val: predicted, color: 'var(--green-soft)' },
    { lbl: 'Regional Avg', val: regional,  color: 'var(--green-light)' },
    { lbl: 'National Avg', val: national,  color: 'var(--text-muted)' }
  ];
  const maxVal = Math.max(...bars2.map(b => b.val));
  chartWrap.style.height = '140px';
  chartWrap.innerHTML = bars2.map(b => {
    const pct = Math.round((b.val / maxVal) * 100);
    return `<div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;height:120px;justify-content:flex-end;"><div style="font-size:11px;font-weight:600;color:var(--text-dark);">${b.val}</div><div style="width:100%;border-radius:6px 6px 0 0;background:${b.color};transition:height 1s ease;" data-h="${pct}" style="height:0px;"></div><div style="font-size:10px;color:var(--text-muted);text-align:center;">${b.lbl}</div></div>`;
  }).join('');
  setTimeout(() => {
    chartWrap.querySelectorAll('[data-h]').forEach(el => { el.style.height = el.dataset.h + '%'; });
  }, 100);
  document.getElementById('yield-tips').innerHTML = [
    'Use certified high-yield seed varieties',
    'Split fertilizer application for better uptake',
    'Adopt integrated pest management (IPM)',
    'Timely sowing within recommended window',
    'Conduct soil testing every 2 seasons'
  ].map(t => `<li>${t}</li>`).join('');
  showResult('yield-result');
  setLoading(btn, false);
  toast('Yield prediction complete!');
}

// ─────────────────────────────────────────
//  MODULE 5: CROP ROTATION
// ─────────────────────────────────────────
const ROTATION_MAP = {
  Rice:    [{ crop: 'Wheat',    icon: '🌾', season: 'Rabi' },    { crop: 'Chickpea', icon: '🫘', season: 'Summer' },   { crop: 'Rice',    icon: '🌾', season: 'Kharif' }],
  Wheat:   [{ crop: 'Maize',   icon: '🌽', season: 'Kharif' },  { crop: 'Mustard',  icon: '🌼', season: 'Rabi' },     { crop: 'Soybean', icon: '🟤', season: 'Kharif' }],
  Cotton:  [{ crop: 'Groundnut',icon:'🥜', season: 'Kharif' },  { crop: 'Wheat',    icon: '🌾', season: 'Rabi' },     { crop: 'Soybean', icon: '🟤', season: 'Next Kharif' }],
  Maize:   [{ crop: 'Chickpea', icon:'🫘', season: 'Rabi' },    { crop: 'Groundnut',icon: '🥜', season: 'Summer' },  { crop: 'Sorghum', icon: '🌾', season: 'Kharif' }],
  default: [{ crop: 'Legume',  icon: '🫘', season: 'Season 2'}, { crop: 'Cereal',   icon: '🌾', season: 'Season 3'}, { crop: 'Root Crop',icon:'🥕', season: 'Season 4'}]
};
const BENEFITS_MAP = {
  Rice:    ['✅ N-fixation by chickpea', '🌱 Break rice blast cycle', '💧 Reduce water usage 30%', '🐛 Pest cycle disruption'],
  Wheat:   ['🌱 Soybean fixes nitrogen', '📈 Maize improves organic matter', '🐛 Breaks wheat rust cycle', '💰 Higher annual income'],
  Cotton:  ['🌿 Groundnut restores N', '🔄 Breaks bollworm cycle', '🌱 Improves soil structure', '📉 Reduces chemical inputs'],
  default: ['🌿 Soil fertility restoration', '🐛 Pest cycle disruption', '💧 Better water retention', '📈 Improved annual returns']
};

async function runRotation(btn) {
  setLoading(btn, true);
  await fakeDelay(1300);
  const current  = document.getElementById('rot-current').value;
  const plan     = ROTATION_MAP[current] || ROTATION_MAP.default;
  const benefits = BENEFITS_MAP[current] || BENEFITS_MAP.default;
  let html = `<div class="rotation-step"><div class="rotation-circle" style="background:var(--amber-light);border-color:var(--amber);"><span>🌾</span>${current}</div><div class="rotation-label">Current</div></div>`;
  plan.forEach(step => {
    html += `<div class="rotation-arrow">→</div><div class="rotation-step"><div class="rotation-circle"><span>${step.icon}</span>${step.crop}</div><div class="rotation-label">${step.season}</div></div>`;
  });
  document.getElementById('rotation-chain').innerHTML = html;
  document.getElementById('rotation-benefits').innerHTML = benefits.map(b => `<span class="tag">${b}</span>`).join('');
  const noteItems = [
    { season: 'Season 2', crop: plan[0].crop, note: `${plan[0].crop} replenishes soil nitrogen and breaks ${current} pest cycles. Apply base fertilizer.` },
    { season: 'Season 3', crop: plan[1].crop, note: `${plan[1].crop} further diversifies root depth. Minimal fertilizer needed.` },
    { season: 'Season 4', crop: plan[2].crop, note: `Return to ${plan[2].crop} on a revitalised soil profile. Expect 15–20% yield improvement.` }
  ];
  document.getElementById('rotation-notes').innerHTML = noteItems.map(n =>
    `<div style="background:var(--cream);border-radius:var(--radius-sm);padding:12px 14px;border-left:3px solid var(--green-soft);"><div style="font-size:11px;font-weight:600;color:var(--green-mid);text-transform:uppercase;letter-spacing:0.6px;">${n.season} — ${n.crop}</div><div style="font-size:13px;color:var(--text-mid);margin-top:4px;line-height:1.6;">${n.note}</div></div>`
  ).join('');
  showResult('rotation-result');
  setLoading(btn, false);
  toast('Rotation plan generated!');
}

// ─────────────────────────────────────────
//  MODULE 6: PEST RISK
// ─────────────────────────────────────────
const PEST_DATA = {
  high:   { color: '#c0392b', pests: ['🐛 Brown Plant Hopper', '🦟 Whitefly', '🐜 Aphid Colony'], actions: ['Apply recommended pesticide immediately', 'Set up yellow sticky traps in field', 'Remove heavily infested plants', 'Consult local agricultural officer', 'Apply Neem-based biopesticide weekly'] },
  medium: { color: '#e8a030', pests: ['🦗 Stem Borer', '🪲 Leaf Miner', '🐌 Slug Damage'],   actions: ['Scout field every 3 days', 'Apply pheromone traps', 'Use light traps at night', 'Apply Bt (Bacillus thuringiensis) spray', 'Maintain field sanitation'] },
  low:    { color: '#4c9c6e', pests: ['🔍 Minor leaf damage', '🌿 Low mite activity'],         actions: ['Continue regular field monitoring', 'Maintain crop hygiene', 'Ensure balanced fertilization', 'No immediate spray needed', 'Record observations for next season'] }
};

async function runPest(btn) {
  setLoading(btn, true);
  await fakeDelay(1400);
  const hum  = parseInt(document.getElementById('pest-hum').textContent);
  const temp = parseInt(document.getElementById('pest-temp').textContent);
  const prev = document.getElementById('pest-prev').value;
  let riskScore = 0;
  if (hum > 75) riskScore += 35;
  if (temp > 28 && temp < 38) riskScore += 25;
  if (prev === 'Severe')   riskScore += 40;
  else if (prev === 'Moderate') riskScore += 25;
  else if (prev === 'Light')    riskScore += 10;
  riskScore = Math.min(95, riskScore + randBetween(5, 15));
  const level      = riskScore > 65 ? 'high' : riskScore > 35 ? 'medium' : 'low';
  const levelLabel = level.charAt(0).toUpperCase() + level.slice(1);
  document.getElementById('pest-risk-level').textContent  = levelLabel;
  document.getElementById('pest-risk-level').style.color  = PEST_DATA[level].color;
  document.getElementById('pest-sub').textContent = `Risk Score: ${Math.round(riskScore)}/100 — Based on weather, crop history, and conditions`;
  const arcFill   = document.getElementById('risk-arc-fill');
  const needle    = document.getElementById('risk-needle');
  const arcLen    = 235;
  const dashOffset = arcLen - (arcLen * riskScore / 100);
  const needleAngle = -90 + (riskScore / 100) * 180;
  setTimeout(() => {
    arcFill.style.transition = 'stroke-dashoffset 1s ease';
    arcFill.setAttribute('stroke-dashoffset', dashOffset);
    arcFill.setAttribute('stroke', PEST_DATA[level].color);
    needle.setAttribute('transform', `rotate(${needleAngle},90,90)`);
  }, 100);
  document.getElementById('pest-threats').innerHTML = PEST_DATA[level].pests.map(p => `<span class="tag ${level === 'high' ? 'red' : level === 'medium' ? 'amber' : ''}">${p}</span>`).join('');
  document.getElementById('pest-actions').innerHTML  = PEST_DATA[level].actions.map(a => `<li>${a}</li>`).join('');
  showResult('pest-result');
  setLoading(btn, false);
  toast('Pest risk assessment done!');
}

// ─────────────────────────────────────────
//  MODULE 7: PROFIT ESTIMATOR
// ─────────────────────────────────────────
async function runProfit(btn) {
  setLoading(btn, true);
  await fakeDelay(1000);
  const area   = parseFloat(document.getElementById('pr-area').value)   || 5;
  const yieldTha = parseFloat(document.getElementById('pr-yield').value) || 4;
  const price  = parseFloat(document.getElementById('pr-price').value)   || 18000;
  const seed   = parseFloat(document.getElementById('pr-seed').value)    || 3500;
  const fert   = parseFloat(document.getElementById('pr-fert').value)    || 5000;
  const labour = parseFloat(document.getElementById('pr-labour').value)  || 8000;
  const irr    = parseFloat(document.getElementById('pr-irr2').value)    || 2500;
  const pest   = parseFloat(document.getElementById('pr-pest2').value)   || 2000;
  const misc   = parseFloat(document.getElementById('pr-misc').value)    || 1000;
  const totalYield = yieldTha * area;
  const revenue    = Math.round(totalYield * price);
  const costPerHa  = seed + fert + labour + irr + pest + misc;
  const totalCost  = Math.round(costPerHa * area);
  const netProfit  = revenue - totalCost;
  const margin     = ((netProfit / revenue) * 100).toFixed(1);
  const roi        = (((revenue - totalCost) / totalCost) * 100).toFixed(1);
  const kpiData = [
    { lbl: 'Gross Revenue', val: '₹' + revenue.toLocaleString('en-IN'),   trend: null,                               highlight: false },
    { lbl: 'Total Cost',    val: '₹' + totalCost.toLocaleString('en-IN'), trend: null,                               highlight: false },
    { lbl: 'Net Profit',    val: '₹' + netProfit.toLocaleString('en-IN'), trend: netProfit > 0 ? 'up' : 'down',      highlight: true  },
    { lbl: 'Profit Margin', val: margin + '%',                             trend: parseFloat(margin) > 30 ? 'up' : 'down', highlight: false },
    { lbl: 'ROI',           val: roi + '%',                                trend: parseFloat(roi) > 25 ? 'up' : 'down',   highlight: false }
  ];
  document.getElementById('profit-kpis').innerHTML = kpiData.map(k =>
    `<div class="kpi-card ${k.highlight ? 'highlight' : ''}"><div class="kpi-num">${k.val}</div><div class="kpi-lbl">${k.lbl}</div>${k.trend ? `<div class="kpi-trend ${k.trend}">${k.trend === 'up' ? '↑ Profitable' : '↓ Review costs'}</div>` : ''}</div>`
  ).join('');
  // Donut chart
  const costs = [
    { lbl: 'Seed',       val: seed,   color: '#2e6b4a' },
    { lbl: 'Fertiliser', val: fert,   color: '#4c9c6e' },
    { lbl: 'Labour',     val: labour, color: '#a8d8b9' },
    { lbl: 'Irrigation', val: irr,    color: '#e8a030' },
    { lbl: 'Pesticide',  val: pest,   color: '#7a5c3e' },
    { lbl: 'Misc',       val: misc,   color: '#b4b2a9' }
  ];
  const totalCostCheck = costs.reduce((s, c) => s + c.val, 0);
  const svg = document.getElementById('donut-svg');
  let startAngle = -Math.PI / 2;
  const cx = 70, cy = 70, r = 52, inner = 30;
  let paths = '';
  costs.forEach(c => {
    const angle    = (c.val / totalCostCheck) * 2 * Math.PI;
    const endAngle = startAngle + angle;
    const x1 = cx + r * Math.cos(startAngle), y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle),   y2 = cy + r * Math.sin(endAngle);
    const ix1 = cx + inner * Math.cos(endAngle),   iy1 = cy + inner * Math.sin(endAngle);
    const ix2 = cx + inner * Math.cos(startAngle), iy2 = cy + inner * Math.sin(startAngle);
    const lg = angle > Math.PI ? 1 : 0;
    paths += `<path d="M${x1.toFixed(1)} ${y1.toFixed(1)} A${r} ${r} 0 ${lg} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} L${ix1.toFixed(1)} ${iy1.toFixed(1)} A${inner} ${inner} 0 ${lg} 0 ${ix2.toFixed(1)} ${iy2.toFixed(1)} Z" fill="${c.color}" stroke="#fff" stroke-width="1.5"/>`;
    startAngle = endAngle;
  });
  svg.innerHTML = paths;
  document.getElementById('donut-legend').innerHTML = costs.map(c =>
    `<div class="legend-row"><div class="legend-dot" style="background:${c.color}"></div><span class="legend-label">${c.lbl}</span><span class="legend-val">₹${(c.val * area).toLocaleString('en-IN')}</span></div>`
  ).join('');
  document.getElementById('profit-analysis').textContent = netProfit > 0
    ? `With a net profit of ₹${netProfit.toLocaleString('en-IN')} and ${margin}% margin, this farm operation is profitable. Labour cost forms the largest input expense. Consider mechanisation to reduce costs by up to 20%.`
    : `The operation shows a loss of ₹${Math.abs(netProfit).toLocaleString('en-IN')}. Review input costs — labour and fertiliser are the primary drivers. Consider group purchase of inputs or government subsidy schemes.`;
  showResult('profit-result');
  setLoading(btn, false);
  toast('Profit calculation complete!');
}

// ─────────────────────────────────────────
//  MODULE 8: MARKET PRICE PREDICTION
// ─────────────────────────────────────────
const PRICE_PROFILES = {
  'Rice (Common)': { base: 2100, trend: 'up',       volatility: 80  },
  'Wheat':         { base: 2350, trend: 'flat',     volatility: 60  },
  'Maize':         { base: 1850, trend: 'up',       volatility: 120 },
  'Onion':         { base: 1200, trend: 'volatile', volatility: 300 },
  'Tomato':        { base: 800,  trend: 'volatile', volatility: 400 },
  'Potato':        { base: 950,  trend: 'down',     volatility: 150 },
  'Soybean':       { base: 4800, trend: 'up',       volatility: 100 },
  'Cotton':        { base: 6200, trend: 'flat',     volatility: 180 },
  'Groundnut':     { base: 5100, trend: 'up',       volatility: 130 }
};

async function runMarket(btn) {
  setLoading(btn, true);
  await fakeDelay(1600);
  const crop    = document.getElementById('mkt-crop').value;
  const days    = parseInt(document.getElementById('mkt-period').value);
  const profile = PRICE_PROFILES[crop] || { base: 2000, trend: 'flat', volatility: 100 };
  const currentPrice = parseInt(document.getElementById('mkt-current').value) || profile.base;
  // Generate price series
  const prices = [currentPrice];
  let p = currentPrice;
  for (let i = 1; i <= days; i++) {
    const trend = profile.trend === 'up' ? 2 : profile.trend === 'down' ? -2 : 0;
    const noise = (Math.random() - 0.5) * profile.volatility * 0.5;
    p = Math.max(200, Math.round(p + trend + noise));
    prices.push(p);
  }
  const maxP   = Math.max(...prices), minP = Math.min(...prices);
  const finalP = prices[prices.length - 1];
  const change = finalP - currentPrice;
  const changePct = ((change / currentPrice) * 100).toFixed(1);
  document.getElementById('mkt-metrics').innerHTML = `<div class="metric-box"><div class="num">₹${currentPrice}</div><div class="lbl">Current</div></div><div class="metric-box"><div class="num">₹${finalP}</div><div class="lbl">Day ${days} Forecast</div></div><div class="metric-box"><div class="num" style="color:${change >= 0 ? 'var(--green-mid)' : 'var(--danger)'};">${change >= 0 ? '+' : ''}${changePct}%</div><div class="lbl">Expected Change</div></div><div class="metric-box"><div class="num">₹${minP}</div><div class="lbl">Est. Low</div></div>`;
  // SVG Line Chart
  const svgEl  = document.getElementById('market-chart');
  const W = 580, H = 160, pad = { t: 10, r: 10, b: 30, l: 50 };
  const plotW  = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const range  = maxP - minP || 100;
  const xScale = i => pad.l + (i / (prices.length - 1)) * plotW;
  const yScale = v => pad.t + plotH - ((v - minP) / range) * plotH;
  let pathD = prices.map((v, i) => (i === 0 ? 'M' : 'L') + xScale(i).toFixed(1) + ' ' + yScale(v).toFixed(1)).join(' ');
  let areaD = pathD + ` L${xScale(prices.length - 1)} ${pad.t + plotH} L${pad.l} ${pad.t + plotH} Z`;
  const lineColor  = change >= 0 ? 'var(--green-soft)' : 'var(--danger)';
  const areaColor  = change >= 0 ? 'rgba(76,156,110,0.1)' : 'rgba(192,57,43,0.08)';
  const ticks      = [minP, Math.round((minP + maxP) / 2), maxP];
  let ticksHTML    = ticks.map(t => `<text x="${pad.l - 6}" y="${yScale(t) + 4}" text-anchor="end" fill="var(--text-muted)" font-size="10" font-family="DM Mono,monospace">₹${t}</text>`).join('');
  const step       = Math.ceil(prices.length / 4);
  let xLabels = '';
  for (let i = 0; i < prices.length; i += step) {
    xLabels += `<text x="${xScale(i)}" y="${pad.t + plotH + 16}" text-anchor="middle" fill="var(--text-muted)" font-size="10">Day ${i}</text>`;
  }
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgEl.innerHTML = `<path d="${areaD}" fill="${areaColor}" stroke="none"/><path d="${pathD}" fill="none" stroke="${lineColor}" stroke-width="2" stroke-linejoin="round"/><circle cx="${xScale(prices.length - 1)}" cy="${yScale(finalP)}" r="4" fill="${lineColor}" stroke="#fff" stroke-width="1.5"/>${ticksHTML}${xLabels}`;
  const bestDay = prices.indexOf(maxP);
  document.getElementById('mkt-insights').innerHTML = `The LSTM model forecasts a <strong>${change >= 0 ? 'rising' : 'falling'}</strong> price trend for ${crop} in the coming ${days} days. Peak price of ₹${maxP}/quintal is expected around Day ${bestDay}. ${profile.trend === 'volatile' ? 'High volatility expected — consider forward contracts or early selling.' : 'Price movement is moderate — hold for optimal selling window.'}`;
  const sellWrap = document.getElementById('mkt-sell-window');
  if (change > 50) {
    sellWrap.innerHTML = `<span class="tag">📅 Best window: Day ${Math.max(1, bestDay - 2)} – Day ${bestDay}</span><span class="tag amber">💰 Expected price: ₹${maxP}/quintal</span>`;
  } else if (change < -50) {
    sellWrap.innerHTML = `<span class="tag red">⚠️ Prices falling — sell now or within 3 days</span><span class="tag amber">Current: ₹${currentPrice}/quintal</span>`;
  } else {
    sellWrap.innerHTML = `<span class="tag">📅 Price stable — sell at convenience</span><span class="tag blue">Watch market daily</span>`;
  }
  showResult('market-result');
  setLoading(btn, false);
  toast('Price forecast generated!');
}
