// ═══════════════════════════════════════════════════════════════════
//  AgroSense — API Connector
//  Replaces the fake fakeDelay() calls in app.js with real Flask API
//  calls. Drop this file AFTER app.js in base.html to override the
//  simulation functions with live model predictions.
//
//  Usage in base.html (add after app.js):
//    <script src="{{ url_for('static', filename='js/api.js') }}"></script>
//
//  Each function matches the signature expected by the HTML buttons.
// ═══════════════════════════════════════════════════════════════════

const API_BASE = '/api';   // Flask API prefix — change if using a separate host

// ─────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────

/**
 * POST JSON to the Flask API and return parsed response.
 * Shows a user-friendly toast on network/server errors.
 */
async function apiPost(endpoint, payload) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * POST multipart/form-data (used for image upload in disease detection).
 */
async function apiPostForm(endpoint, formData) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body:   formData,     // browser sets Content-Type with boundary automatically
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function apiError(btn, msg) {
  setLoading(btn, false);
  const w = document.getElementById('toastWrap');
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.background = 'var(--danger)';
  t.textContent = '❌ ' + msg;
  w.appendChild(t);
  setTimeout(() => t.remove(), 5000);
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 1: CROP RECOMMENDATION  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runCropRecommend(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      N:           parseFloat(document.getElementById('crop-n-range').value),
      P:           parseFloat(document.getElementById('crop-p-range').value),
      K:           parseFloat(document.getElementById('crop-k-range').value),
      temperature: parseFloat(document.getElementById('crop-temp-range').value),
      humidity:    parseFloat(document.getElementById('crop-hum-range').value),
      ph:          parseFloat(document.getElementById('crop-ph').textContent),
      rainfall:    parseFloat(document.getElementById('crop-rain').textContent),
    };

    const data = await apiPost('/crop/recommend', payload);

    document.getElementById('crop-result-name').textContent = data.crop;
    document.getElementById('crop-result-conf').textContent =
      `Model confidence: ${data.confidence}%`;
    setTimeout(() => {
      document.getElementById('crop-conf-bar').style.width = data.confidence + '%';
    }, 100);

    document.getElementById('crop-alts').innerHTML =
      (data.alternatives || []).map(a => `<span class="tag">🌱 ${a}</span>`).join('');

    document.getElementById('crop-metrics').innerHTML =
      (data.indicators || []).map(i =>
        `<div class="metric-box"><div class="num">${i.v}</div><div class="lbl">${i.l}</div></div>`
      ).join('');

    document.getElementById('crop-tips').textContent = data.tips || '';
    showResult('crop-result');
    toast('Crop recommendation ready!');
  } catch (e) {
    apiError(btn, `Crop Rec: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 2: DISEASE DETECTION  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runDiseaseDetect(btn) {
  if (!hasImage) return;
  setLoading(btn, true);
  try {
    let data;

    if (selectedDisease) {
      // Sample buttons — still call API with the synthetic crop name
      data = await apiPost('/disease/predict-sample', {
        name: selectedDisease.name,
        type: selectedDisease.type,
        conf: selectedDisease.conf,
      });
    } else {
      // Real image upload
      const fileInput = document.getElementById('leaf-input');
      const form = new FormData();
      form.append('image', fileInput.files[0]);
      data = await apiPostForm('/disease/predict', form);
    }

    document.getElementById('disease-badge-wrap').innerHTML =
      `<span class="disease-badge ${data.type}">${
        data.type === 'healthy' ? '✅ Healthy' : '⚠️ Disease Detected'
      }</span>`;
    document.getElementById('disease-name').textContent    = data.name;
    document.getElementById('disease-conf').textContent    = `Confidence: ${data.confidence}%`;
    setTimeout(() => {
      document.getElementById('disease-conf-bar').style.width = data.confidence + '%';
    }, 100);
    document.getElementById('disease-treatment').textContent = data.treatment;
    document.getElementById('disease-parts').innerHTML =
      (data.parts || []).length
        ? data.parts.map(p =>
            `<span class="tag ${data.type === 'diseased' ? 'red' : ''}">${p}</span>`
          ).join('')
        : '<span style="font-size:13px;color:var(--text-muted);">No disease symptoms detected.</span>';
    document.getElementById('disease-preventive').innerHTML =
      (data.preventive || []).map(p => `<li>${p}</li>`).join('');

    showResult('disease-result');
    toast('Disease analysis complete!');
  } catch (e) {
    apiError(btn, `Disease: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 3: SMART IRRIGATION  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runIrrigation(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      crop:               document.getElementById('irr-crop').value,
      soil_moisture:      parseFloat(document.getElementById('irr-moisture-range')?.value
                            ?? document.getElementById('irr-moisture')?.textContent ?? 35),
      stage:              document.getElementById('irr-stage').value,
      temperature:        parseFloat(document.getElementById('irr-temp-range')?.value
                            ?? document.getElementById('irr-temp')?.textContent ?? 32),
      humidity:           parseFloat(document.getElementById('irr-hum-range')?.value
                            ?? document.getElementById('irr-hum')?.textContent  ?? 55),
      rainfall_forecast:  parseFloat(document.getElementById('irr-rain-range')?.value
                            ?? document.getElementById('irr-rain')?.textContent ?? 5),
    };


    const data = await apiPost('/irrigation/plan', payload);

    document.getElementById('irr-decision').textContent = data.decision;
    document.getElementById('irr-sub').textContent      = data.summary;

    // Water bars
    const barItems = [
      { lbl: 'Soil\nMoisture', val: payload.soil_moisture,             color: 'var(--green-soft)' },
      { lbl: 'Rain\nForecast', val: Math.min(payload.rainfall_forecast, 80), color: 'var(--info)' },
      { lbl: 'Target\nMoisture', val: 70,                              color: 'var(--amber)' },
    ];
    document.getElementById('water-bars').innerHTML = barItems.map(b =>
      `<div class="water-bar-wrap"><div class="water-bar-bg"><div class="water-bar-fill"
        style="height:0%;background:${b.color};" data-val="${b.val}"></div></div>
       <div class="water-bar-label" style="font-size:10px;text-align:center;">${b.lbl}</div></div>`
    ).join('');
    setTimeout(() => {
      document.querySelectorAll('.water-bar-fill').forEach(el => {
        el.style.height = el.dataset.val + '%';
      });
    }, 100);

    document.getElementById('irr-metrics').innerHTML =
      (data.metrics || []).map(m =>
        `<div class="metric-box"><div class="num">${m.val}</div><div class="lbl">${m.lbl}</div></div>`
      ).join('');

    let html = '<div style="display:flex;flex-direction:column;gap:6px;">';
    (data.schedule || []).forEach(d => {
      html += `<div style="display:flex;align-items:center;gap:10px;font-size:13px;
        padding:6px 10px;background:${d.irrigate ? 'var(--green-pale)' : 'var(--cream)'};
        border-radius:6px;">
        <span style="font-weight:600;width:36px;color:var(--text-mid);">${d.day}</span>
        <span>${d.irrigate ? '💧 Irrigate — ' + d.amount + 'L/acre' : '⏸ Skip'}</span></div>`;
    });
    html += '</div>';
    document.getElementById('irr-schedule').innerHTML = html;

    showResult('irr-result');
    toast('Irrigation schedule ready!');
  } catch (e) {
    apiError(btn, `Irrigation: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 4: YIELD PREDICTION  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runYield(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      crop:        document.getElementById('yd-crop').value,
      season:      document.getElementById('yd-season').value,
      state:       document.getElementById('yd-state').value,
      area:        parseFloat(document.getElementById('yd-area').value) || 5,
      rainfall:    parseFloat(document.getElementById('yd-rain').textContent),
      fertiliser:  parseFloat(document.getElementById('yd-fert').textContent),
      irrigation:  document.getElementById('yd-irr').value,
      pesticide:   document.getElementById('yd-pest').value,
    };

    const data = await apiPost('/yield/predict', payload);

    document.getElementById('yield-val').textContent = data.yield_per_ha;
    document.getElementById('yield-sub').textContent = data.summary;
    setTimeout(() => {
      document.getElementById('yield-bar').style.width = data.confidence + '%';
    }, 100);

    document.getElementById('yield-metrics').innerHTML =
      (data.metrics || []).map(m =>
        `<div class="metric-box"><div class="num">${m.val}</div><div class="lbl">${m.lbl}</div></div>`
      ).join('');

    // Bar chart
    const chartWrap = document.getElementById('yield-chart-wrap');
    const bars = data.comparison || [];
    const maxVal = Math.max(...bars.map(b => parseFloat(b.val)));
    chartWrap.style.height = '140px';
    chartWrap.innerHTML = bars.map(b => {
      const pct = Math.round((parseFloat(b.val) / maxVal) * 100);
      return `<div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;height:120px;justify-content:flex-end;">
        <div style="font-size:11px;font-weight:600;color:var(--text-dark);">${b.val}</div>
        <div style="width:100%;border-radius:6px 6px 0 0;background:${b.color};transition:height 1s ease;"
             data-h="${pct}" style="height:0px;"></div>
        <div style="font-size:10px;color:var(--text-muted);text-align:center;">${b.lbl}</div></div>`;
    }).join('');
    setTimeout(() => {
      chartWrap.querySelectorAll('[data-h]').forEach(el => {
        el.style.height = el.dataset.h + '%';
      });
    }, 100);

    document.getElementById('yield-tips').innerHTML =
      (data.tips || []).map(t => `<li>${t}</li>`).join('');

    showResult('yield-result');
    toast('Yield prediction complete!');
  } catch (e) {
    apiError(btn, `Yield: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 5: CROP ROTATION  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runRotation(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      current_crop: document.getElementById('rot-current').value,
      soil_type:    document.getElementById('rot-soil').value,
      region:       document.getElementById('rot-region').value,
      n_level:      document.getElementById('rot-n').value,
      pest_history: document.getElementById('rot-pest-hist').value,
    };

    const data = await apiPost('/rotation/recommend', payload);

    // Rotation chain
    let html = `<div class="rotation-step">
      <div class="rotation-circle" style="background:var(--amber-light);border-color:var(--amber);">
        <span>🌾</span>${payload.current_crop}
      </div><div class="rotation-label">Current</div></div>`;
    (data.plan || []).forEach(step => {
      html += `<div class="rotation-arrow">→</div>
        <div class="rotation-step">
          <div class="rotation-circle"><span>${step.icon}</span>${step.crop}</div>
          <div class="rotation-label">${step.season}</div>
        </div>`;
    });
    document.getElementById('rotation-chain').innerHTML = html;
    document.getElementById('rotation-benefits').innerHTML =
      (data.benefits || []).map(b => `<span class="tag">${b}</span>`).join('');
    document.getElementById('rotation-notes').innerHTML =
      (data.notes || []).map(n =>
        `<div style="background:var(--cream);border-radius:var(--radius-sm);
          padding:12px 14px;border-left:3px solid var(--green-soft);">
          <div style="font-size:11px;font-weight:600;color:var(--green-mid);
            text-transform:uppercase;letter-spacing:0.6px;">${n.season} — ${n.crop}</div>
          <div style="font-size:13px;color:var(--text-mid);margin-top:4px;line-height:1.6;">${n.note}</div>
         </div>`
      ).join('');

    showResult('rotation-result');
    toast('Rotation plan generated!');
  } catch (e) {
    apiError(btn, `Rotation: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 6: PEST RISK  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runPest(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      crop:           document.getElementById('pest-crop').value,
      season:         document.getElementById('pest-season').value,
      temperature:    parseFloat(document.getElementById('pest-temp').textContent),
      humidity:       parseFloat(document.getElementById('pest-hum').textContent),
      prev_occurrence:document.getElementById('pest-prev').value,
      crop_density:   document.getElementById('pest-density').value,
      near_water:     document.getElementById('pest-water').value,
    };

    const data = await apiPost('/pest/assess', payload);

    const LEVEL_COLOR = { Low: '#4c9c6e', Medium: '#e8a030', High: '#c0392b' };
    const color = LEVEL_COLOR[data.level] || '#4c9c6e';

    document.getElementById('pest-risk-level').textContent  = data.level;
    document.getElementById('pest-risk-level').style.color  = color;
    document.getElementById('pest-sub').textContent =
      `Risk Score: ${data.score}/100 — Based on weather, crop history, and conditions`;

    // Animate SVG gauge
    const arcFill    = document.getElementById('risk-arc-fill');
    const needle     = document.getElementById('risk-needle');
    const dashOffset = 235 - (235 * data.score / 100);
    const needleAngle = -90 + (data.score / 100) * 180;
    setTimeout(() => {
      arcFill.style.transition = 'stroke-dashoffset 1s ease';
      arcFill.setAttribute('stroke-dashoffset', dashOffset);
      arcFill.setAttribute('stroke', color);
      needle.setAttribute('transform', `rotate(${needleAngle},90,90)`);
    }, 100);

    const tagClass = data.level === 'High' ? 'red' : data.level === 'Medium' ? 'amber' : '';
    document.getElementById('pest-threats').innerHTML =
      (data.threats || []).map(p => `<span class="tag ${tagClass}">${p}</span>`).join('');
    document.getElementById('pest-actions').innerHTML =
      (data.actions || []).map(a => `<li>${a}</li>`).join('');

    showResult('pest-result');
    toast('Pest risk assessment done!');
  } catch (e) {
    apiError(btn, `Pest Risk: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 7: PROFIT ESTIMATOR  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runProfit(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      crop:             document.getElementById('pr-crop').value,
      area:             parseFloat(document.getElementById('pr-area').value)   || 5,
      yield_per_ha:     parseFloat(document.getElementById('pr-yield').value)  || 4,
      selling_price:    parseFloat(document.getElementById('pr-price').value)  || 18000,
      seed_cost:        parseFloat(document.getElementById('pr-seed').value)   || 3500,
      fertiliser_cost:  parseFloat(document.getElementById('pr-fert').value)   || 5000,
      labour_cost:      parseFloat(document.getElementById('pr-labour').value) || 8000,
      irrigation_cost:  parseFloat(document.getElementById('pr-irr2').value)   || 2500,
      pesticide_cost:   parseFloat(document.getElementById('pr-pest2').value)  || 2000,
      misc_cost:        parseFloat(document.getElementById('pr-misc').value)   || 1000,
    };

    const data = await apiPost('/profit/calculate', payload);

    document.getElementById('profit-kpis').innerHTML =
      (data.kpis || []).map(k =>
        `<div class="kpi-card ${k.highlight ? 'highlight' : ''}">
          <div class="kpi-num">${k.val}</div>
          <div class="kpi-lbl">${k.lbl}</div>
          ${k.trend ? `<div class="kpi-trend ${k.trend}">${k.trend === 'up' ? '↑ Profitable' : '↓ Review costs'}</div>` : ''}
         </div>`
      ).join('');

    // Donut chart
    const costs = data.costs || [];
    if (costs.length) {
      const total = costs.reduce((s, c) => s + c.val, 0);
      const svg   = document.getElementById('donut-svg');
      const cx = 70, cy = 70, r = 52, inner = 30;
      let paths = '', startAngle = -Math.PI / 2;
      costs.forEach(c => {
        const angle    = (c.val / total) * 2 * Math.PI;
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
      document.getElementById('donut-legend').innerHTML =
        costs.map(c =>
          `<div class="legend-row">
            <div class="legend-dot" style="background:${c.color}"></div>
            <span class="legend-label">${c.lbl}</span>
            <span class="legend-val">${c.display}</span>
           </div>`
        ).join('');
    }

    document.getElementById('profit-analysis').textContent = data.analysis || '';
    showResult('profit-result');
    toast('Profit calculation complete!');
  } catch (e) {
    apiError(btn, `Profit: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ─────────────────────────────────────────────────────────────────
//  MODULE 8: MARKET PRICE  (overrides app.js)
// ─────────────────────────────────────────────────────────────────
async function runMarket(btn) {
  setLoading(btn, true);
  try {
    const payload = {
      crop:          document.getElementById('mkt-crop').value,
      market:        document.getElementById('mkt-market').value,
      forecast_days: parseInt(document.getElementById('mkt-period').value),
      current_price: parseInt(document.getElementById('mkt-current').value) || 2100,
      season:        document.getElementById('mkt-season2').value,
    };

    const data = await apiPost('/market/forecast', payload);

    document.getElementById('mkt-metrics').innerHTML =
      (data.metrics || []).map(m =>
        `<div class="metric-box">
          <div class="num" ${m.color ? `style="color:${m.color};"` : ''}>${m.val}</div>
          <div class="lbl">${m.lbl}</div>
         </div>`
      ).join('');

    // SVG Line Chart
    const prices = data.prices || [];
    if (prices.length > 1) {
      const svgEl  = document.getElementById('market-chart');
      const W = 580, H = 160, pad = { t: 10, r: 10, b: 30, l: 50 };
      const plotW  = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
      const maxP   = Math.max(...prices), minP = Math.min(...prices);
      const range  = maxP - minP || 100;
      const xScale = i => pad.l + (i / (prices.length - 1)) * plotW;
      const yScale = v => pad.t + plotH - ((v - minP) / range) * plotH;
      const finalP = prices[prices.length - 1];
      const change = finalP - payload.current_price;
      const lineColor = change >= 0 ? 'var(--green-soft)' : 'var(--danger)';
      const areaColor = change >= 0 ? 'rgba(76,156,110,0.1)' : 'rgba(192,57,43,0.08)';
      let pathD = prices.map((v, i) => (i === 0 ? 'M' : 'L') + xScale(i).toFixed(1) + ' ' + yScale(v).toFixed(1)).join(' ');
      let areaD = pathD + ` L${xScale(prices.length - 1)} ${pad.t + plotH} L${pad.l} ${pad.t + plotH} Z`;
      const ticks  = [minP, Math.round((minP + maxP) / 2), maxP];
      const step   = Math.ceil(prices.length / 4);
      let xLbls = '';
      for (let i = 0; i < prices.length; i += step) {
        xLbls += `<text x="${xScale(i)}" y="${pad.t + plotH + 16}" text-anchor="middle" fill="var(--text-muted)" font-size="10">Day ${i}</text>`;
      }
      svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svgEl.innerHTML =
        `<path d="${areaD}" fill="${areaColor}" stroke="none"/>
         <path d="${pathD}" fill="none" stroke="${lineColor}" stroke-width="2" stroke-linejoin="round"/>
         <circle cx="${xScale(prices.length - 1)}" cy="${yScale(finalP)}" r="4" fill="${lineColor}" stroke="#fff" stroke-width="1.5"/>
         ${ticks.map(t => `<text x="${pad.l - 6}" y="${yScale(t) + 4}" text-anchor="end" fill="var(--text-muted)" font-size="10" font-family="DM Mono,monospace">₹${t}</text>`).join('')}
         ${xLbls}`;
    }

    document.getElementById('mkt-insights').innerHTML     = data.insights || '';
    document.getElementById('mkt-sell-window').innerHTML  = (data.sell_tags || [])
      .map(t => `<span class="tag ${t.cls || ''}">${t.text}</span>`).join('');

    showResult('market-result');
    toast('Price forecast generated!');
  } catch (e) {
    apiError(btn, `Market: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
}
