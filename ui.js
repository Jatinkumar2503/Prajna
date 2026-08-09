/* ============================================================
   UI — ui.js
   DOM manipulation helpers, sensor panel updates, alert cards,
   AI gauge render, banner controller, and Canvas trend charts.
   ============================================================ */

/* ---- Basic DOM helpers ------------------------------------ */
function gel(id) { return document.getElementById(id); }
function stxt(id, v) { var e = gel(id); if (e) e.textContent = v; }

/* ---- Live clock ------------------------------------------ */
setInterval(function () {
  stxt('clock', new Date().toLocaleTimeString('en-US', { hour12: false }));
}, 1000);

/* Shared AI state reference (written by app.js) */
var _aiState = null;

/* ---- Sensor level classifier ----------------------------- */
function lvl(v, t) {
  if (t.inv) {
    if (v <= t.critical) return 'c';
    if (v <= t.danger)   return 'd';
    if (v <= t.warning)  return 'w';
  } else {
    if (v >= t.critical) return 'c';
    if (v >= t.danger)   return 'd';
    if (v >= t.warning)  return 'w';
  }
  return 'n';
}

/* ---- Bar percentage normaliser --------------------------- */
function barPct(v, t) {
  if (t.inv) return Math.min(100, Math.max(2, ((90 - v) / 90) * 100));
  return Math.min(100, Math.max(2, (v / (t.max || 400)) * 100));
}

/* ---- Update a single sensor block ------------------------ */
function updateSensor(key, r) {
  var v  = r[key].value;
  var t  = r.thresholds[key];
  var lv = lvl(v, t);

  // Value display
  var el = gel('sv-' + key);
  if (el) {
    el.textContent = (key === 'neutronFlux') ? v.toFixed(3) : v.toFixed(1);
    el.className   = 'sval ' + lv;
  }

  // Progress bar
  var bar = gel('sb-' + key);
  if (bar) {
    bar.style.width = barPct(v, t) + '%';
    bar.className   = 'sbar b' + lv;
  }

  // Threshold labels
  stxt('slo-' + key, t.inv ? 'CRIT: ' + t.critical : 'WARN: ' + t.warning);
  stxt('shi-' + key, t.inv ? 'WARN: ' + t.warning   : 'CRIT: ' + t.critical);

  // Predictive countdown
  var pr = r.predictions[key];
  var pe = gel('sp-' + key);
  if (pe) {
    if (pr !== null && pr !== undefined && lv === 'n') {
      var m = Math.floor(pr / 60), s = pr % 60;
      pe.style.display    = 'block';
      pe.style.borderColor = pr < 120 ? '#ff2244' : '#ffaa00';
      pe.style.color       = pr < 120 ? '#ff2244' : '#ffaa00';
      pe.style.background  = pr < 120 ? 'rgba(255,34,68,0.1)' : 'rgba(255,170,0,0.1)';
      pe.textContent       = '⏱ DANGER IN ' + m + 'm ' + s + 's';
    } else {
      pe.style.display = 'none';
    }
  }

  // Trend arrow
  var tr = _aiState && _aiState.trends ? _aiState.trends[key] : 'stable';
  var te = gel('tr-' + key);
  if (te) te.textContent = tr === 'up' ? '↑' : tr === 'down' ? '↓' : '→';
}

/* ---- Render alert cards ---------------------------------- */
function updateAlerts(alerts) {
  stxt('acnt', alerts.length > 0 ? alerts.length + ' ACTIVE' : '0');
  var list = gel('alist');
  if (!list) return;
  if (!alerts.length) {
    list.innerHTML = '<div class="noalerts"><span style="font-size:18px">✓</span>ALL SYSTEMS NOMINAL</div>';
    return;
  }
  var ord    = { critical: 0, danger: 1, warning: 2, predictive: 3 };
  var sorted = alerts.slice().sort(function (a, b) { return (ord[a.level] || 9) - (ord[b.level] || 9); });
  var cols   = { critical: '#ff0022', danger: '#ff6644', warning: '#ffaa00', predictive: '#00d4ff' };
  list.innerHTML = sorted.map(function (al) {
    var col = cols[al.level] || '#fff';
    var cd  = (al.ttl !== null && al.ttl !== undefined)
      ? '<div class="acd">⏱ ' + Math.floor(al.ttl / 60) + 'm ' + (al.ttl % 60) + 's to danger limit</div>'
      : '';
    return '<div class="acard ' + al.level + '">'
      + '<div class="ahdr"><span class="alvl" style="color:' + col + '">' + al.level.toUpperCase() + '</span>'
      + '<span class="atm">' + al.ts + '</span></div>'
      + '<div class="aparam">' + al.param + '</div>'
      + '<div class="aval" style="color:' + col + '">' + al.value
      + ' <span style="font-size:8px;opacity:0.7">' + al.unit + '</span></div>'
      + cd + '</div>';
  }).join('');
}


/* ---- Render AI risk gauge & metadata --------------------- */
function updateAI(ai) {
  var rs  = ai.riskScore;
  var col = rs > 75 ? '#ff0022' : rs > 50 ? '#ff6644' : rs > 25 ? '#ffaa00' : '#00ff88';

  // Arc gauge
  var cv = gel('rgc');
  if (cv) {
    var ctx = cv.getContext('2d'), W = cv.width, H = cv.height, cx = W/2, cy = H/2, rad = cx * 0.82;
    ctx.clearRect(0, 0, W, H);
    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, rad, Math.PI * 0.75, Math.PI * 0.25, false);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 8; ctx.lineCap = 'round';
    ctx.stroke();
    // Value arc
    ctx.beginPath();
    ctx.arc(cx, cy, rad, Math.PI * 0.75, Math.PI * 0.75 + (rs / 100) * Math.PI * 1.5, false);
    ctx.strokeStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 12;
    ctx.lineWidth = 8; ctx.lineCap = 'round';
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Numeric risk value
  var rv = gel('rval');
  if (rv) { rv.textContent = Math.round(rs); rv.className = 'rval ' + (rs > 75 ? 'c' : rs > 50 ? 'd' : rs > 25 ? 'w' : 'n'); }

  // Classification label
  var rc = gel('rcls');
  if (rc) { rc.textContent = ai.classification; rc.className = 'rcls ' + (rs > 75 ? 'c' : rs > 50 ? 'd' : rs > 25 ? 'w' : 'n'); }

  stxt('rconf',   'CONFIDENCE: ' + (ai.confidence * 100).toFixed(1) + '%');
  stxt('pinndev', 'PINN Δ: '    + ai.pinnDeviation + '°C');
  stxt('xscore',  'Cross: '     + (ai.crossAnomalyScore * 100).toFixed(1) + '%');

  // PINN Neural Forecast & IAEA EOP Action Display
  if (ai.pinnForecast) {
    var pForecastEl = gel('pinn-forecast-text');
    if (pForecastEl) {
      pForecastEl.textContent = 'T+10s Forecast: Core ' + ai.pinnForecast.temperature10s + '°C | Flow ' + ai.pinnForecast.coolantFlow10s + ' kg/s';
    }
  }
  if (ai.eopGuidance) {
    var eopEl = gel('eop-guidance-badge');
    if (eopEl) {
      eopEl.textContent = ai.eopGuidance.eopCode + ' (' + ai.eopGuidance.confidencePct + '%)';
      eopEl.title = ai.eopGuidance.eopTitle;
      eopEl.className = 'eop-badge ' + (ai.eopGuidance.priority === 'CRITICAL' ? 'crit' : 'norm');
    }
  }
  if (ai.timeToThreshold && ai.timeToThreshold.criticalTTL !== null) {
    var ttlBadge = gel('ttl-countdown-badge');
    if (ttlBadge) {
      ttlBadge.textContent = '⏱ TTL: ' + ai.timeToThreshold.criticalTTL + 's';
      ttlBadge.style.display = 'inline-block';
    }
  } else {
    var ttlBadge2 = gel('ttl-countdown-badge');
    if (ttlBadge2) ttlBadge2.style.display = 'none';
  }
}

/* ---- Top alert banner ------------------------------------ */
function updateBanner(alerts) {
  var b = gel('alert-banner');
  if (!b) return;
  var c = alerts.filter(function (a) { return a.level === 'critical'; })[0];
  var d = alerts.filter(function (a) { return a.level === 'danger';   })[0];
  var w = alerts.filter(function (a) { return a.level === 'warning';  })[0];
  if (c) {
    b.textContent = '⚠ CRITICAL — ' + c.param.toUpperCase() + ' EXCEEDS CRITICAL THRESHOLD ⚠';
    b.className   = 'show critical';
    b.style.display = 'flex';
  } else if (d) {
    b.textContent = '⚠ DANGER — ' + d.param.toUpperCase() + ' IN DANGER ZONE ⚠';
    b.className   = 'show danger';
    b.style.display = 'flex';
  } else if (w) {
    b.textContent = 'WARNING — ' + w.param.toUpperCase() + ' APPROACHING LIMIT';
    b.className   = 'show warning';
    b.style.display = 'flex';
  } else {
    b.className     = '';
    b.style.display = 'none';
  }
}


/* ---- Canvas trend charts --------------------------------- */
var CDEFS = [
  { key: 'temperature', color: '#ff6644', id: 'ct' },
  { key: 'coolantFlow', color: '#00aaff', id: 'cc' },
  { key: 'neutronFlux', color: '#00ff88', id: 'cf' },
  { key: 'radiation',   color: '#ffaa00', id: 'cr' }
];

function drawChart(def, hist, thr) {
  var cv = gel(def.id);
  if (!cv) return;
  var pw = cv.parentElement;
  var W  = pw ? pw.clientWidth  : 100;
  var H  = pw ? pw.clientHeight : 55;
  if (W < 1 || H < 1) return;
  cv.width = W; cv.height = H;
  var ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (!hist || hist.length < 2) return;

  var vals = hist.map(function (h) { return h.v; });
  var mn   = Math.min.apply(null, vals) * 0.98;
  var mx   = Math.max.apply(null, vals) * 1.02;
  var rng  = mx - mn || 1;

  // Background
  ctx.fillStyle = 'rgba(0,8,16,0.6)';
  ctx.fillRect(0, 0, W, H);

  // Threshold lines
  var lines = [[thr.warning, '#ffaa00'], [thr.danger, '#ff2244']];
  for (var i = 0; i < lines.length; i++) {
    var tv = lines[i][0], tc = lines[i][1];
    if (tv === undefined) continue;
    var y = H - ((tv - mn) / rng) * H;
    ctx.strokeStyle = tc; ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }

  // Signal line
  ctx.beginPath();
  ctx.strokeStyle = def.color; ctx.lineWidth = 1.5;
  ctx.shadowColor = def.color; ctx.shadowBlur = 4;
  for (var i = 0; i < vals.length; i++) {
    var x = (i / (vals.length - 1)) * W;
    var y = H - ((vals[i] - mn) / rng) * H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Fill gradient under line
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  var g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, def.color + '44');
  g.addColorStop(1, def.color + '00');
  ctx.fillStyle = g;
  ctx.fill();
}

function updateCharts(r) {
  for (var i = 0; i < CDEFS.length; i++) {
    drawChart(CDEFS[i], r.history[CDEFS[i].key], r.thresholds[CDEFS[i].key]);
  }
}