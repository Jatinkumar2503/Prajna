/* ============================================================
   AI MODEL — ai-model.js
   PINN + LSTM (EWMA) + GNN ensemble for anomaly detection,
   cross-parameter correlation scoring, and risk classification.
   ============================================================ */

var AI = (function () {

  /* ---- EWMA (Exponentially Weighted Moving Average) ---------
     Acts as the LSTM-style sequential memory component.
     Tracks online mean, variance, trend for each sensor signal. */
  function EWMA(a, b) {
    this.a        = a || 0.15;   // mean smoothing factor
    this.b        = b || 0.1;    // variance smoothing factor
    this.mean     = null;
    this.variance = 0.01;
    this.win      = [];          // sliding window for trend
  }

  EWMA.prototype.update = function (v) {
    this.win.push(v);
    if (this.win.length > 30) this.win.shift();
    if (this.mean === null) { this.mean = v; return; }
    var e = v - this.mean;
    this.mean     += this.a * e;
    this.variance  = (1 - this.b) * this.variance + this.b * e * e;
  };

  /** Z-score normalised anomaly score for a new observation. */
  EWMA.prototype.score = function (v) {
    if (this.mean === null || this.variance < 1e-9) return 0;
    return Math.abs((v - this.mean) / Math.sqrt(this.variance));
  };

  /** Linear trend direction from the sliding window. */
  EWMA.prototype.trend = function () {
    if (this.win.length < 6) return 0;
    var h  = Math.floor(this.win.length / 2);
    var f  = this.win.slice(0, h).reduce(function (a, b) { return a + b; }, 0) / h;
    var l  = this.win.slice(-h).reduce(function (a, b)  { return a + b; }, 0) / h;
    return (l - f) / h;
  };

  /* ---- Per-channel EWMA models (LSTM substitute) ----------- */
  var PR = {
    temperature: new EWMA(0.12, 0.08),
    coolantFlow: new EWMA(0.10, 0.07),
    neutronFlux: new EWMA(0.15, 0.10),
    radiation:   new EWMA(0.13, 0.09)
  };

  /* ---- GNN: short history buffers for correlation ----------- */
  var gnnH = { temperature: [], coolantFlow: [], neutronFlux: [], radiation: [] };

  /* ---- Expected correlation matrix (physics prior) ----------
     Based on reactor physics relationships between parameters. */
  var CM = {
    temperature: { coolantFlow: -0.72, neutronFlux:  0.65, radiation:  0.40 },
    coolantFlow: { temperature: -0.72, neutronFlux: -0.30, radiation: -0.25 },
    neutronFlux: { temperature:  0.65, coolantFlow: -0.30, radiation:  0.58 },
    radiation:   { temperature:  0.40, coolantFlow: -0.25, neutronFlux: 0.58 }
  };

  /* ---- Model state ----------------------------------------- */
  var ms = {
    riskScore:         0,
    crossAnomalyScore: 0,
    pinnDeviation:     0,
    classification:    'NOMINAL',
    confidence:        0.97,
    frameCount:        0,
    history:           [],
    trends:            {}
  };

  /* ---- GNN: Pearson correlation deviation score ------------- */
  function gnnScore() {
    var keys = Object.keys(gnnH);
    if (gnnH.temperature.length < 5) return 0;
    var td = 0, cnt = 0;
    for (var i = 0; i < keys.length; i++) {
      for (var j = 0; j < keys.length; j++) {
        var k1 = keys[i], k2 = keys[j];
        if (k1 >= k2) continue;
        var a1 = gnnH[k1], a2 = gnnH[k2], n = a1.length;
        var m1 = a1.reduce(function (s, v) { return s + v; }, 0) / n;
        var m2 = a2.reduce(function (s, v) { return s + v; }, 0) / n;
        var cov = a1.reduce(function (s, v, i) { return s + (v - m1) * (a2[i] - m2); }, 0) / n;
        var s1  = Math.sqrt(a1.reduce(function (s, v) { return s + (v-m1)*(v-m1); }, 0) / n) || 1e-9;
        var s2  = Math.sqrt(a2.reduce(function (s, v) { return s + (v-m2)*(v-m2); }, 0) / n) || 1e-9;
        td  += Math.abs(cov / (s1 * s2) - CM[k1][k2]);
        cnt++;
      }
    }
    return Math.min(1, (cnt ? td / cnt : 0) / 1.5);
  }

  /* ---- Main inference call ---------------------------------- */
  function infer(reading) {
    ms.frameCount++;

    var v = {
      temperature: reading.temperature.value,
      coolantFlow: reading.coolantFlow.value,
      neutronFlux: reading.neutronFlux.value,
      radiation:   reading.radiation.value
    };
    var keys = Object.keys(v);

    // Update EWMA models and GNN history
    for (var i = 0; i < keys.length; i++) {
      PR[keys[i]].update(v[keys[i]]);
      gnnH[keys[i]].push(v[keys[i]]);
      if (gnnH[keys[i]].length > 20) gnnH[keys[i]].shift();
    }

    // Individual anomaly scores per channel
    var ind = {};
    for (var i = 0; i < keys.length; i++) ind[keys[i]] = PR[keys[i]].score(v[keys[i]]);

    // Cross-parameter GNN score
    var cs = gnnScore();

    // PINN Neural Inference Runtime Integration
    var pinnResult = null;
    if (typeof window !== 'undefined' && window.pinnEngine) {
      pinnResult = window.pinnEngine.predict(reading);
    } else if (typeof pinnEngine !== 'undefined') {
      pinnResult = pinnEngine.predict(reading);
    }

    // Physics-informed thermal deviation
    var edt     = (v.neutronFlux / 2.3) * 285 - 285;
    var cc      = (v.coolantFlow / 78)  * 285;
    var pinnDev = Math.abs(v.temperature - (285 + edt - (cc - 285) * 0.3));
    if (pinnResult && pinnResult.physicsResiduals) {
      pinnDev = Math.max(pinnDev, pinnResult.physicsResiduals.energyBalanceLoss * 50.0);
    }

    // Composite risk score
    var maxI = Math.max(ind.temperature, ind.coolantFlow, ind.neutronFlux, ind.radiation);
    var pF   = Math.min(1, pinnDev / 30);
    var raw  = (maxI * 0.45 + cs * 0.35 + pF * 0.20) * 25;

    // Alert-level boost
    var boost = reading.alerts.filter(function (a) { return a.level === 'critical';  }).length * 20
              + reading.alerts.filter(function (a) { return a.level === 'danger';    }).length * 10
              + reading.alerts.filter(function (a) { return a.level === 'warning';   }).length * 4;

    var risk = Math.min(100, raw + boost);

    // Classification
    var cls = 'NOMINAL', conf = 0.97 - raw * 0.003;
    if      (risk > 75) { cls = 'CRITICAL ANOMALY'; conf = 0.91; }
    else if (risk > 50) { cls = 'ANOMALY DETECTED'; conf = 0.88; }
    else if (risk > 25) { cls = 'ELEVATED RISK';    conf = 0.93; }
    else if (risk > 10) { cls = 'MONITORING';        conf = 0.96; }

    // Per-channel trend labels
    var trends = {};
    for (var i = 0; i < keys.length; i++) {
      var t = PR[keys[i]].trend();
      trends[keys[i]] = t > 0.05 ? 'up' : t < -0.05 ? 'down' : 'stable';
    }

    ms = {
      riskScore:         +risk.toFixed(1),
      crossAnomalyScore: +cs.toFixed(3),
      pinnDeviation:     +pinnDev.toFixed(2),
      classification:    cls,
      confidence:        +Math.min(0.999, Math.max(0.70, conf)).toFixed(3),
      frameCount:        ms.frameCount,
      trends:            trends,
      pinnForecast:      pinnResult ? pinnResult.forecast : null,
      timeToThreshold:   pinnResult ? pinnResult.timeToThreshold : null,
      eopGuidance:       pinnResult ? pinnResult.eopGuidance : null,
      physicsResiduals:  pinnResult ? pinnResult.physicsResiduals : null
    };
    ms.history = (ms.history || []).concat([{ t: reading.time, risk: risk }]).slice(-60);
    return ms;
  }

  /* ---- Public API ------------------------------------------- */
  return { infer: infer };

})();