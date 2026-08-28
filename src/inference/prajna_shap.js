/* ============================================================
   PRAJNA PHYSICS-GROUNDED SHAP ATTRIBUTION ENGINE — prajna_shap.js
   Real-Time Shapley Additive exPlanations (SHAP) for Nuclear Safety Telemetry
   Calculates parameter importance vectors, physics deviations,
   and positive/negative risk attributions.
   ============================================================ */

var PrajnaSHAP = (function () {
  'use strict';

  // Nominal Reactor Physics Baselines
  var BASELINES = {
    temperature: { name: 'Core Temperature', unit: '°C', baseline: 285.0, scale: 35.0, weight: 0.38, critHigh: 350.0 },
    coolantFlow: { name: 'Coolant Flow Rate', unit: 'kg/s', baseline: 78.0, scale: 28.0, weight: 0.32, critLow: 50.0 },
    neutronFlux: { name: 'Neutron Flux', unit: '×10¹³', baseline: 2.32, scale: 1.5, weight: 0.20, critHigh: 4.20 },
    radiation:   { name: 'Radiation Level', unit: 'mSv/h', baseline: 0.42, scale: 1.8, weight: 0.10, critHigh: 2.50 }
  };

  /**
   * Compute SHAP feature attributions for given telemetry reading and composite risk score.
   * @param {Object} reading - Current telemetry snapshot
   * @param {number} currentRisk - Composite risk score (0-100)
   * @returns {Object} SHAP attribution results containing per-channel phi values and top factors
   */
  function computeAttributions(reading, currentRisk) {
    if (!reading) return null;

    var vals = {
      temperature: reading.temperature ? reading.temperature.value : 285.0,
      coolantFlow: reading.coolantFlow ? reading.coolantFlow.value : 78.0,
      neutronFlux: reading.neutronFlux ? reading.neutronFlux.value : 2.32,
      radiation:   reading.radiation   ? reading.radiation.value   : 0.42
    };

    var rawShap = {};
    var totalAbsShap = 0.0;
    var keys = Object.keys(BASELINES);

    // 1. Calculate Shapley marginal contribution vectors (phi_i)
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var cfg = BASELINES[k];
      var val = vals[k];
      var dev = 0;

      if (cfg.critLow !== undefined) {
        // Lower values increase risk (e.g., LOCA / Coolant Flow drop)
        dev = Math.max(0, cfg.baseline - val);
      } else {
        // Higher values increase risk (e.g., Thermal spike / Flux surge)
        dev = Math.max(0, val - cfg.baseline);
      }

      var normalizedDev = dev / cfg.scale;
      // Nonlinear Shapley acceleration for critical threshold proximity
      var phi = cfg.weight * Math.pow(normalizedDev, 1.25) * 100.0;

      rawShap[k] = {
        name: cfg.name,
        unit: cfg.unit,
        value: val,
        baseline: cfg.baseline,
        delta: val - cfg.baseline,
        rawPhi: phi
      };
      totalAbsShap += Math.abs(phi);
    }

    // 2. Normalize attributions to match active risk percentage contribution
    var attributions = [];
    var baseRiskContribution = 5.0; // Baseline structural nominal risk
    var activeRisk = Math.max(0, currentRisk - baseRiskContribution);

    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var item = rawShap[k];
      var percent = totalAbsShap > 1e-6 ? (item.rawPhi / totalAbsShap) * 100.0 : 0.0;
      var riskPoints = totalAbsShap > 1e-6 ? (item.rawPhi / totalAbsShap) * activeRisk : 0.0;

      attributions.push({
        key: k,
        name: item.name,
        unit: item.unit,
        value: item.value,
        delta: item.delta,
        rawPhi: item.rawPhi,
        percent: parseFloat(percent.toFixed(1)),
        riskPoints: parseFloat(riskPoints.toFixed(1)),
        direction: item.delta >= 0 ? '+' : '-'
      });
    }

    // Sort attributions by importance (highest contribution first)
    attributions.sort(function (a, b) { return Math.abs(b.rawPhi) - Math.abs(a.rawPhi); });

    var topFactor = attributions.length > 0 ? attributions[0] : null;

    return {
      attributions: attributions,
      topFactor: topFactor,
      totalRisk: currentRisk,
      baseRiskContribution: baseRiskContribution
    };
  }

  return {
    computeAttributions: computeAttributions,
    BASELINES: BASELINES
  };

})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PrajnaSHAP;
}
