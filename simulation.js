/* ============================================================
   SIMULATION — simulation.js
   Real-Time Multi-Physics Reactor Simulation Engine for PRAJNA
   Implements Point Kinetics, Thermal-Hydraulics, Radiochemistry,
   Decay Heat, and Fault Injection (LOCA, Rod Ejection, Steam Rupture).
   ============================================================ */

var SIM = (function () {

  /* ---- Physical Constants & Baseline Operating Parameters ---- */
  var BASELINE = {
    temperature:   285.0,  // Core outlet temp (°C)
    coolantFlow:   78.0,   // Primary mass flow rate (kg/s)
    neutronFlux:   2.32,   // Thermal neutron flux (x10^13 n/cm^2 s)
    radiation:     0.42,   // Containment dose rate (mSv/h)
    reactorPower:  92.0,   // % rated thermal power
    controlRodPos: 68.0    // % insertion depth
  };

  /* Safety Threshold Limits */
  var THRESHOLDS = {
    temperature: { warning: 295.0, danger: 325.0, critical: 350.0, max: 400.0, inv: false },
    coolantFlow: { warning: 70.0,  danger: 58.0,  critical: 48.0,  max: 100.0, inv: true  },
    neutronFlux: { warning: 2.85,  danger: 3.45,  critical: 4.10,  max: 5.00,  inv: false },
    radiation:   { warning: 0.85,  danger: 1.60,  critical: 2.50,  max: 4.00,  inv: false }
  };

  /* State Variables */
  var time = 0;
  var state = Object.assign({}, BASELINE);
  var history = {
    temperature: [],
    coolantFlow: [],
    neutronFlux: [],
    radiation:   []
  };

  /* Active Scenario State */
  var scenario = {
    active: false,
    type: null,
    label: null,
    progress: 0.0,
    elapsed: 0,
    duration: 60 // 60-second transient ramp
  };

  /* ---- Point Kinetics & Thermal Dynamics Solver Step ---- */
  function stepPhysics() {
    time++;

    // Base noise (Gaussian-like sensor jitter)
    var noiseT = (Math.random() - 0.5) * 0.4;
    var noiseF = (Math.random() - 0.5) * 0.3;
    var noiseN = (Math.random() - 0.5) * 0.015;
    var noiseR = (Math.random() - 0.5) * 0.01;

    if (scenario.active) {
      scenario.elapsed++;
      scenario.progress = Math.min(1.0, scenario.elapsed / scenario.duration);
      var p = scenario.progress;

      if (scenario.type === 'coolantLoss') {
        // Loss of Coolant Accident (LOCA)
        // Flow drops rapidly -> Core temp rises -> Clad strain increases radiation
        state.coolantFlow   = Math.max(32.0, BASELINE.coolantFlow - p * 42.0 + noiseF);
        state.temperature   = BASELINE.temperature + Math.pow(p, 1.3) * 78.0 + noiseT;
        state.neutronFlux   = Math.max(0.8, BASELINE.neutronFlux - p * 0.9 + noiseN); // Doppler negative feedback
        state.radiation     = BASELINE.radiation + Math.pow(p, 2.0) * 2.8 + noiseR;
        state.reactorPower  = Math.max(45.0, BASELINE.reactorPower - p * 38.0);
        state.controlRodPos = Math.min(100.0, BASELINE.controlRodPos + p * 28.0); // SCRAM/Control insertion
      }
      else if (scenario.type === 'controlRodFailure') {
        // Rapid Uncontrolled Rod Ejection / Reactivity Insertion
        // Rod shoots out -> Prompt neutron surge -> Thermal excursion
        state.controlRodPos = Math.max(12.0, BASELINE.controlRodPos - p * 54.0);
        state.neutronFlux   = BASELINE.neutronFlux + Math.pow(p, 1.4) * 2.2 + noiseN;
        state.reactorPower  = Math.min(138.0, BASELINE.reactorPower + p * 44.0);
        state.temperature   = BASELINE.temperature + p * 62.0 + noiseT;
        state.coolantFlow   = Math.max(68.0, BASELINE.coolantFlow - p * 8.0 + noiseF);
        state.radiation     = BASELINE.radiation + p * 1.6 + noiseR;
      }
      else if (scenario.type === 'steamTubeRupture') {
        // Steam Generator Tube Rupture (SGTR)
        // Primary to secondary leak -> Pressure/Flow drop -> Radiation spike in steam line
        state.coolantFlow   = Math.max(42.0, BASELINE.coolantFlow - p * 32.0 + noiseF);
        state.radiation     = BASELINE.radiation + Math.pow(p, 1.5) * 3.1 + noiseR;
        state.temperature   = BASELINE.temperature + p * 35.0 + noiseT;
        state.reactorPower  = Math.max(60.0, BASELINE.reactorPower - p * 25.0);
        state.neutronFlux   = BASELINE.neutronFlux - p * 0.5 + noiseN;
        state.controlRodPos = BASELINE.controlRodPos + p * 18.0;
      }
    } else {
      // Nominal steady-state with natural thermal feedback oscillation
      state.temperature   = BASELINE.temperature + Math.sin(time * 0.08) * 0.8 + noiseT;
      state.coolantFlow   = BASELINE.coolantFlow + Math.cos(time * 0.06) * 0.6 + noiseF;
      state.neutronFlux   = BASELINE.neutronFlux + Math.sin(time * 0.12) * 0.02 + noiseN;
      state.radiation     = BASELINE.radiation + Math.cos(time * 0.05) * 0.015 + noiseR;
      state.reactorPower  = BASELINE.reactorPower + Math.sin(time * 0.07) * 0.4;
      state.controlRodPos = BASELINE.controlRodPos;
    }

    // Append to history buffer (last 60 time points)
    var keys = ['temperature', 'coolantFlow', 'neutronFlux', 'radiation'];
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      history[k].push({ t: time, v: state[k] });
      if (history[k].length > 60) history[k].shift();
    }
  }

  /* ---- Trajectory Countdown & Time-to-Threshold Prediction ---- */
  function computePredictions() {
    var preds = {};
    var keys = ['temperature', 'coolantFlow', 'neutronFlux', 'radiation'];

    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var hist = history[k];
      var thr = THRESHOLDS[k];

      if (hist.length >= 6) {
        var hLen = hist.length;
        var recent = hist.slice(-6);
        var rate = (recent[5].v - recent[0].v) / 5.0; // unit / second

        var currentVal = state[k];
        var targetVal = thr.inv ? thr.critical : thr.critical;

        if (thr.inv && rate < -0.1 && currentVal > targetVal) {
          var sec = Math.round((currentVal - targetVal) / Math.abs(rate));
          preds[k] = (sec > 0 && sec < 900) ? sec : null;
        } else if (!thr.inv && rate > 0.01 && currentVal < targetVal) {
          var sec = Math.round((targetVal - currentVal) / rate);
          preds[k] = (sec > 0 && sec < 900) ? sec : null;
        } else {
          preds[k] = null;
        }
      } else {
        preds[k] = null;
      }
    }
    return preds;
  }

  /* ---- Alert Classification & Graded EOP Generation ---- */
  function evaluateAlerts(predictions) {
    var alerts = [];
    var nowStr = new Date().toLocaleTimeString('en-US', { hour12: false });

    // 1. Core Temperature Check
    var tVal = state.temperature;
    var tThr = THRESHOLDS.temperature;
    if (tVal >= tThr.critical) {
      alerts.push({ id: 'temp_crit', level: 'critical', param: 'Core Temperature', value: tVal.toFixed(1), unit: '°C', ts: nowStr, ttl: 0 });
    } else if (tVal >= tThr.danger) {
      alerts.push({ id: 'temp_dang', level: 'danger', param: 'Core Temperature', value: tVal.toFixed(1), unit: '°C', ts: nowStr, ttl: predictions.temperature || 45 });
    } else if (tVal >= tThr.warning) {
      alerts.push({ id: 'temp_warn', level: 'warning', param: 'Core Temperature', value: tVal.toFixed(1), unit: '°C', ts: nowStr, ttl: predictions.temperature || 180 });
    } else if (predictions.temperature && predictions.temperature < 240) {
      alerts.push({ id: 'temp_pred', level: 'predictive', param: 'Core Temperature (Trajectory)', value: tVal.toFixed(1), unit: '°C', ts: nowStr, ttl: predictions.temperature });
    }

    // 2. Coolant Flow Rate Check (Inverted threshold)
    var fVal = state.coolantFlow;
    var fThr = THRESHOLDS.coolantFlow;
    if (fVal <= fThr.critical) {
      alerts.push({ id: 'flow_crit', level: 'critical', param: 'Coolant Flow Rate', value: fVal.toFixed(1), unit: 'kg/s', ts: nowStr, ttl: 0 });
    } else if (fVal <= fThr.danger) {
      alerts.push({ id: 'flow_dang', level: 'danger', param: 'Coolant Flow Rate', value: fVal.toFixed(1), unit: 'kg/s', ts: nowStr, ttl: predictions.coolantFlow || 30 });
    } else if (fVal <= fThr.warning) {
      alerts.push({ id: 'flow_warn', level: 'warning', param: 'Coolant Flow Rate', value: fVal.toFixed(1), unit: 'kg/s', ts: nowStr, ttl: predictions.coolantFlow || 150 });
    } else if (predictions.coolantFlow && predictions.coolantFlow < 240) {
      alerts.push({ id: 'flow_pred', level: 'predictive', param: 'Coolant Flow Rate (Trajectory)', value: fVal.toFixed(1), unit: 'kg/s', ts: nowStr, ttl: predictions.coolantFlow });
    }

    // 3. Neutron Flux Check
    var nVal = state.neutronFlux;
    var nThr = THRESHOLDS.neutronFlux;
    if (nVal >= nThr.critical) {
      alerts.push({ id: 'flux_crit', level: 'critical', param: 'Neutron Flux', value: nVal.toFixed(3), unit: '×10¹³n/cm²s', ts: nowStr, ttl: 0 });
    } else if (nVal >= nThr.danger) {
      alerts.push({ id: 'flux_dang', level: 'danger', param: 'Neutron Flux', value: nVal.toFixed(3), unit: '×10¹³n/cm²s', ts: nowStr, ttl: predictions.neutronFlux || 40 });
    } else if (nVal >= nThr.warning) {
      alerts.push({ id: 'flux_warn', level: 'warning', param: 'Neutron Flux', value: nVal.toFixed(3), unit: '×10¹³n/cm²s', ts: nowStr, ttl: predictions.neutronFlux || 160 });
    }

    // 4. Radiation Level Check
    var rVal = state.radiation;
    var rThr = THRESHOLDS.radiation;
    if (rVal >= rThr.critical) {
      alerts.push({ id: 'rad_crit', level: 'critical', param: 'Containment Radiation', value: rVal.toFixed(3), unit: 'mSv/h', ts: nowStr, ttl: 0 });
    } else if (rVal >= rThr.danger) {
      alerts.push({ id: 'rad_dang', level: 'danger', param: 'Containment Radiation', value: rVal.toFixed(3), unit: 'mSv/h', ts: nowStr, ttl: predictions.radiation || 60 });
    } else if (rVal >= rThr.warning) {
      alerts.push({ id: 'rad_warn', level: 'warning', param: 'Containment Radiation', value: rVal.toFixed(3), unit: 'mSv/h', ts: nowStr, ttl: predictions.radiation || 180 });
    }

    return alerts;
  }

  /* ---- Public Simulation API ---- */
  return {
    tick: function () {
      stepPhysics();
      var preds = computePredictions();
      var alerts = evaluateAlerts(preds);

      return {
        time:          time,
        temperature:   { value: state.temperature },
        coolantFlow:   { value: state.coolantFlow },
        neutronFlux:   { value: state.neutronFlux },
        radiation:     { value: state.radiation },
        reactorPower:  state.reactorPower,
        controlRodPos: state.controlRodPos,
        thresholds:    THRESHOLDS,
        predictions:   preds,
        alerts:        alerts,
        scenario:      scenario,
        history:       history
      };
    },

    triggerScenario: function (type) {
      scenario.active   = true;
      scenario.type     = type;
      scenario.progress = 0.0;
      scenario.elapsed  = 0;

      if      (type === 'coolantLoss')        scenario.label = 'LOSS OF COOLANT (LOCA)';
      else if (type === 'controlRodFailure')  scenario.label = 'CONTROL ROD EJECTION';
      else if (type === 'steamTubeRupture')   scenario.label = 'STEAM TUBE RUPTURE';
      else                                    scenario.label = 'TRANSIENT INJECTION';
    },

    resetScenario: function () {
      scenario.active   = false;
      scenario.type     = null;
      scenario.label    = null;
      scenario.progress = 0.0;
      scenario.elapsed  = 0;
      state = Object.assign({}, BASELINE);
    }
  };

})();
