//  ============================================================
//    SIMULATION ENGINE — PHYSICS-ENHANCED VERSION
//    Now includes real reactor physics equations
//    ============================================================ */

var SIM = (function () {

  /* ---- Threshold definitions -------------------------------- */
  var THR = {
    temperature: { warning: 290, danger: 320, critical: 350, max: 420,  unit: '°C',          label: 'Core Temperature',  inv: false },
    coolantFlow: { warning: 72,  danger: 60,  critical: 50,  min: 0,    unit: 'kg/s',         label: 'Coolant Flow Rate', inv: true  },
    neutronFlux: { warning: 2.8, danger: 3.5, critical: 4.2, max: 5.5,  unit: '×10¹³n/cm²s', label: 'Neutron Flux',      inv: false },
    radiation:   { warning: 0.8, danger: 1.5, critical: 2.5, max: 6.0,  unit: 'mSv/h',        label: 'Radiation Level',   inv: false }
  };

  /* ---- Baseline values -------------------------------------- */
  var BASE = { temperature: 285, coolantFlow: 78, neutronFlux: 2.32, radiation: 0.42 };

  /* ---- Physics constants (NEW) ------------------------------- */
  var PHYS = {
    k_power: 40,        // scaling: neutronFlux → power
    heatCoeff: 0.08,    // power → temperature increase
    coolingCoeff: 0.06, // coolantFlow → cooling
    radiationCoeff: 0.5 // neutronFlux → radiation
  };

  /* ---- Scenario definitions --------------------------------- */
  var SCEN = {
    coolantLoss: {
      label: 'Loss of Coolant (LOCA)',
      dur: 120,
      fx: function (p) { return { coolantFlow: -36*p, neutronFlux: 1.2*p }; }
    },
    controlRodFailure: {
      label: 'Control Rod Ejection',
      dur: 90,
      fx: function (p) { return { neutronFlux: 2.8*p }; }
    },
    steamTubeRupture: {
      label: 'Steam Generator Tube Rupture',
      dur: 110,
      fx: function (p) { return { coolantFlow: -22*p }; }
    }
  };

  /* ---- Internal state --------------------------------------- */
  var st = {
    time: 0,
    temperature:  { value: 285 },
    coolantFlow:  { value: 78  },
    neutronFlux:  { value: 2.32 },
    radiation:    { value: 0.42 },
    reactorPower: 92,
    controlRodPos: 68,
    scenario: { active: false, type: null, progress: 0 },
    history:  { temperature: [], coolantFlow: [], neutronFlux: [], radiation: [] },
    alerts:   [],
    predictions: {}
  };

  /* ---- Gaussian noise --------------------------------------- */
  function gauss(s) {
    var u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---- Cyclic variation ------------------------------------- */
  function cyc(t) {
    return 0.3*Math.sin(t*0.02) + 0.15*Math.sin(t*0.073+1.2);
  }

  /* ---- Prediction ------------------------------------------- */
  function predict(key) {
    var h = st.history[key];
    if (h.length < 12) return null;

    var arr = h.slice(-20), n = arr.length, xm = (n-1)/2;
    var ym  = arr.reduce((s, x) => s + x.v, 0) / n;
    var num = arr.reduce((s, x, i) => s + (i - xm) * (x.v - ym), 0);
    var den = arr.reduce((s, x, i) => s + (i - xm)*(i - xm), 0);
    var sl = num / den;

    var t = THR[key], cur = st[key].value;

    if (t.inv) {
      if (sl >= 0) return null;
      return Math.max(0, Math.round((cur - t.danger) / Math.abs(sl)));
    } else {
      if (sl <= 0) return null;
      return Math.max(0, Math.round((t.danger - cur) / sl));
    }
  }

  /* ---- Alerts ----------------------------------------------- */
  function makeAlerts() {
    var al = [];
    var keys = ['temperature', 'coolantFlow', 'neutronFlux', 'radiation'];

    keys.forEach(function (key) {
      var t = THR[key], v = st[key].value, pr = st.predictions[key];
      var lv = 'normal';

      if (t.inv ? v <= t.critical : v >= t.critical) lv = 'critical';
      else if (t.inv ? v <= t.danger : v >= t.danger) lv = 'danger';
      else if (t.inv ? v <= t.warning : v >= t.warning) lv = 'warning';

      if (lv !== 'normal') {
        al.push({ param: t.label, value: v.toFixed(2), unit: t.unit, level: lv });
      }
    });

    st.alerts = al;
  }

  /* ---- MAIN PHYSICS TICK ------------------------------------ */
  function tick() {
    st.time++;

    var sc = st.scenario;
    var fx = { coolantFlow: 0, neutronFlux: 0 };

    if (sc.active && sc.type) {
      var S = SCEN[sc.type];
      sc.progress += 1 / S.dur;
      if (sc.progress >= 1) sc.active = false;
      else fx = S.fx(sc.progress);
    }

    /* ========= PHYSICS EQUATIONS ========= */

    // 1. Neutron Flux Dynamics (affected by rods)
    var rodEffect = (100 - st.controlRodPos) / 100;
    st.neutronFlux.value += 0.02 * rodEffect + fx.neutronFlux + gauss(0.01);

    // 2. Reactor Power ∝ Neutron Flux
    st.reactorPower = PHYS.k_power * st.neutronFlux.value;

    // 3. Heat Generation
    var heatGenerated = PHYS.heatCoeff * st.reactorPower;

    // 4. Cooling Effect
    var cooling = PHYS.coolingCoeff * (st.coolantFlow.value + fx.coolantFlow);

    // 5. Temperature Equation
    st.temperature.value += heatGenerated - cooling + gauss(0.3);

    // 6. Coolant Flow update
    st.coolantFlow.value += fx.coolantFlow + gauss(0.2);

    // 7. Radiation ∝ Neutron Flux
    st.radiation.value = PHYS.radiationCoeff * st.neutronFlux.value + gauss(0.01);

    /* ==================================== */

    // Clamp values
    st.temperature.value = Math.min(THR.temperature.max, Math.max(0, st.temperature.value));
    st.coolantFlow.value = Math.max(0, st.coolantFlow.value);
    st.neutronFlux.value = Math.max(0, st.neutronFlux.value);
    st.radiation.value   = Math.max(0, st.radiation.value);

    // Save history
    ['temperature','coolantFlow','neutronFlux','radiation'].forEach(function (k) {
      st.history[k].push({ t: st.time, v: st[k].value });
      if (st.history[k].length > 120) st.history[k].shift();
      st.predictions[k] = predict(k);
    });

    // Control rod response (auto safety)
    st.controlRodPos = Math.min(100, Math.max(0, st.controlRodPos - st.neutronFlux.value * 0.5));

    makeAlerts();
    return snap();
  }

  function snap() {
    return {
      time: st.time,
      temperature:   { value: +st.temperature.value.toFixed(1) },
      coolantFlow:   { value: +st.coolantFlow.value.toFixed(1) },
      neutronFlux:   { value: +st.neutronFlux.value.toFixed(3) },
      radiation:     { value: +st.radiation.value.toFixed(3) },
      reactorPower:  +st.reactorPower.toFixed(1),
      controlRodPos: +st.controlRodPos.toFixed(1),
      alerts: st.alerts,
      predictions: st.predictions,
      history: st.history,
      thresholds: THR,
      scenario: st.scenario
    };
  }

  function triggerScenario(type) {
    if (!SCEN[type]) return false;
    st.scenario = { active: true, type: type, progress: 0 };
    return true;
  }

  function resetScenario() {
    st.scenario = { active: false, type: null, progress: 0 };
  }

  return {
    tick: tick,
    triggerScenario: triggerScenario,
    resetScenario: resetScenario
  };

})();