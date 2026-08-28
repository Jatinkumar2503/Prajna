/* ============================================================
   PRAJNA HIGH-FIDELITY NUCLEAR EMERGENCY SIREN SYNTHESIZER — audio.js
   Web Audio API dual-oscillator wailing air-horn & alarm system.
   Generates continuous dual-tone industrial evacuation sirens
   during any critical or high-risk reactor condition.
   ============================================================ */

var AUDIO = (function () {
  'use strict';

  var _actx = null;
  var _activeSiren = null;
  var _lastLevel = null;

  function getACtx() {
    if (!_actx) {
      var AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        _actx = new AudioContextClass();
      }
    }
    if (_actx && _actx.state === 'suspended') {
      _actx.resume().catch(function () {});
    }
    return _actx;
  }

  /**
   * Start continuous authentic nuclear wailing air siren / warning horn
   * @param {string} mode - 'critical' | 'danger' | 'warning'
   */
  function startSiren(mode) {
    stopSiren(); // Clear any existing sound

    var ctx = getACtx();
    if (!ctx) return;

    var now = ctx.currentTime;

    // Dual Oscillators: Sawtooth + Sub-Square for rich metallic industrial timbre
    var osc1 = ctx.createOscillator();
    var osc2 = ctx.createOscillator();

    // LFO for continuous frequency modulation (pitch sweep up and down)
    var lfo = ctx.createOscillator();
    var lfoGain = ctx.createGain();

    // Biquad filter to warm up harsh harmonics into acoustic air-horn resonator
    var filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';

    // Master gain & smooth envelope
    var masterGain = ctx.createGain();

    osc1.type = 'sawtooth';
    osc2.type = 'square';
    lfo.type  = 'sine';

    if (mode === 'critical') {
      // Nuclear Evacuation Air Siren: 420 Hz <-> 980 Hz continuous sweep at 0.75 Hz
      osc1.frequency.setValueAtTime(700, now);
      osc2.frequency.setValueAtTime(700 * 1.498, now); // Perfect fifth harmonic (1.5x)
      
      lfo.frequency.setValueAtTime(0.75, now); // Continuous wail frequency
      lfoGain.gain.setValueAtTime(280, now);   // Pitch modulation depth

      filter.frequency.setValueAtTime(3000, now);
      masterGain.gain.setValueAtTime(0.001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.38, now + 0.25);
    } else if (mode === 'danger') {
      // High Danger Warning Siren: 520 Hz <-> 860 Hz sweep at 1.2 Hz cycle
      osc1.frequency.setValueAtTime(680, now);
      osc2.frequency.setValueAtTime(680 * 1.25, now); // Major third harmonic

      lfo.frequency.setValueAtTime(1.2, now);
      lfoGain.gain.setValueAtTime(180, now);

      filter.frequency.setValueAtTime(2500, now);
      masterGain.gain.setValueAtTime(0.001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.28, now + 0.2);
    } else {
      // Warning Chime
      osc1.type = 'sine';
      osc2.type = 'triangle';
      osc1.frequency.setValueAtTime(800, now);
      osc2.frequency.setValueAtTime(1600, now);
      lfo.frequency.setValueAtTime(2.5, now);
      lfoGain.gain.setValueAtTime(50, now);

      filter.frequency.setValueAtTime(3200, now);
      masterGain.gain.setValueAtTime(0.001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.15, now + 0.2);
    }

    // Connect LFO modulation to oscillator frequencies
    lfo.connect(lfoGain);
    lfoGain.connect(osc1.frequency);
    lfoGain.connect(osc2.frequency);

    // Connect oscillators through lowpass filter and master gain
    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(masterGain);
    masterGain.connect(ctx.destination);

    // Start generators
    osc1.start(now);
    osc2.start(now);
    lfo.start(now);

    _activeSiren = {
      osc1: osc1,
      osc2: osc2,
      lfo: lfo,
      masterGain: masterGain,
      mode: mode
    };
  }

  /**
   * Stop active siren smoothly
   */
  function stopSiren() {
    if (!_activeSiren) return;
    var ctx = getACtx();
    if (ctx && _activeSiren.masterGain) {
      try {
        var now = ctx.currentTime;
        var curGain = Math.max(0.001, _activeSiren.masterGain.gain.value);
        _activeSiren.masterGain.gain.setValueAtTime(curGain, now);
        _activeSiren.masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);
        var s = _activeSiren;
        setTimeout(function () {
          try { s.osc1.stop(); s.osc2.stop(); s.lfo.stop(); } catch (e) {}
        }, 160);
      } catch (e) {}
    }
    _activeSiren = null;
  }

  /**
   * Main audio loop checking active AI alert state
   */
  function update() {
    var level = null;

    if (window._aiState) {
      var rs = window._aiState.riskScore || 0;
      var hasCritAlert = false;
      if (_aiState.history && _aiState.history.length) {
        var lastH = _aiState.history[_aiState.history.length - 1];
        if (lastH && lastH.alerts) {
          hasCritAlert = lastH.alerts.some(function (a) { return a.level === 'critical' || a.level === 'danger'; });
        }
      }

      if (rs >= 45 || hasCritAlert) level = 'critical';
      else if (rs >= 28) level = 'danger';
      else if (rs >= 15) level = 'warning';
    }

    if (level !== _lastLevel) {
      _lastLevel = level;
      if (level === 'critical' || level === 'danger') {
        startSiren(level); // Triggers full continuous evacuation siren!
      } else {
        stopSiren();
      }
    }
  }

  // Poll alert state every 150ms
  setInterval(update, 150);

  // Auto unlock AudioContext on initial click or keypress
  if (typeof window !== 'undefined') {
    var unlock = function () {
      getACtx();
      window.removeEventListener('click', unlock);
      window.removeEventListener('keydown', unlock);
    };
    window.addEventListener('click', unlock);
    window.addEventListener('keydown', unlock);
  }

  function playSiren(level) { startSiren(level || 'critical'); }
  function playBeep(freq, dur) { startSiren('warning'); setTimeout(stopSiren, (dur || 0.5) * 1000); }

  return {
    startSiren: startSiren,
    stopSiren: stopSiren,
    playSiren: playSiren,
    playBeep: playBeep,
    update: update
  };
})();