/* ============================================================
   AUDIO — audio.js
   Auto siren + beep based on AI alerts (no other file changes)
   ============================================================ */

var _actx = null;
var _lastLevel = null;
var _lastSirenTime = 0;

/* ---- Audio Context ---- */
function getACtx() {
  if (!_actx) _actx = new (window.AudioContext || window.webkitAudioContext)();
  return _actx;
}

/* ---- Siren ---- */
function playSiren(level) {
  try {
    var ctx = getACtx();
    if (ctx.state === 'suspended') ctx.resume();

    var osc = ctx.createOscillator();
    var gain = ctx.createGain();

    osc.type = "square";
    osc.connect(gain);
    gain.connect(ctx.destination);

    gain.gain.value = level === 'critical' ? 0.9 : 0.7;

    var t = ctx.currentTime;

    osc.frequency.setValueAtTime(600, t);
    osc.frequency.linearRampToValueAtTime(1000, t + 0.5);
    osc.frequency.linearRampToValueAtTime(600, t + 1);
    osc.frequency.linearRampToValueAtTime(1000, t + 1.5);
    osc.frequency.linearRampToValueAtTime(600, t + 2);

    osc.start(t);
    osc.stop(t + 2);

  } catch (e) {
    console.log("Siren error:", e);
  }
}

/* ---- Beep ---- */
function playBeep(freq, dur) {
  try {
    var ctx = getACtx();
    if (ctx.state === 'suspended') ctx.resume();

    var osc = ctx.createOscillator();
    var gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.value = freq || 800;

    osc.connect(gain);
    gain.connect(ctx.destination);

    gain.gain.value = 0.6;

    osc.start();
    osc.stop(ctx.currentTime + (dur || 0.0));
  } catch (e) {}
}

/* ---- Detect Alert Level ---- */
function getAlertLevel() {
  if (!window._aiState || !_aiState.alerts) return null;

  var alerts = _aiState.alerts;

  if (alerts.some(a => a.level === 'critical')) return 'critical';
  if (alerts.some(a => a.level === 'danger')) return 'danger';
  if (alerts.some(a => a.level === 'warning')) return 'warning';
  if (alerts.some(a => a.level === 'predictive')) return 'predictive';

  return null;
}

/* ---- Main Audio Loop ---- */
setInterval(function () {
  var level = getAlertLevel();
  if (!level) return;

  var now = Date.now();

  if (level === 'critical') {
    if (now - _lastSirenTime > 5000) {
      playSiren('critical');
      _lastSirenTime = now;
    }
  }
  else if (level === 'danger') {
    if (now - _lastSirenTime > 4000) {
      playSiren('danger');
      _lastSirenTime = now;
    }
  }
  else if (level === 'warning' && _lastLevel !== 'warning') {
    playBeep(900, 0.6);
  }
  else if (level === 'predictive' && _lastLevel !== 'predictive') {
    playBeep(600, 0.15);
  }

  _lastLevel = level;

}, 1000);

/* ---- Unlock audio on first click ---- */
document.body.addEventListener('click', function () {
  try {
    var ctx = getACtx();
    if (ctx.state === 'suspended') ctx.resume();
    console.log("Audio unlocked");
  } catch (e) {}
}, { once: true });
function getAlertLevel() {
  if (!window._aiState) return null;

  var rs = _aiState.riskScore;

  if (rs >= 60) return "critical";
  if (rs >= 42) return "danger";
  if (rs >= 25) return "warning";
  if (rs >= 10) return "predictive";

  return null;
}