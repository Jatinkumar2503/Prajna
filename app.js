/* ============================================================
   APP — app.js
   Main application loop: orchestrates SIM → AI → UI → Audio.
   Entry point — depends on simulation.js, ai-model.js,
   audio.js, ui.js, and simulation3d.js.
   ============================================================ */

var _running  = false;
var _iv       = null;
var _lastIds  = [];

var APP = {

  /* ---- Start the 1 Hz simulation loop --------------------- */
  start: function () {
    if (_running) return;
    _running = true;
    gel('bstart').style.display = 'none';
    gel('bstop').style.display  = 'inline-block';

    _iv = setInterval(function () {

      /* 1. Advance physics simulation by one tick */
      var r = SIM.tick();

      /* 2. Run AI inference on new sensor snapshot */
      _aiState = AI.infer(r);

      /* 3. Audio: play sirens and alerts for active risk states */
      if (typeof AUDIO !== 'undefined') {
        AUDIO.update();
      } else if (typeof playSiren !== 'undefined') {
        var curIds = r.alerts.map(function (a) { return a.id; });
        for (var i = 0; i < r.alerts.length; i++) {
          var al = r.alerts[i];
          if (_lastIds.indexOf(al.id) === -1) {
            if (al.level === 'critical' || al.level === 'danger') playSiren(al.level);
            else if (al.level === 'warning') playBeep(1000, 0.2);
          }
        }
        _lastIds = curIds;
      }


      /* 4. Update sensor panels */
      var keys = ['temperature', 'coolantFlow', 'neutronFlux', 'radiation'];
      for (var i = 0; i < keys.length; i++) updateSensor(keys[i], r);

      /* 5. Update right-panel UI components */
      updateAlerts(r.alerts);
      updateAI(_aiState);
      updateBanner(r.alerts);
      updateCharts(r);

      /* 6. Header counters */
      stxt('tick',      'T+' + r.time + 's');
      stxt('pwr',       r.reactorPower.toFixed(1));
      stxt('rod',       Math.round(r.controlRodPos));
      stxt('pwr-badge', 'PWR ' + r.reactorPower.toFixed(1) + '%');
      stxt('rod-badge', 'ROD ' + Math.round(r.controlRodPos) + '%');
      stxt('scenlbl',   r.scenario.active ? 'ACTIVE: ' + r.scenario.label : 'NO ACTIVE SCENARIO');
      stxt('scen-badge', r.scenario.active ? 'SCENARIO: ' + (r.scenario.label || '').split(' ')[0].toUpperCase() : 'SCENARIO: NONE');

      /* 7. Scenario progress bar */
      var sb = gel('sprogbar');
      if (sb) sb.style.width = (r.scenario.progress * 100).toFixed(1) + '%';

      /* 8. Highlight active scenario button */
      var btns = document.querySelectorAll('.sbtn');
      for (var i = 0; i < btns.length; i++) {
        var active = btns[i].getAttribute('data-sc') === r.scenario.type && r.scenario.active;
        if (active) btns[i].classList.add('active');
        else        btns[i].classList.remove('active');
      }

      /* 9. Status dot in header */
      var maxLv = 'n';
      for (var i = 0; i < r.alerts.length; i++) {
        if (r.alerts[i].level === 'critical') { maxLv = 'danger';  break; }
        if (r.alerts[i].level === 'danger')   { maxLv = 'danger';  }
        if (r.alerts[i].level === 'warning' && maxLv === 'n') maxLv = 'warning';
      }
      var dot = gel('sdot');
      if (dot) dot.className = 'sdot' + (maxLv === 'n' ? '' : ' ' + (maxLv === 'warning' ? 'warning' : 'danger'));
      stxt('stext', _aiState.classification);

      /* 10. Feed the 3D digital twin */
      if (_three) _three.update(r, _aiState);

    }, 1000);
  },

  /* ---- Pause the simulation ------------------------------- */
  stop: function () {
    _running = false;
    clearInterval(_iv);
    gel('bstart').style.display = 'inline-block';
    gel('bstop').style.display  = 'none';
  },

  /* ---- Inject a fault scenario ---------------------------- */
  trigger: function (type) {
    SIM.triggerScenario(type);
    playBeep(440, 0.3);
    if (!_running) APP.start();
  },

  /* ---- Clear active scenario and reset injectors ---------- */
  reset: function () {
    SIM.resetScenario();
    _lastIds = [];
    playBeep(660, 0.2);
  }
};

/* ---- Bootstrap after DOM ready --------------------------- */
window.addEventListener('DOMContentLoaded', function () {
  // Slight delay to let the container size be calculated
  setTimeout(function () {
    _three = initThree();
  }, 400);
});
document.body.addEventListener('click', function () {
  if (window.getACtx) getACtx();
}, { once: true });