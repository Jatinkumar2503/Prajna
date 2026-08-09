/* ============================================================
   3D SIMULATION — simulation3d.js
   Three.js digital twin: vessel, fuel rods, control rods,
   coolant particle system, radiation rings, orbit controls.
   ============================================================ */

var _three = null;

function initThree() {
  var cont = gel('twinc');
  if (!cont || !window.THREE) return null;

  var W = cont.clientWidth  || 600;
  var H = cont.clientHeight || 300;
  if (W < 10 || H < 10) { W = 600; H = 300; }

  /* ---- Scene setup ---------------------------------------- */
  var scene  = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 500);
  camera.position.set(0, 6, 14);
  camera.lookAt(0, 0, 0);

  var cv = gel('rcanvas');
  cv.width = W; cv.height = H;
  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, canvas: cv });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled   = true;
  renderer.toneMapping         = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  scene.fog = new THREE.FogExp2(0x000810, 0.04);

  /* ---- Lighting ------------------------------------------- */
  scene.add(new THREE.AmbientLight(0x001133, 0.8));
  var dlight = new THREE.DirectionalLight(0x00aaff, 1.5);
  dlight.position.set(5, 10, 5);
  dlight.castShadow = true;
  scene.add(dlight);
  var coreLight = new THREE.PointLight(0x00ff88, 3, 8);
  scene.add(coreLight);

  /* ---- Reactor vessel ------------------------------------- */
  var vMat   = new THREE.MeshStandardMaterial({ color: 0x0a2040, metalness: 0.9, roughness: 0.2, emissive: 0x001122, emissiveIntensity: 0.3 });
  var vessel = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 6, 32), vMat);
  vessel.castShadow = true;
  scene.add(vessel);

  var dome = new THREE.Mesh(new THREE.SphereGeometry(1.8, 32, 16, 0, Math.PI*2, 0, Math.PI/2), vMat);
  dome.position.set(0, 3, 0);
  scene.add(dome);

  var dbot = new THREE.Mesh(new THREE.SphereGeometry(1.8, 32, 16, 0, Math.PI*2, Math.PI/2, Math.PI/2), vMat);
  dbot.position.set(0, -3, 0);
  scene.add(dbot);

  // Wireframe highlight
  var wire = new THREE.Mesh(
    new THREE.CylinderGeometry(1.82, 1.82, 6.05, 32),
    new THREE.MeshBasicMaterial({ color: 0x00d4ff, wireframe: true, transparent: true, opacity: 0.08 })
  );
  scene.add(wire);

  /* ---- Fuel rods ------------------------------------------ */
  var coreMat = new THREE.MeshStandardMaterial({
    color: 0x00ff88, emissive: 0x00ff44, emissiveIntensity: 1.5,
    transparent: true, opacity: 0.85, roughness: 0.1
  });
  var rodPositions = [[0,0],[0.6,0],[0,0.6],[-0.6,0],[0,-0.6],[0.45,0.45],[-0.45,0.45],[0.45,-0.45],[-0.45,-0.45]];
  var fuelRods = [];
  for (var i = 0; i < rodPositions.length; i++) {
    var rod = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 4, 8), coreMat.clone());
    rod.position.set(rodPositions[i][0], 0, rodPositions[i][1]);
    scene.add(rod);
    fuelRods.push(rod);
  }

  /* ---- Control rods --------------------------------------- */
  var ctrlMat  = new THREE.MeshStandardMaterial({ color: 0x223344, metalness: 0.95, roughness: 0.1 });
  var ctrlRods = [];
  for (var i = 0; i < 5; i++) {
    var cr = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3, 8), ctrlMat);
    cr.position.set(rodPositions[i][0], 1.5, rodPositions[i][1]);
    scene.add(cr);
    ctrlRods.push(cr);
  }

  /* ---- Coolant pipes -------------------------------------- */
  var pipeMat  = new THREE.MeshStandardMaterial({ color: 0x003366, metalness: 0.8, roughness: 0.3 });
  var pipeConf = [
    { x:  2.2, z: 0,   rz: Math.PI/2 },
    { x: -2.2, z: 0,   rz: Math.PI/2 },
    { x: 0,    z:  2.2, rx: Math.PI/2 },
    { x: 0,    z: -2.2, rx: Math.PI/2 }
  ];
  for (var i = 0; i < pipeConf.length; i++) {
    var pc = pipeConf[i];
    var pm = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 1.2, 16), pipeMat);
    pm.position.set(pc.x, 0, pc.z);
    pm.rotation.x = pc.rx || 0;
    pm.rotation.z = pc.rz || 0;
    scene.add(pm);
  }

  /* ---- Coolant particle system ---------------------------- */
  var PC   = 200;
  var pGeo = new THREE.BufferGeometry();
  var pBuf = new Float32Array(PC * 3);
  var pSpd = new Float32Array(PC);
  var pAng = new Float32Array(PC);
  for (var i = 0; i < PC; i++) {
    var a = Math.random() * Math.PI * 2;
    var pr = 1.4 + Math.random() * 0.3;
    pBuf[i*3]   = Math.cos(a) * pr;
    pBuf[i*3+1] = (Math.random() - 0.5) * 5.5;
    pBuf[i*3+2] = Math.sin(a) * pr;
    pSpd[i] = 0.02 + Math.random() * 0.04;
    pAng[i] = a;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pBuf, 3));
  var pMat      = new THREE.PointsMaterial({ color: 0x0088ff, size: 0.06, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending });
  var particles = new THREE.Points(pGeo, pMat);
  scene.add(particles);

  /* ---- Radiation rings ------------------------------------ */
  var rings      = [];
  var ringRadii  = [1.2, 1.6, 2.0];
  for (var i = 0; i < ringRadii.length; i++) {
    var rm = new THREE.MeshBasicMaterial({ color: 0x00ff88, wireframe: true, transparent: true, opacity: 0.15 });
    var rg = new THREE.Mesh(new THREE.TorusGeometry(ringRadii[i], 0.01, 8, 40), rm);
    rg.rotation.x = Math.PI / 2;
    scene.add(rg);
    rings.push(rg);
  }

  /* ---- Grid floor ----------------------------------------- */
  var grid = new THREE.GridHelper(20, 20, 0x003355, 0x001830);
  grid.position.y = -5;
  scene.add(grid);

  /* ---- Orbit controls (manual implementation) ------------- */
  var drag = false, omx = 0, omy = 0, theta = 0.3, phi = 0.35, orbitR = 14, lt = null;

  cv.addEventListener('mousedown',  function (e) { drag = true; omx = e.clientX; omy = e.clientY; });
  window.addEventListener('mouseup',    function ()  { drag = false; });
  window.addEventListener('mousemove',  function (e) {
    if (!drag) return;
    theta -= (e.clientX - omx) * 0.008;
    phi   -= (e.clientY - omy) * 0.005;
    phi    = Math.max(0.1, Math.min(1.4, phi));
    omx    = e.clientX; omy = e.clientY;
  });
  cv.addEventListener('wheel', function (e) {
    e.preventDefault();
    orbitR = Math.max(5, Math.min(25, orbitR + e.deltaY * 0.02));
  }, { passive: false });
  cv.addEventListener('touchstart',  function (e) { if (e.touches.length === 1) { drag = true; lt = e.touches[0]; } });
  window.addEventListener('touchend',    function ()  { drag = false; });
  cv.addEventListener('touchmove', function (e) {
    if (!drag || !lt) return;
    var t = e.touches[0];
    theta -= (t.clientX - lt.clientX) * 0.008;
    phi   -= (t.clientY - lt.clientY) * 0.005;
    phi    = Math.max(0.1, Math.min(1.4, phi));
    lt     = t;
  });
  window.addEventListener('resize', function () {
    var W = cont.clientWidth, H = cont.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  });

  /* ---- Live reactor state (updated by APP via .update()) -- */
  var rs = { temperature: 285, coolantFlow: 78, neutronFlux: 2.3, radiation: 0.4, reactorPower: 92, controlRodPos: 68 };
  var animT = 0;

  /* ---- Animation loop ------------------------------------- */
  function animate() {
    requestAnimationFrame(animate);
    animT += 0.016;

    // Orbit camera
    camera.position.x = orbitR * Math.sin(theta) * Math.cos(phi);
    camera.position.y = orbitR * Math.sin(phi);
    camera.position.z = orbitR * Math.cos(theta) * Math.cos(phi);
    camera.lookAt(0, 0, 0);

    // Core glow driven by neutron flux and temperature
    var tf = Math.max(0, Math.min(1, (rs.temperature - 200) / 200));
    var ff = Math.max(0, Math.min(1, rs.neutronFlux / 5));
    coreLight.intensity = 0.3 + ff * 2.5 + Math.sin(animT * 3) * 0.3;
    coreLight.color.setRGB(tf, 1 - tf * 0.8, 0.2);

    // Fuel rod colour & glow
    for (var i = 0; i < fuelRods.length; i++) {
      fuelRods[i].material.emissiveIntensity = 0.8 + ff * 2 + Math.sin(animT * 4) * 0.3;
      fuelRods[i].material.color.setRGB(tf * 0.6, 1 - tf * 0.8, 0.4);
    }

    // Control rod insertion depth
    var cd = (100 - (rs.controlRodPos || 68)) / 100 * 2.5;
    for (var i = 0; i < ctrlRods.length; i++) {
      ctrlRods[i].position.y = 1.5 + cd * Math.sin(animT * 0.5 + 0.3);
    }

    // Coolant particle flow
    var fF = rs.coolantFlow / 78;
    for (var i = 0; i < PC; i++) {
      pAng[i]    += pSpd[i] * fF;
      var r2      = 1.4 + (i % 3) * 0.1;
      pBuf[i*3]   = Math.cos(pAng[i]) * r2;
      pBuf[i*3+1] -= pSpd[i] * 2 * fF;
      if (pBuf[i*3+1] < -3) pBuf[i*3+1] = 3;
      pBuf[i*3+2] = Math.sin(pAng[i]) * r2;
    }
    pGeo.attributes.position.needsUpdate = true;
    particles.material.color.setRGB(0, fF * 0.8, 1);

    // Radiation rings
    var rF = rs.radiation / 3;
    for (var i = 0; i < rings.length; i++) {
      rings[i].rotation.y = animT * (0.3 + i * 0.1);
      rings[i].rotation.z = animT * (0.2 + i * 0.05);
      rings[i].material.opacity = (0.08 + rF * 0.25) * (1 + Math.sin(animT * 2 + i) * 0.3);
      rings[i].scale.setScalar(1 + Math.sin(animT * 1.5 + i * 0.7) * 0.05);
      rings[i].material.color.setRGB(0, 1 - rF * 0.6, rF * 0.4);
    }

    // Vessel danger pulse
    var danger = rs.temperature > 310 || rs.coolantFlow < 65 || rs.neutronFlux > 3;
    vessel.material.emissiveIntensity = danger ? 0.4 + Math.sin(animT * 4) * 0.2 : 0.2;
    vessel.material.emissive.setRGB(danger ? 0.3 : 0, 0.05, danger ? 0.05 : 0.1);

    renderer.render(scene, camera);
  }
  animate();

  /* ---- Public update interface ---------------------------- */
  return {
    update: function (r, aiState) {
      rs = {
        temperature:   r.temperature.value,
        coolantFlow:   r.coolantFlow.value,
        neutronFlux:   r.neutronFlux.value,
        radiation:     r.radiation.value,
        reactorPower:  r.reactorPower,
        controlRodPos: r.controlRodPos,
        pinnForecast:  aiState && aiState.pinnForecast ? aiState.pinnForecast : null,
        timeToThreshold: aiState && aiState.timeToThreshold ? aiState.timeToThreshold : null
      };

      // If PINN forecasts temperature excursion, add forward thermal glow to rods
      if (rs.pinnForecast && rs.pinnForecast.temperature10s > 310.0) {
        var predictedTf = Math.min(1.0, (rs.pinnForecast.temperature10s - 280.0) / 80.0);
        for (var k = 0; k < fuelRods.length; k++) {
          fuelRods[k].material.color.setRGB(predictedTf, 1.0 - predictedTf * 0.9, 0.1);
        }
      }
    }
  };
}