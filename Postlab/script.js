/* ================================================================
   VisionAI — app.js
   MobileNetV2 Real-Time Classifier  |  ImageNet-1K
   ================================================================ */

'use strict';

/* ── State ─────────────────────────────────────────── */
let model     = null;
let stream    = null;
let running   = false;
let cont      = true;
let minConf   = 0.10;
let lastPreds = [];
let frames    = 0;
let fpsCount  = 0;
let lastFpsT  = performance.now();
let flipped   = true;

/* ── DOM ───────────────────────────────────────────── */
const video      = document.getElementById('webcam');
const overlay    = document.getElementById('overlay');
const startBtn   = document.getElementById('startBtn');
const logBody    = document.getElementById('logBody');
const hudScan    = document.getElementById('hudScan');
const hudFps     = document.getElementById('hudFps');
const snapBtn    = document.getElementById('snapBtn');
const stopBtn    = document.getElementById('stopBtn');
const singleBtn  = document.getElementById('singleBtn');
const snapPreview = document.getElementById('snapPreview');
const snapImg    = document.getElementById('snapImg');
const snapCanvas = document.getElementById('snapCanvas');
const corners    = document.querySelectorAll('.hud-corner');
const livechip   = document.getElementById('livechip');
const liveDot    = document.querySelector('#livechip .blink-dot');

/* ================================================================
   LOGGER
   ================================================================ */
function log(msg, type = 'dim') {
  const t = new Date().toTimeString().slice(0, 8);
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<span class="log-time">[${t}]</span><span class="log-msg ${type}">${msg}</span>`;
  logBody.appendChild(div);
  logBody.scrollTop = logBody.scrollHeight;
  while (logBody.children.length > 120) logBody.removeChild(logBody.firstChild);
}

function clearLog() { logBody.innerHTML = ''; }

/* ── Chip helper ── */
function setChip(el, text, colorClass) {
  el.innerHTML = text;
  el.className = `hchip ${colorClass}`;
}

/* ================================================================
   MODEL LOADING
   ================================================================ */
async function loadModel() {
  log('🚀 Initialising TensorFlow.js…', 'info');
  await tf.ready();
  const be = tf.getBackend();
  log(`⚡ Backend: ${be.toUpperCase()}`, 'ok');
  document.getElementById('backendChip').textContent = `⚡ ${be.toUpperCase()}`;
  document.getElementById('ovStatus').textContent = 'Downloading MobileNetV2 weights…';
  log('📦 Fetching MobileNetV2 (α1.0) from TF Hub…', 'info');

  try {
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    log('✅ MobileNetV2 loaded — 1000 ImageNet classes', 'ok');

    // Warmup pass to pre-compile shaders
    const dummy = tf.zeros([1, 224, 224, 3]);
    await model.classify(dummy);
    dummy.dispose();
    log('🔥 Warmup pass complete — ready!', 'ok');

    // Update overlay to ready state
    document.getElementById('ovIcon').textContent   = '✅';
    document.getElementById('ovTitle').textContent  = 'Model Ready!';
    document.getElementById('ovDesc').textContent   = 'Click the button below to launch your camera and start classifying objects in real time.';
    document.getElementById('ovSpinner').style.display = 'none';
    document.getElementById('ovStatus').textContent = 'MobileNetV2 · α1.0 · 1000 classes ready';
    startBtn.disabled = false;
    setChip(document.getElementById('modelChip'), '✅ Ready', 'hchip-lime');

  } catch (err) {
    log('❌ Model load failed: ' + err.message, 'err');
    document.getElementById('ovIcon').textContent  = '❌';
    document.getElementById('ovTitle').textContent = 'Load Failed';
    document.getElementById('ovDesc').textContent  = 'Check your internet connection and reload the page.';
    document.getElementById('ovSpinner').style.display = 'none';
    setChip(document.getElementById('modelChip'), '❌ Error', 'hchip-pink');
  }
}

/* ================================================================
   CAMERA
   ================================================================ */
async function startCamera() {
  log('📷 Requesting webcam access…', 'info');
  startBtn.disabled = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(res => { video.onloadedmetadata = res; });

    const W = video.videoWidth, H = video.videoHeight;
    log(`📐 Camera: ${W}×${H}`, 'ok');
    document.getElementById('resBadge').textContent = `${W} × ${H}`;

    // Dismiss overlay
    overlay.style.opacity = '0';
    setTimeout(() => { overlay.style.display = 'none'; }, 500);

    // Activate HUD
    hudScan.classList.add('on');
    corners.forEach(c => c.classList.add('on'));
    hudFps.style.display = 'block';

    // Show controls
    snapBtn.style.display   = 'inline-flex';
    stopBtn.style.display   = 'inline-flex';
    singleBtn.style.display = 'inline-flex';

    // Live glow on camera card
    document.getElementById('camCard').classList.add('live');

    // Update header chip
    livechip.innerHTML = `<span class="blink-dot" style="background:var(--lime);box-shadow:0 0 6px var(--lime)"></span> LIVE`;

    setChip(document.getElementById('modelChip'), '🟢 LIVE', 'hchip-lime');

    running = true;
    classifyLoop();
    log('🎬 Live classification running', 'ok');

  } catch (err) {
    log('🚫 Camera error: ' + err.message, 'err');
    overlay.style.display = 'flex';
    overlay.style.opacity = '1';
    document.getElementById('ovIcon').textContent  = '🚫';
    document.getElementById('ovTitle').textContent = 'Camera Denied';
    document.getElementById('ovDesc').textContent  = 'Allow camera access in your browser settings, then retry.';
    document.getElementById('ovSpinner').style.display = 'none';
    startBtn.disabled    = false;
    startBtn.textContent = '🔁 Retry';
  }
}

function stopCamera() {
  running = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

  overlay.style.display = 'flex';
  overlay.style.opacity = '1';
  document.getElementById('ovIcon').textContent  = '⏸';
  document.getElementById('ovTitle').textContent = 'Camera Stopped';
  document.getElementById('ovDesc').textContent  = 'Click Start Camera to resume recognition.';
  document.getElementById('ovSpinner').style.display = 'none';
  startBtn.disabled    = false;
  startBtn.textContent = '🎥 Start Camera';

  hudScan.classList.remove('on');
  corners.forEach(c => c.classList.remove('on'));
  hudFps.style.display  = 'none';
  snapBtn.style.display   = 'none';
  stopBtn.style.display   = 'none';
  singleBtn.style.display = 'none';

  document.getElementById('camCard').classList.remove('live');
  livechip.innerHTML = `<span class="blink-dot"></span> OFFLINE`;
  setChip(document.getElementById('modelChip'), '⏸ Stopped', 'hchip-orange');
  log('⏹ Camera stopped', 'warn');
}

/* ================================================================
   INFERENCE LOOP
   ================================================================ */
async function classifyLoop() {
  if (!running) return;

  if (video.readyState >= 2) {
    const t0 = performance.now();
    try {
      const preds = await model.classify(video, 5);
      const ms    = Math.round(performance.now() - t0);
      lastPreds   = preds;
      renderResults(preds);
      document.getElementById('sInf').textContent = ms;
    } catch (e) {
      log('⚠ Inference error: ' + e.message, 'err');
    }

    frames++;
    fpsCount++;
    document.getElementById('sFrames').textContent = frames;

    const now = performance.now();
    if (now - lastFpsT >= 1000) {
      const fps = Math.round(fpsCount * 1000 / (now - lastFpsT));
      document.getElementById('sFps').textContent = fps;
      hudFps.textContent = `⚡ ${fps} fps`;
      fpsCount = 0;
      lastFpsT = now;
    }
  }

  if (cont) requestAnimationFrame(classifyLoop);
}

async function singleInfer() {
  if (!model || video.readyState < 2) return;
  log('🔍 Single inference…', 'info');
  const t0    = performance.now();
  const preds = await model.classify(video, 5);
  const ms    = Math.round(performance.now() - t0);
  lastPreds   = preds;
  renderResults(preds);
  log(`✅ ${ms}ms · top: ${fmt(preds[0].className)} (${pct(preds[0].probability)}%)`, 'ok');
}

/* ================================================================
   RESULTS RENDERER
   ================================================================ */
function renderResults(preds) {
  const filtered = preds.filter(p => p.probability >= minConf);

  if (!filtered.length) {
    document.getElementById('resultList').innerHTML =
      '<div class="empty-msg"><span class="em-icon">🔍</span><span>Nothing above threshold</span></div>';
    return;
  }

  document.getElementById('resultList').innerHTML = filtered.map((p, i) => {
    const pc    = pct(p.probability);
    const tier  = pc >= 60 ? 'h' : pc >= 30 ? 'm' : 'l';
    const isTop = i === 0;
    const name  = fmt(p.className);
    const cat   = category(p.className);

    return `
    <div class="result-item">
      <div class="ri-top">
        <div class="ri-num ${isTop ? 'gold' : ''}">${i + 1}</div>
        <div class="ri-name ${isTop ? 'gold' : ''}">${name}</div>
        <div class="ri-pct pct-${tier}">${pc}%</div>
      </div>
      <div class="ri-track">
        <div class="ri-fill fill-${tier}" style="width:${pc}%"></div>
      </div>
      ${cat ? `<div class="ri-cat">📂 ${cat}</div>` : ''}
    </div>`;
  }).join('');
}

/* ── helpers ── */
const pct = p => Math.round(p * 100);

function fmt(n) {
  return n.split(',')[0].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function category(n) {
  n = n.toLowerCase();
  const map = [
    [['dog','cat','bird','fish','horse','elephant','tiger','lion','bear','rabbit','snake','turtle','frog'], '🐾 Animals & Wildlife'],
    [['car','truck','bus','motorcycle','bicycle','airplane','boat','train','vehicle'],                     '🚗 Vehicles & Transport'],
    [['laptop','computer','keyboard','phone','camera','monitor','tv','remote','tablet'],                  '💻 Electronics'],
    [['pizza','burger','apple','banana','coffee','wine','food','bread','cup','bottle','cake'],            '🍕 Food & Drink'],
    [['tree','flower','grass','plant','mushroom','rose','fern'],                                          '🌿 Plants & Nature'],
    [['chair','table','sofa','bed','desk','lamp','shelf'],                                                '🪑 Furniture'],
    [['shirt','shoes','hat','jacket','dress','glasses','watch'],                                          '👗 Clothing & Fashion'],
    [['house','building','bridge','church','stadium','tower'],                                            '🏛 Architecture'],
    [['cat','kitten','persian','siamese','tabby'],                                                        '🐱 Cats'],
  ];
  for (const [kws, label] of map) {
    if (kws.some(k => n.includes(k))) return label;
  }
  return null;
}

/* ================================================================
   ACTIONS
   ================================================================ */
function clearResults() {
  document.getElementById('resultList').innerHTML =
    '<div class="empty-msg"><span class="em-icon">🗑</span><span>Cleared</span></div>';
  lastPreds = [];
}

function copyResults() {
  if (!lastPreds.length) { log('⚠ Nothing to copy', 'warn'); return; }
  const text = lastPreds.map((p, i) =>
    `${i + 1}. ${fmt(p.className)}: ${pct(p.probability)}%`
  ).join('\n');
  navigator.clipboard.writeText(text)
    .then(() => log('📋 Copied to clipboard', 'ok'))
    .catch(() => log('Copy failed', 'err'));
}

function doSnapshot() {
  if (video.readyState < 2) return;
  snapCanvas.width  = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const ctx = snapCanvas.getContext('2d');
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -video.videoWidth, 0);
  ctx.restore();
  snapImg.src = snapCanvas.toDataURL('image/jpeg', 0.88);
  snapPreview.style.display = 'block';
  log('📸 Snapshot captured', 'ok');
  setTimeout(() => { snapPreview.style.display = 'none'; }, 4500);
}

/* ================================================================
   CONTROLS
   ================================================================ */
function toggleCont() {
  cont = !cont;
  const tog = document.getElementById('togCont');
  tog.classList.toggle('tog-on', cont);
  log(`🔄 Continuous mode: ${cont ? 'ON' : 'OFF'}`, 'info');
  if (cont && running) classifyLoop();
}

function toggleFlip() {
  flipped = !flipped;
  const tog = document.getElementById('togFlip');
  tog.classList.toggle('tog-on', flipped);
  video.style.transform = flipped ? 'scaleX(-1)' : 'scaleX(1)';
}

function setConf(v) {
  minConf = v / 100;
  document.getElementById('confLbl').textContent = v + '%';
}

/* ================================================================
   BOOT
   ================================================================ */
log('👁 VisionAI — booting…', 'info');
log('📦 TensorFlow.js + MobileNetV2 (v2 α1.0)', 'dim');
log('🗄 ImageNet-1K global database (1000 categories)', 'dim');
loadModel();