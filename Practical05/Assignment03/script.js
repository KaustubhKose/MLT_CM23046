/**
 * Assignment 3: Measure FPS & Analyze Performance Impact of Real-Time Inference
 *
 * Metrics captured:
 *  - Render FPS   : requestAnimationFrame loop rate (how fast the browser ticks)
 *  - Inference FPS: how many model.classify() calls complete per second
 *  - Latency      : wall-clock time (ms) for each classify() call
 *  - Frame drop % : frames where inference couldn't keep up with rendering
 *  - Min / Max / Avg / P95 latency over the session
 *
 * Architecture:
 *  - One rAF loop for rendering & FPS counting (never blocked)
 *  - Inference runs asynchronously; a rate limiter caps calls/sec
 *  - All raw samples stored in `latencyLog[]` for summary stats
 *  - Canvas chart draws rolling 60-sample FPS history (render + infer)
 */

// ── DOM ────────────────────────────────────────────────────────────
const video      = document.getElementById('webcam');
const overlay    = document.getElementById('overlay');
const fpsHud     = document.getElementById('fps-hud');
const fpsChart   = document.getElementById('fps-chart');
const btnStart   = document.getElementById('btn-start');
const btnStop    = document.getElementById('btn-stop');
const btnClear   = document.getElementById('btn-clear');
const rateSlider = document.getElementById('rate-limit');
const rateVal    = document.getElementById('rate-val');
const logBody    = document.getElementById('log-body');
const logCount   = document.getElementById('log-count');
const statsGrid  = document.getElementById('stats-grid');

const mFps      = document.getElementById('m-fps');
const mInferFps = document.getElementById('m-infer-fps');
const mLatency  = document.getElementById('m-latency');
const mDrop     = document.getElementById('m-drop');
const mCount    = document.getElementById('m-count');

const overlayCtx = overlay.getContext('2d');
const chartCtx   = fpsChart.getContext('2d');

// ── Config & State ─────────────────────────────────────────────────
let model       = null;
let stream      = null;
let running     = false;
let animId      = null;
let maxInferPerSec = 10;

// FPS counters
let renderFrames  = 0;
let inferFrames   = 0;
let droppedFrames = 0;
let lastSecTs     = 0;
let currentRenderFps = 0;
let currentInferFps  = 0;

// Latency tracking
let latencyLog   = [];   // { ts, latency, label, confidence }
let latencySum   = 0;
let inferBusy    = false;
let lastInferTs  = 0;    // timestamp of last inference START
let sessionStart = 0;

// Chart history (last 60 seconds)
const HIST = 60;
const renderFpsHistory = [];
const inferFpsHistory  = [];

// ── Rate Slider ────────────────────────────────────────────────────
rateSlider.addEventListener('input', () => {
  maxInferPerSec = parseInt(rateSlider.value);
  rateVal.textContent = maxInferPerSec;
});

// ── Load Model ─────────────────────────────────────────────────────
async function loadModel() {
  mFps.textContent = '…';
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
}

// ── Start ──────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (!model) await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));

  running      = true;
  sessionStart = performance.now();
  lastSecTs    = sessionStart;

  btnStart.disabled = true;
  btnStop.disabled  = false;
  btnClear.disabled = false;

  requestAnimationFrame(renderLoop);
});

// ── Stop ───────────────────────────────────────────────────────────
btnStop.addEventListener('click', () => {
  running = false;
  cancelAnimationFrame(animId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  stream = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  btnStart.disabled = false;
  btnStop.disabled  = true;
  renderSummary();
});

// ── Clear Log ──────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
  latencyLog = [];
  latencySum = 0;
  renderFrames = inferFrames = droppedFrames = 0;
  renderFpsHistory.length = 0;
  inferFpsHistory.length  = 0;
  logBody.innerHTML = '<tr><td colspan="5" class="empty-row">Log cleared</td></tr>';
  logCount.textContent = '(0 entries)';
  statsGrid.innerHTML  = '<p class="stats-placeholder">Log cleared.</p>';
  mCount.textContent   = '0';
  mLatency.textContent = '—';
  mDrop.textContent    = '—';
});

// ── Render Loop (rAF — never blocked by inference) ─────────────────
async function renderLoop(now) {
  if (!running) return;

  // Sync canvas to video
  overlay.width  = video.videoWidth  || 640;
  overlay.height = video.videoHeight || 480;

  renderFrames++;

  // ── Per-second tick ──
  const elapsed = now - lastSecTs;
  if (elapsed >= 1000) {
    currentRenderFps = Math.round(renderFrames / (elapsed / 1000));
    currentInferFps  = Math.round(inferFrames  / (elapsed / 1000));

    renderFpsHistory.push(currentRenderFps);
    inferFpsHistory.push(currentInferFps);
    if (renderFpsHistory.length > HIST) renderFpsHistory.shift();
    if (inferFpsHistory.length  > HIST) inferFpsHistory.shift();

    const drop = renderFrames > 0
      ? ((droppedFrames / renderFrames) * 100).toFixed(1)
      : '0.0';

    mFps.textContent      = currentRenderFps;
    mInferFps.textContent = currentInferFps;
    mDrop.textContent     = drop + '%';
    fpsHud.textContent    = `FPS: ${currentRenderFps}`;

    renderFrames  = 0;
    inferFrames   = 0;
    droppedFrames = 0;
    lastSecTs     = now;

    drawChart();
  }

  // ── Throttled inference ──
  const minGap = 1000 / maxInferPerSec; // ms between inferences
  if (!inferBusy && video.readyState === 4 && (now - lastInferTs) >= minGap) {
    lastInferTs = now;
    runInference(now);       // fire-and-forget; renderLoop continues
  } else if (inferBusy) {
    droppedFrames++;
  }

  animId = requestAnimationFrame(renderLoop);
}

// ── Inference (async, detached from rAF) ──────────────────────────
async function runInference(startTs) {
  inferBusy = true;
  const t0 = performance.now();
  try {
    const preds   = await model.classify(video, 3);
    const latency = performance.now() - t0;
    const top     = preds[0];

    inferFrames++;
    latencySum += latency;
    const entry = {
      ts: ((startTs - sessionStart) / 1000).toFixed(2),
      latency: Math.round(latency),
      label: top.className.split(',')[0],
      confidence: (top.probability * 100).toFixed(1)
    };
    latencyLog.push(entry);

    // Update live metrics
    mLatency.textContent = Math.round(latencySum / latencyLog.length) + ' ms';
    mCount.textContent   = latencyLog.length;

    drawOverlayLabel(preds);
    appendLogRow(entry);
  } catch (_) {}
  inferBusy = false;
}

// ── Draw label on canvas overlay ──────────────────────────────────
function drawOverlayLabel(preds) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!preds || preds.length === 0) return;
  const top   = preds[0];
  const label = `${top.className.split(',')[0]}  ${(top.probability * 100).toFixed(1)}%`;
  const x = 14, y = overlay.height - 20;
  const fs = Math.max(13, overlay.width / 46);

  overlayCtx.font = `600 ${fs}px 'Segoe UI', sans-serif`;
  const tw = overlayCtx.measureText(label).width;
  overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
  overlayCtx.beginPath();
  overlayCtx.roundRect(x - 8, y - fs - 6, tw + 20, fs + 14, 6);
  overlayCtx.fill();
  overlayCtx.fillStyle = '#a78bfa';
  overlayCtx.fillText(label, x, y);
}

// ── Append row to log table ────────────────────────────────────────
function appendLogRow(entry) {
  if (logBody.querySelector('.empty-row')) logBody.innerHTML = '';

  const latClass = entry.latency < 80
    ? 'latency-fast'
    : entry.latency < 200 ? 'latency-med' : 'latency-slow';

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${latencyLog.length}</td>
    <td>${entry.ts}</td>
    <td class="${latClass}">${entry.latency}</td>
    <td>${entry.label}</td>
    <td>${entry.confidence}%</td>
  `;
  logBody.prepend(tr);

  // Keep max 200 rows in DOM
  while (logBody.children.length > 200) logBody.removeChild(logBody.lastChild);
  logCount.textContent = `(${latencyLog.length} entries)`;
}

// ── Chart: rolling FPS history ─────────────────────────────────────
function drawChart() {
  const W = fpsChart.offsetWidth || 800;
  const H = 160;
  fpsChart.width  = W;
  fpsChart.height = H;

  chartCtx.clearRect(0, 0, W, H);

  const maxFps = Math.max(60, ...renderFpsHistory, ...inferFpsHistory, 1);
  const pad    = { top: 12, right: 16, bottom: 28, left: 40 };
  const cW     = W - pad.left - pad.right;
  const cH     = H - pad.top - pad.bottom;

  // Grid lines
  chartCtx.strokeStyle = '#1e1e30';
  chartCtx.lineWidth   = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const y = pad.top + cH * (1 - t);
    chartCtx.beginPath();
    chartCtx.moveTo(pad.left, y);
    chartCtx.lineTo(pad.left + cW, y);
    chartCtx.stroke();
    chartCtx.fillStyle = '#4b5563';
    chartCtx.font = '10px sans-serif';
    chartCtx.fillText(Math.round(maxFps * t), 4, y + 4);
  });

  // X-axis label
  chartCtx.fillStyle = '#4b5563';
  chartCtx.font = '10px sans-serif';
  chartCtx.fillText('← last 60 s', pad.left, H - 8);

  function drawLine(data, color) {
    if (data.length < 2) return;
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth   = 1.5;
    chartCtx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (HIST - 1)) * cW;
      const y = pad.top  + cH * (1 - v / maxFps);
      i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  }

  drawLine(renderFpsHistory, '#4ade80');
  drawLine(inferFpsHistory,  '#fbbf24');
}

// ── Performance Summary ────────────────────────────────────────────
function renderSummary() {
  if (latencyLog.length === 0) {
    statsGrid.innerHTML = '<p class="stats-placeholder">No data recorded.</p>';
    return;
  }
  const lats = latencyLog.map(e => e.latency).sort((a, b) => a - b);
  const avg  = Math.round(lats.reduce((s, v) => s + v, 0) / lats.length);
  const min  = lats[0];
  const max  = lats[lats.length - 1];
  const p95  = lats[Math.floor(lats.length * 0.95)];
  const p50  = lats[Math.floor(lats.length * 0.5)];
  const dur  = ((performance.now() - sessionStart) / 1000).toFixed(1);

  const cards = [
    { k: 'Session Duration',    v: dur + ' s' },
    { k: 'Total Inferences',    v: latencyLog.length },
    { k: 'Avg Latency',         v: avg + ' ms' },
    { k: 'Min Latency',         v: min + ' ms' },
    { k: 'Max Latency',         v: max + ' ms' },
    { k: 'Median (P50)',        v: p50 + ' ms' },
    { k: '95th Percentile',     v: p95 + ' ms' },
    { k: 'Peak Render FPS',     v: Math.max(...renderFpsHistory) || '—' },
    { k: 'Peak Inference FPS',  v: Math.max(...inferFpsHistory)  || '—' },
  ];

  statsGrid.innerHTML = cards.map(({ k, v }) => `
    <div class="stat-card">
      <div class="sv">${v}</div>
      <div class="sk">${k}</div>
    </div>
  `).join('');
}

// ── Init ───────────────────────────────────────────────────────────
loadModel();