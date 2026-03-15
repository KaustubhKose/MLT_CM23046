/**
 * Assignment 2: Overlay Classification Labels on Live Video Feed
 *
 * Approach:
 *  - A <canvas> is positioned directly on top of the <video> (CSS position:absolute)
 *  - Each inference cycle: draw label text + confidence bar onto the canvas
 *  - Also update the HUD <div> and the live predictions table below the video
 *  - Toggle controls let the user turn the overlay and continuous mode on/off
 */

// ── DOM References ────────────────────────────────────────────────
const video       = document.getElementById('webcam');
const overlay     = document.getElementById('overlay');
const hudLabel    = document.getElementById('hud-label');
const hudStatus   = document.getElementById('hud-status');
const btnStart    = document.getElementById('btn-start');
const btnStop     = document.getElementById('btn-stop');
const predBody    = document.getElementById('pred-body');
const toggleShow  = document.getElementById('toggle-overlay');
const toggleCont  = document.getElementById('toggle-continuous');

const ctx = overlay.getContext('2d');

// ── State ─────────────────────────────────────────────────────────
let model     = null;
let stream    = null;
let animId    = null;
let running   = false;
let lastPreds = [];

// ── Load Model ────────────────────────────────────────────────────
async function loadModel() {
  hudStatus.textContent = 'loading model…';
  hudStatus.className   = 'hud-status loading';
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  hudStatus.textContent = 'ready';
  hudStatus.className   = 'hud-status';
}

// ── Start Webcam ─────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (!model) await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));

  btnStart.disabled = true;
  btnStop.disabled  = false;
  running = true;
  hudStatus.textContent = 'live';
  hudStatus.className   = 'hud-status live';

  requestAnimationFrame(inferLoop);
});

// ── Stop ──────────────────────────────────────────────────────────
btnStop.addEventListener('click', () => {
  running = false;
  cancelAnimationFrame(animId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  hudLabel.textContent  = '—';
  hudStatus.textContent = 'stopped';
  hudStatus.className   = 'hud-status';
  btnStart.disabled = false;
  btnStop.disabled  = true;
});

// ── Inference Loop ────────────────────────────────────────────────
let inferBusy = false;

async function inferLoop() {
  if (!running) return;

  // Sync canvas size to actual rendered video dimensions
  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;

  if (!inferBusy && toggleCont.checked && video.readyState === 4) {
    inferBusy = true;
    try {
      const preds = await model.classify(video, 3);
      lastPreds = preds;
      drawOverlay(preds);
      updateHUD(preds);
      updateTable(preds);
    } catch (_) {}
    inferBusy = false;
  } else if (!toggleCont.checked) {
    // Still redraw last results on every frame (overlay follows video)
    drawOverlay(lastPreds);
  }

  animId = requestAnimationFrame(inferLoop);
}

// ── Draw Canvas Overlay ───────────────────────────────────────────
function drawOverlay(preds) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!toggleShow.checked || preds.length === 0) return;

  const W = overlay.width;
  const H = overlay.height;
  const barH = 28;
  const barW = Math.min(W * 0.55, 360);
  const startY = H - (preds.length * (barH + 8)) - 56; // above HUD label
  const startX = 16;

  preds.forEach(({ className, probability }, i) => {
    const y = startY + i * (barH + 8);
    const filled = probability * barW;

    // Bar background
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    roundRect(ctx, startX, y, barW, barH, 6);
    ctx.fill();

    // Filled bar
    const colors = ['#66fcf1', '#45a29e', '#2a7875'];
    ctx.fillStyle = colors[i] || '#66fcf1';
    if (filled > 6) {
      roundRect(ctx, startX, y, filled, barH, 6);
      ctx.fill();
    }

    // Label text
    ctx.fillStyle = i === 0 ? '#ffffff' : '#c5c6c7';
    ctx.font = `${i === 0 ? 600 : 400} ${i === 0 ? 13 : 12}px 'Segoe UI', sans-serif`;
    ctx.fillText(
      `${className.split(',')[0]}  ${(probability * 100).toFixed(1)}%`,
      startX + 10,
      y + 18
    );
  });
}

// Helper: rounded rect path
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ── Update HUD Div ────────────────────────────────────────────────
function updateHUD(preds) {
  if (preds.length === 0) { hudLabel.textContent = '—'; return; }
  const top = preds[0];
  hudLabel.textContent = `${top.className.split(',')[0]}  ${(top.probability * 100).toFixed(1)}%`;
}

// ── Update Predictions Table ──────────────────────────────────────
function updateTable(preds) {
  if (preds.length === 0) {
    predBody.innerHTML = '<tr><td colspan="4" class="empty-row">No predictions yet</td></tr>';
    return;
  }
  predBody.innerHTML = preds.map(({ className, probability }, i) => `
    <tr class="rank-${i + 1}">
      <td>${i + 1}</td>
      <td>${className}</td>
      <td>${(probability * 100).toFixed(2)}%</td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar" style="width:${(probability * 100).toFixed(1)}%"></div>
        </div>
      </td>
    </tr>
  `).join('');
}

// ── Init ──────────────────────────────────────────────────────────
loadModel();