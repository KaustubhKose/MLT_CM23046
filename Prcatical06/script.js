/**
 * PoseNet Human Pose Detection
 * ─────────────────────────────────────────────────────────────────
 * Features:
 *  1. Load PoseNet (MobileNet v1 backbone, quantBytes 2)
 *  2. Access webcam via getUserMedia
 *  3. Run real-time multi-pose estimation on every animation frame
 *  4. Draw skeleton (keypoints + connections) on canvas overlay
 *  5. HUD with FPS, latency, pose count
 *  6. Live keypoint table with confidence colour-coding
 *  7. Body angle calculator (elbow, knee, shoulder, hip)
 *  8. Snapshot capture (pose + skeleton burned into a single image)
 */

// ── DOM References ────────────────────────────────────────────────
const video         = document.getElementById('webcam');
const canvas        = document.getElementById('pose-canvas');
const ctx           = canvas.getContext('2d');
const videoFrame    = document.getElementById('video-frame');
const scanLine      = document.getElementById('scan-line');

const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const btnSnapshot   = document.getElementById('btn-snapshot');
const btnClearSnaps = document.getElementById('btn-clear-snaps');
const confRange     = document.getElementById('conf-range');
const confVal       = document.getElementById('conf-val');
const maxPosesEl    = document.getElementById('max-poses');

// Header status
const modelDot      = document.getElementById('model-dot');
const modelLabel    = document.getElementById('model-label');
const camDot        = document.getElementById('cam-dot');
const camLabel      = document.getElementById('cam-label');

// HUD
const hudFps        = document.getElementById('hud-fps');
const hudPoses      = document.getElementById('hud-poses');
const hudBottom     = document.getElementById('hud-bottom');

// Metrics
const mFps          = document.getElementById('m-fps');
const mPoses        = document.getElementById('m-poses');
const mLatency      = document.getElementById('m-latency');
const mKeypoints    = document.getElementById('m-keypoints');

// Data panels
const kpCountBadge  = document.getElementById('kp-count-badge');
const keypointBody  = document.getElementById('keypoint-body');
const anglesList    = document.getElementById('angles-list');
const snapshotGrid  = document.getElementById('snapshot-grid');

// ── State ─────────────────────────────────────────────────────────
let net        = null;
let stream     = null;
let animId     = null;
let running    = false;
let snapCount  = 0;

// FPS tracking
let frameCount = 0;
let lastFpsTick = 0;
let smoothFps  = 0;

// ── PoseNet Skeleton Definition ───────────────────────────────────
const CONNECTED_PAIRS = [
  // Torso
  ['leftShoulder','rightShoulder'],
  ['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],
  ['leftHip','rightHip'],
  // Left arm
  ['leftShoulder','leftElbow'],
  ['leftElbow','leftWrist'],
  // Right arm
  ['rightShoulder','rightElbow'],
  ['rightElbow','rightWrist'],
  // Left leg
  ['leftHip','leftKnee'],
  ['leftKnee','leftAnkle'],
  // Right leg
  ['rightHip','rightKnee'],
  ['rightKnee','rightAnkle'],
  // Face
  ['nose','leftEye'],
  ['nose','rightEye'],
  ['leftEye','leftEar'],
  ['rightEye','rightEar'],
];

// Keypoint display name map
const KP_NAMES = {
  nose:'Nose', leftEye:'L.Eye', rightEye:'R.Eye',
  leftEar:'L.Ear', rightEar:'R.Ear',
  leftShoulder:'L.Shoulder', rightShoulder:'R.Shoulder',
  leftElbow:'L.Elbow', rightElbow:'R.Elbow',
  leftWrist:'L.Wrist', rightWrist:'R.Wrist',
  leftHip:'L.Hip', rightHip:'R.Hip',
  leftKnee:'L.Knee', rightKnee:'R.Knee',
  leftAnkle:'L.Ankle', rightAnkle:'R.Ankle',
};

// ── 1. Load PoseNet ───────────────────────────────────────────────
async function loadModel() {
  modelDot.className = 'status-dot loading';
  modelLabel.textContent = 'LOADING MODEL…';
  try {
    net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 640, height: 480 },
      multiplier: 0.75,
      quantBytes: 2,
    });
    modelDot.className = 'status-dot ready';
    modelLabel.textContent = 'MODEL READY';
  } catch (e) {
    modelDot.className = 'status-dot error';
    modelLabel.textContent = 'MODEL ERROR';
    console.error(e);
  }
}

// ── 2. Start Webcam ───────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (!net) await loadModel();
  try {
    camDot.className = 'status-dot loading';
    camLabel.textContent = 'REQUESTING…';
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(r => (video.onloadedmetadata = r));

    running = true;
    btnStart.disabled    = true;
    btnStop.disabled     = false;
    btnSnapshot.disabled = false;

    camDot.className   = 'status-dot active';
    camLabel.textContent = 'CAMERA LIVE';
    videoFrame.classList.add('active');
    scanLine.classList.add('active');
    hudBottom.innerHTML = '';

    lastFpsTick = performance.now();
    requestAnimationFrame(detectionLoop);
  } catch (e) {
    camDot.className   = 'status-dot error';
    camLabel.textContent = 'ACCESS DENIED';
    hudBottom.innerHTML = '<span class="hud-idle-msg">⚠ CAMERA PERMISSION DENIED</span>';
  }
});

// ── 3. Stop ───────────────────────────────────────────────────────
btnStop.addEventListener('click', stop);

function stop() {
  running = false;
  cancelAnimationFrame(animId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  btnStart.disabled    = false;
  btnStop.disabled     = true;
  btnSnapshot.disabled = true;
  camDot.className     = 'status-dot';
  camLabel.textContent = 'CAMERA OFF';
  videoFrame.classList.remove('active');
  scanLine.classList.remove('active');
  hudBottom.innerHTML  = '<span class="hud-idle-msg">▶ START CAMERA TO BEGIN DETECTION</span>';

  // Clear live data
  keypointBody.innerHTML = '<tr><td colspan="4" class="table-empty">Waiting for pose…</td></tr>';
  anglesList.innerHTML   = '<p class="panel-placeholder">Detect a pose to calculate joint angles.</p>';
  [mFps, mPoses, mLatency, mKeypoints].forEach(el => el.textContent = '—');
  kpCountBadge.textContent = '0 / 17';
  hudFps.textContent   = 'FPS: —';
  hudPoses.textContent = 'POSES: 0';
}

// ── 4. Detection Loop ─────────────────────────────────────────────
async function detectionLoop(now) {
  if (!running) return;

  // Sync canvas to video
  if (video.videoWidth) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  // FPS
  frameCount++;
  const fpsDelta = now - lastFpsTick;
  if (fpsDelta >= 600) {
    smoothFps = Math.round(frameCount / (fpsDelta / 1000));
    frameCount = 0;
    lastFpsTick = now;
    mFps.textContent  = smoothFps;
    hudFps.textContent = `FPS: ${smoothFps}`;
  }

  if (video.readyState === 4) {
    const t0 = performance.now();
    const maxPoses = parseInt(maxPosesEl.value) || 1;
    const minConf  = parseInt(confRange.value) / 100;

    let poses = [];
    try {
      if (maxPoses === 1) {
        const pose = await net.estimateSinglePose(video, { flipHorizontal: false });
        poses = [pose];
      } else {
        poses = await net.estimateMultiplePoses(video, {
          flipHorizontal: false,
          maxDetections: maxPoses,
          scoreThreshold: minConf,
          nmsRadius: 20,
        });
      }
    } catch (e) { /* skip frame */ }

    const latency = Math.round(performance.now() - t0);

    // Filter by min confidence
    const validPoses = poses.filter(p => p.score >= minConf * 0.7);

    // Draw
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    validPoses.forEach((pose, i) => drawPose(pose, i, minConf));

    // Update UI
    mPoses.textContent    = validPoses.length;
    mLatency.textContent  = latency;
    hudPoses.textContent  = `POSES: ${validPoses.length}`;

    if (validPoses.length > 0) {
      const kps = validPoses[0].keypoints.filter(k => k.score >= minConf);
      mKeypoints.textContent = kps.length;
      kpCountBadge.textContent = `${kps.length} / 17`;
      updateKeypointTable(validPoses[0].keypoints, minConf);
      updateAngles(validPoses[0].keypoints);
    } else {
      mKeypoints.textContent = '0';
      kpCountBadge.textContent = '0 / 17';
    }
  }

  animId = requestAnimationFrame(detectionLoop);
}

// ── 5. Draw Skeleton ──────────────────────────────────────────────
const POSE_COLORS = ['#00ff8c', '#00e5ff', '#ffb300', '#ff3d5a', '#c084fc'];

function drawPose(pose, index, minConf) {
  const color = POSE_COLORS[index % POSE_COLORS.length];
  const kpMap = {};
  pose.keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  const scaleX = canvas.width  / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;

  // Draw connections
  CONNECTED_PAIRS.forEach(([a, b]) => {
    const kpA = kpMap[a], kpB = kpMap[b];
    if (!kpA || !kpB) return;
    if (kpA.score < minConf || kpB.score < minConf) return;

    const ax = kpA.position.x * scaleX;
    const ay = kpA.position.y * scaleY;
    const bx = kpB.position.x * scaleX;
    const by = kpB.position.y * scaleY;

    // Gradient line
    const grad = ctx.createLinearGradient(ax, ay, bx, by);
    grad.addColorStop(0, color + 'cc');
    grad.addColorStop(1, '#00e5ffcc');

    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    ctx.stroke();
    ctx.shadowBlur  = 0;
  });

  // Draw keypoints
  pose.keypoints.forEach(kp => {
    if (kp.score < minConf) return;
    const x = kp.position.x * scaleX;
    const y = kp.position.y * scaleY;
    const r = kp.score > 0.8 ? 5 : 3.5;

    // Outer ring
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, Math.PI * 2);
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12;
    ctx.fill();
    ctx.shadowBlur  = 0;
  });
}

// ── 6. Keypoint Table ─────────────────────────────────────────────
function updateKeypointTable(keypoints, minConf) {
  const rows = keypoints.map(kp => {
    const conf    = (kp.score * 100).toFixed(0);
    const confClass = kp.score >= 0.75 ? 'conf-high' : kp.score >= 0.4 ? 'conf-med' : 'conf-low';
    const display = kp.score >= minConf * 0.5;
    return `
      <tr style="opacity:${display ? 1 : 0.3}">
        <td>${KP_NAMES[kp.part] || kp.part}</td>
        <td>${Math.round(kp.position.x)}</td>
        <td>${Math.round(kp.position.y)}</td>
        <td class="${confClass}">${conf}%</td>
      </tr>
    `;
  }).join('');
  keypointBody.innerHTML = rows;
}

// ── 7. Body Angles ────────────────────────────────────────────────
function angleBetween(a, b, c) {
  // Angle at point B formed by A-B-C
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.sqrt(ab.x**2 + ab.y**2) * Math.sqrt(cb.x**2 + cb.y**2);
  if (mag === 0) return 0;
  return Math.round((Math.acos(Math.min(1, Math.max(-1, dot / mag))) * 180) / Math.PI);
}

const ANGLE_DEFS = [
  { name:'L.Elbow',    parts:['leftShoulder','leftElbow','leftWrist'] },
  { name:'R.Elbow',    parts:['rightShoulder','rightElbow','rightWrist'] },
  { name:'L.Shoulder', parts:['leftHip','leftShoulder','leftElbow'] },
  { name:'R.Shoulder', parts:['rightHip','rightShoulder','rightElbow'] },
  { name:'L.Knee',     parts:['leftHip','leftKnee','leftAnkle'] },
  { name:'R.Knee',     parts:['rightHip','rightKnee','rightAnkle'] },
  { name:'L.Hip',      parts:['leftShoulder','leftHip','leftKnee'] },
  { name:'R.Hip',      parts:['rightShoulder','rightHip','rightKnee'] },
];

function updateAngles(keypoints) {
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });
  const minC = parseInt(confRange.value) / 100;

  const items = ANGLE_DEFS.map(def => {
    const [a, b, c] = def.parts.map(p => kpMap[p]);
    if (!a || !b || !c) return null;
    if (a.score < minC || b.score < minC || c.score < minC) return null;
    const angle = angleBetween(a.position, b.position, c.position);
    const pct   = (angle / 180) * 100;
    return `
      <div class="angle-row">
        <span class="angle-name">${def.name}</span>
        <div class="angle-bar-wrap"><div class="angle-bar" style="width:${pct}%"></div></div>
        <span class="angle-val">${angle}°</span>
      </div>
    `;
  }).filter(Boolean);

  if (items.length === 0) {
    anglesList.innerHTML = '<p class="panel-placeholder">Not enough keypoints visible.</p>';
  } else {
    anglesList.innerHTML = items.join('');
  }
}

// ── 8. Snapshot ───────────────────────────────────────────────────
btnSnapshot.addEventListener('click', () => {
  if (!running) return;

  // Composite video + skeleton into one image
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width  = canvas.width  || 640;
  snapCanvas.height = canvas.height || 480;
  const sCtx = snapCanvas.getContext('2d');
  sCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
  sCtx.drawImage(canvas, 0, 0);

  const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.85);
  snapCount++;

  const placeholder = snapshotGrid.querySelector('.panel-placeholder');
  if (placeholder) placeholder.remove();

  const item  = document.createElement('div');
  item.className = 'snap-item';
  const img  = document.createElement('img');
  img.src    = dataUrl;
  img.alt    = `Pose snapshot ${snapCount}`;
  const lbl  = document.createElement('div');
  lbl.className = 'snap-label';
  lbl.textContent = `#${String(snapCount).padStart(3,'0')}`;
  item.appendChild(img);
  item.appendChild(lbl);

  // Click to download
  item.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href     = dataUrl;
    a.download = `posenet-snapshot-${snapCount}.jpg`;
    a.click();
  });

  snapshotGrid.prepend(item);
});

// Clear snapshots
btnClearSnaps.addEventListener('click', () => {
  snapshotGrid.innerHTML = '<p class="panel-placeholder">No snapshots yet.</p>';
  snapCount = 0;
});

// ── Confidence slider ─────────────────────────────────────────────
confRange.addEventListener('input', () => {
  confVal.textContent = confRange.value + '%';
});

// ── Init ──────────────────────────────────────────────────────────
loadModel();