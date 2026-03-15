/**
 * Assignment 1: Capture Webcam Frames & Classify Using MobileNet
 *
 * Steps:
 *  1. Load MobileNet model via TensorFlow.js
 *  2. Access webcam via getUserMedia
 *  3. On demand, capture a frame from the video into a canvas
 *  4. Pass that canvas to model.classify() to get top-3 predictions
 *  5. Display results and save captured frames with labels
 */

// ── DOM References ──────────────────────────────────────────────
const video       = document.getElementById('webcam');
const canvas      = document.getElementById('frame-canvas');
const btnStart    = document.getElementById('btn-start');
const btnCapture  = document.getElementById('btn-capture');
const btnStop     = document.getElementById('btn-stop');
const resultList  = document.getElementById('result-list');
const statusMsg   = document.getElementById('status-msg');
const framesGrid  = document.getElementById('frames-grid');

const ctx = canvas.getContext('2d');

// ── State ────────────────────────────────────────────────────────
let model  = null;   // MobileNet model instance
let stream = null;   // MediaStream from webcam
let captureCount = 0;

// ── 1. Load MobileNet ────────────────────────────────────────────
async function loadModel() {
  statusMsg.textContent = 'Loading MobileNet model…';
  try {
    // mobilenet.load() fetches weights from TF Hub
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    statusMsg.textContent = 'Model ready. Start your webcam to begin.';
    statusMsg.style.color = '#86efac';
  } catch (err) {
    statusMsg.textContent = 'Error loading model: ' + err.message;
    statusMsg.style.color = '#fca5a5';
  }
}

// ── 2. Start Webcam ──────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => (video.onloadedmetadata = resolve));

    btnStart.disabled   = true;
    btnCapture.disabled = false;
    btnStop.disabled    = false;
    statusMsg.textContent = 'Webcam active. Click "Capture & Classify Frame" to analyse a frame.';
  } catch (err) {
    statusMsg.textContent = 'Camera access denied: ' + err.message;
    statusMsg.style.color = '#fca5a5';
  }
});

// ── 3 & 4. Capture Frame + Classify ─────────────────────────────
btnCapture.addEventListener('click', async () => {
  if (!model) { statusMsg.textContent = 'Model not ready yet.'; return; }

  // Draw current video frame onto canvas
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  statusMsg.textContent = 'Classifying…';
  statusMsg.style.color = '#fbbf24';
  btnCapture.disabled = true;

  try {
    // classify() returns top-N { className, probability } objects
    const predictions = await model.classify(canvas, 3);
    renderResults(predictions);
    saveFrameCard(canvas, predictions[0].className);
  } catch (err) {
    statusMsg.textContent = 'Classification error: ' + err.message;
    statusMsg.style.color = '#fca5a5';
  } finally {
    btnCapture.disabled = false;
  }
});

// ── 5. Render Results ────────────────────────────────────────────
function renderResults(predictions) {
  resultList.innerHTML = '';
  predictions.forEach(({ className, probability }) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="label">${className}</span>
      <span class="prob">${(probability * 100).toFixed(1)}%</span>
    `;
    resultList.appendChild(li);
  });
  statusMsg.textContent = `Top prediction: "${predictions[0].className}"`;
  statusMsg.style.color = '#86efac';
}

// ── Save Frame Card ───────────────────────────────────────────────
function saveFrameCard(sourceCanvas, topLabel) {
  captureCount++;
  const card = document.createElement('div');
  card.className = 'frame-card';

  const img = document.createElement('img');
  img.src = sourceCanvas.toDataURL('image/jpeg', 0.8);

  const label = document.createElement('div');
  label.className = 'frame-label';
  label.textContent = `#${captureCount} – ${topLabel}`;

  card.appendChild(img);
  card.appendChild(label);
  framesGrid.prepend(card);
}

// ── Stop Webcam ──────────────────────────────────────────────────
btnStop.addEventListener('click', () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  stream = null;
  btnStart.disabled   = false;
  btnCapture.disabled = true;
  btnStop.disabled    = true;
  statusMsg.textContent = 'Webcam stopped.';
  statusMsg.style.color = '#6b7280';
});

// ── Init ─────────────────────────────────────────────────────────
loadModel();