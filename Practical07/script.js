/**
 * Model Deployment — Save, Load & Predict
 * ─────────────────────────────────────────────────────────────────
 * Full pipeline:
 *  1. TRAIN  — synthetic house-price regression model (TF.js)
 *  2. SAVE   — IndexedDB / download / localStorage
 *  3. LOAD   — reload from storage (or uploaded file)
 *  4. PREDICT — run inference on both original + loaded models
 *
 * The "deploy" metaphor: after saving and reloading, the loaded
 * model is treated as if it came from a server — completely
 * independent from the in-memory trained model.
 */

// ── DOM Shortcuts ─────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const setStatus = (id, text, cls) => {
  const el = $(id);
  el.textContent = text;
  el.className   = 'action-status ' + (cls || '');
};
const setBadge = (id, text, cls) => {
  const el = $(id);
  el.textContent = text;
  el.className   = 'card-badge ' + (cls || '');
};
const setGlobal = (text, cls) => {
  $('global-label').textContent = text;
  $('global-status').className  = 'status-pill ' + (cls || '');
  $('global-dot').className     = 'status-dot '  + (cls || '');
};
const setPipeStep = (id, state) => {
  $(id).classList.remove('active','done');
  if (state) $(id).classList.add(state);
};

// ── State ─────────────────────────────────────────────────────────
let originalModel = null;   // trained in Step 1
let loadedModel   = null;   // reloaded in Step 3
let trainedOnce   = false;
let savedOnce     = false;
let lastSaveKey   = '';
let lossHistory   = [];
let predLog       = [];
let logSeq        = 0;

// ── Normalisation constants (fixed for the synthetic dataset) ─────
const INPUT_MIN  = 100;
const INPUT_MAX  = 10000;
const OUTPUT_MIN = 10000;
const OUTPUT_MAX = 1800000;

const normalise   = (val, min, max) => (val - min) / (max - min);
const denormalise = (val, min, max) => val * (max - min) + min;

// ── Generate synthetic dataset ────────────────────────────────────
function generateData(n) {
  const sizes  = [];
  const prices = [];
  for (let i = 0; i < n; i++) {
    const size  = INPUT_MIN + Math.random() * (INPUT_MAX - INPUT_MIN);
    const noise = (Math.random() - 0.5) * 40000;
    const price = size * 170 + 20000 + noise;
    sizes.push(normalise(size, INPUT_MIN, INPUT_MAX));
    prices.push(normalise(Math.min(OUTPUT_MAX, Math.max(OUTPUT_MIN, price)), OUTPUT_MIN, OUTPUT_MAX));
  }
  return { xs: tf.tensor2d(sizes, [n, 1]), ys: tf.tensor2d(prices, [n, 1]) };
}

// ── Build model ───────────────────────────────────────────────────
function buildModel(units) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units, activation: 'relu', inputShape: [1] }));
  m.add(tf.layers.dense({ units: Math.max(4, units / 2), activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1 }));
  return m;
}

// ══ STEP 1: TRAIN ═════════════════════════════════════════════════
$('btn-train').addEventListener('click', async () => {
  $('btn-train').disabled = true;
  setStatus('train-status', 'Preparing data…', 'busy');
  setGlobal('Training…', 'busy');
  setPipeStep('step-train', 'active');
  lossHistory = [];

  const n      = parseInt($('cfg-samples').value) || 200;
  const units  = parseInt($('cfg-units').value)   || 16;
  const epochs = parseInt($('cfg-epochs').value)  || 80;
  const lr     = parseFloat($('cfg-lr').value)    || 0.01;

  const { xs, ys } = generateData(n);
  const model = buildModel(units);
  model.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError', metrics: ['mae'] });

  $('train-progress-row').style.display = 'flex';
  $('loss-canvas').style.display        = 'block';

  const t0 = performance.now();
  setStatus('train-status', 'Training…', 'busy');

  await model.fit(xs, ys, {
    epochs,
    batchSize: 32,
    validationSplit: 0.1,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossHistory.push({ loss: logs.loss, valLoss: logs.val_loss ?? logs.loss });
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        $('train-fill').style.width   = pct + '%';
        $('train-pct').textContent    = pct + '%';
        drawLossChart();
        await tf.nextFrame();
      }
    }
  });

  const elapsed  = ((performance.now() - t0) / 1000).toFixed(1);
  const evalRes  = model.evaluate(xs, ys, { batchSize: 32 });
  const finalLoss = (await evalRes[0].data())[0];
  const finalMae  = (await evalRes[1].data())[0];
  evalRes.forEach(t => t.dispose());
  xs.dispose(); ys.dispose();

  // Format MAE back to real dollar values
  const maeReal = denormalise(finalMae, OUTPUT_MIN, OUTPUT_MAX) - OUTPUT_MIN;

  $('m-loss').textContent   = finalLoss.toFixed(5);
  $('m-mae').textContent    = '$' + Math.round(maeReal).toLocaleString();
  $('m-params').textContent = model.countParams().toLocaleString();
  $('m-time').textContent   = elapsed + 's';
  $('train-metrics').style.display = 'grid';

  originalModel = model;
  trainedOnce   = true;

  setBadge('train-badge', '✓ Trained', 'done');
  setStatus('train-status', `Done — ${(finalLoss * 100).toFixed(3)}% loss`, 'ok');
  setGlobal('Model trained', 'ok');
  setPipeStep('step-train', 'done');
  setPipeStep('step-save', 'active');

  $('btn-train').disabled = false;
  $('btn-save').disabled  = false;
  setStatus('save-status', 'Ready to save', '');
  renderArchitecture(model);
});

// ── Loss Chart ────────────────────────────────────────────────────
function drawLossChart() {
  const canvas = $('loss-canvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 600;
  const H = 140;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  if (lossHistory.length < 2) return;
  const pad = { t: 12, r: 16, b: 28, l: 50 };
  const cW  = W - pad.l - pad.r;
  const cH  = H - pad.t - pad.b;
  const maxL = Math.max(...lossHistory.map(d => d.loss), 0.001);
  const n    = lossHistory.length;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  [0,.25,.5,.75,1].forEach(t => {
    const y = pad.t + cH * (1 - t);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '9px DM Mono, monospace';
    ctx.fillText((maxL * t).toFixed(4), 2, y + 3);
  });

  const drawLine = (data, color1, color2) => {
    if (data.length < 2) return;
    const grad = ctx.createLinearGradient(pad.l, 0, pad.l + cW, 0);
    grad.addColorStop(0, color1); grad.addColorStop(1, color2);
    ctx.strokeStyle = grad; ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.l + (i / (n - 1)) * cW;
      const y = pad.t + cH * (1 - v / maxL);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  drawLine(lossHistory.map(d => d.loss),    '#00d4aa', '#38bdf8');
  drawLine(lossHistory.map(d => d.valLoss), '#7c3aed', '#f472b6');

  // Legend
  const leg = (x, color, label) => {
    ctx.fillStyle = color; ctx.fillRect(x, H - 12, 14, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '9px DM Mono, monospace';
    ctx.fillText(label, x + 18, H - 9);
  };
  leg(pad.l, '#00d4aa', 'Train loss');
  leg(pad.l + 90, '#7c3aed', 'Val loss');
}

// ── Architecture display ──────────────────────────────────────────
function renderArchitecture(model) {
  const container = $('arch-layers');
  const layers = model.layers.map(l => ({
    name: l.name,
    type: l.getClassName(),
    outputShape: JSON.stringify(l.outputShape),
    params: l.countParams(),
  }));

  container.innerHTML = layers.map((l, i) => `
    <div class="arch-layer">
      <span class="al-name">${l.type}</span>
      <span class="al-info">out: ${l.outputShape} · params: ${l.params}</span>
    </div>
    ${i < layers.length - 1 ? '<div class="al-arrow">▼</div>' : ''}
  `).join('');
}

// ══ STEP 2: SAVE ══════════════════════════════════════════════════
$('btn-save').addEventListener('click', async () => {
  if (!originalModel) return;
  $('btn-save').disabled = true;
  setStatus('save-status', 'Saving…', 'busy');
  setGlobal('Saving…', 'busy');

  const dest = document.querySelector('input[name="save-dest"]:checked').value;
  const key  = 'tfjs-model-deploy-demo';
  lastSaveKey = key;

  try {
    let savePath = '';
    if (dest === 'indexeddb') {
      await originalModel.save(`indexeddb://${key}`);
      savePath = `indexeddb://${key}`;
    } else if (dest === 'downloads') {
      await originalModel.save('downloads://model-deploy-demo');
      savePath = 'downloads (model.json + weights.bin)';
    } else {
      await originalModel.save(`localstorage://${key}`);
      savePath = `localstorage://${key}`;
    }

    savedOnce = true;
    $('save-path').textContent = savePath;
    $('save-time').textContent = new Date().toLocaleTimeString();
    $('save-key').textContent  = key;
    $('save-result').style.display = 'flex';

    setBadge('save-badge', '✓ Saved', 'done');
    setStatus('save-status', 'Saved successfully', 'ok');
    setGlobal('Model saved', 'ok');
    setPipeStep('step-save', 'done');
    setPipeStep('step-load', 'active');

    $('btn-load').disabled = false;
    setStatus('load-status', 'Ready to reload', '');
  } catch (e) {
    setStatus('save-status', 'Error: ' + e.message, 'err');
    setGlobal('Save error', 'error');
    setBadge('save-badge', 'Error', 'error');
  }
  $('btn-save').disabled = false;
});

// ── Load source toggle ────────────────────────────────────────────
$('load-source').addEventListener('change', () => {
  const v = $('load-source').value;
  $('file-upload-wrap').style.display = v === 'file' ? 'block' : 'none';
});

// ══ STEP 3: LOAD ══════════════════════════════════════════════════
$('btn-load').addEventListener('click', async () => {
  $('btn-load').disabled = true;
  setStatus('load-status', 'Loading…', 'busy');
  setGlobal('Loading model…', 'busy');

  try {
    const src = $('load-source').value;

    if (src === 'file') {
      const file = $('model-file-input').files[0];
      if (!file) throw new Error('No file selected — choose a model.json file');
      loadedModel = await tf.loadLayersModel(tf.io.browserFiles([file]));
    } else {
      // Try IndexedDB first, fall back to localstorage
      try {
        loadedModel = await tf.loadLayersModel(`indexeddb://${lastSaveKey || 'tfjs-model-deploy-demo'}`);
      } catch {
        loadedModel = await tf.loadLayersModel(`localstorage://${lastSaveKey || 'tfjs-model-deploy-demo'}`);
      }
    }

    // Recompile for evaluation
    loadedModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Show architecture of loaded model
    renderArchitecture(loadedModel);
    $('model-summary').style.display = 'block';

    setBadge('load-badge', '✓ Loaded', 'done');
    setStatus('load-status', 'Loaded successfully', 'ok');
    setGlobal('Model ready', 'ok');
    setPipeStep('step-load', 'done');
    setPipeStep('step-predict', 'active');

    $('btn-predict').disabled = false;
    $('btn-batch').disabled   = false;
    $('pred-input').disabled  = false;
    $('batch-input').disabled = false;
    setBadge('predict-badge', 'Ready', 'done');
    setStatus('load-status', 'Reload complete — run predictions ↓', 'ok');
  } catch (e) {
    setStatus('load-status', 'Error: ' + e.message, 'err');
    setGlobal('Load error', 'error');
    setBadge('load-badge', 'Error', 'error');
    console.error(e);
  }
  $('btn-load').disabled = false;
});

// ══ STEP 4: PREDICT ═══════════════════════════════════════════════
async function predictSingle(inputSqFt) {
  const normIn = normalise(inputSqFt, INPUT_MIN, INPUT_MAX);
  const tensor = tf.tensor2d([[normIn]]);

  let origPrice = null, loadPrice = null;

  if (originalModel) {
    const out = originalModel.predict(tensor);
    origPrice = denormalise((await out.data())[0], OUTPUT_MIN, OUTPUT_MAX);
    out.dispose();
  }
  if (loadedModel) {
    const out = loadedModel.predict(tensor);
    loadPrice = denormalise((await out.data())[0], OUTPUT_MIN, OUTPUT_MAX);
    out.dispose();
  }
  tensor.dispose();
  return { orig: origPrice, loaded: loadPrice };
}

const fmt = v => v !== null ? '$' + Math.round(v).toLocaleString() : '—';

$('btn-predict').addEventListener('click', async () => {
  const input = parseFloat($('pred-input').value);
  if (isNaN(input) || input < 100) return;

  $('btn-predict').disabled = true;
  const { orig, loaded } = await predictSingle(input);
  $('btn-predict').disabled = false;

  $('pred-result').style.display  = 'flex';
  $('pred-original').textContent  = fmt(orig);
  $('pred-loaded').textContent    = fmt(loaded);

  const diff = orig !== null && loaded !== null
    ? Math.abs(orig - loaded)
    : null;
  $('pred-diff').textContent = diff !== null
    ? '$' + Math.round(diff).toLocaleString() + (diff < 1000 ? ' ✓ match' : ' ⚠ differs')
    : '—';

  setPipeStep('step-predict', 'done');
  addLogRow(input, orig, loaded);
  updatePredChart();
});

// Batch
$('btn-batch').addEventListener('click', async () => {
  const raw = $('batch-input').value.trim();
  if (!raw) return;
  const values = raw.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
  if (!values.length) return;

  $('btn-batch').disabled = true;
  $('batch-results').innerHTML = '';

  for (const v of values) {
    const { orig, loaded } = await predictSingle(v);
    const chip = document.createElement('div');
    chip.className = 'batch-chip';
    chip.innerHTML = `
      <span class="bc-input">${v.toLocaleString()} sqft</span>
      <span class="bc-val">${fmt(loaded ?? orig)}</span>
    `;
    $('batch-results').appendChild(chip);
    addLogRow(v, orig, loaded);
  }

  $('btn-batch').disabled = false;
  updatePredChart();
});

// ── Log row ───────────────────────────────────────────────────────
function addLogRow(input, orig, loaded) {
  logSeq++;
  const diff = orig && loaded ? Math.abs(orig - loaded) : null;
  const empty = $('log-body').querySelector('.log-empty');
  if (empty) empty.parentElement.remove();

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${logSeq}</td>
    <td>${input.toLocaleString()}</td>
    <td>${fmt(orig)}</td>
    <td>${fmt(loaded)}</td>
    <td style="color:${diff !== null && diff < 2000 ? '#34d399' : '#f87171'}">${diff !== null ? '$' + Math.round(diff).toLocaleString() : '—'}</td>
    <td>${new Date().toLocaleTimeString()}</td>
  `;
  $('log-body').prepend(tr);
  while ($('log-body').children.length > 50) $('log-body').removeChild($('log-body').lastChild);
  predLog.unshift({ input, orig, loaded });
}

$('btn-clear-log').addEventListener('click', () => {
  $('log-body').innerHTML = '<tr><td colspan="6" class="log-empty">No predictions yet</td></tr>';
  predLog = []; logSeq = 0;
  $('pred-chart').style.display = 'none';
  $('chart-placeholder').style.display = 'block';
});

// ── Prediction Chart ──────────────────────────────────────────────
function updatePredChart() {
  if (predLog.length < 2) return;
  $('pred-chart').style.display    = 'block';
  $('chart-placeholder').style.display = 'none';

  const canvas = $('pred-chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 800;
  const H = 200;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const data  = [...predLog].reverse();
  const allV  = data.flatMap(d => [d.orig, d.loaded].filter(Boolean));
  const maxY  = Math.max(...allV) * 1.05;
  const minY  = Math.min(...allV) * 0.95;
  const pad   = { t: 14, r: 20, b: 32, l: 70 };
  const cW    = W - pad.l - pad.r;
  const cH    = H - pad.t - pad.b;
  const n     = data.length;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  [0,.25,.5,.75,1].forEach(t => {
    const y = pad.t + cH * (1 - t);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '9px DM Mono,monospace';
    const val = minY + (maxY - minY) * t;
    ctx.fillText('$' + (val / 1000).toFixed(0) + 'k', 2, y + 3);
  });

  const plotLine = (values, c1, c2) => {
    const pts = values.map((v, i) => ({
      x: pad.l + (i / (n - 1)) * cW,
      y: pad.t + cH * (1 - (v - minY) / (maxY - minY)),
    }));
    const grad = ctx.createLinearGradient(pad.l, 0, pad.l + cW, 0);
    grad.addColorStop(0, c1); grad.addColorStop(1, c2);
    ctx.strokeStyle = grad; ctx.lineWidth = 2.5;
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();
    pts.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = c1; ctx.fill();
    });
  };

  const origV   = data.map(d => d.orig).filter(Boolean);
  const loadedV = data.map(d => d.loaded).filter(Boolean);
  if (origV.length === n)   plotLine(origV,   '#00d4aa', '#38bdf8');
  if (loadedV.length === n) plotLine(loadedV, '#7c3aed', '#f472b6');

  // Legend
  const leg = (x, c, label) => {
    ctx.fillStyle = c; ctx.fillRect(x, H - 12, 14, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '9px DM Mono,monospace';
    ctx.fillText(label, x + 18, H - 9);
  };
  leg(pad.l, '#00d4aa', 'Original model');
  leg(pad.l + 100, '#7c3aed', 'Loaded model');
}

// ── Init ──────────────────────────────────────────────────────────
setGlobal('Ready to train', 'ok');
setPipeStep('step-train', 'active');