/**
 * Assignment 1: Train a Sentiment Classifier on Positive/Negative Sentences
 *
 * Pipeline:
 *  1. Built-in dataset of 40 labelled sentences (20 pos, 20 neg)
 *  2. Simple tokenizer → integer sequences → padded tensors
 *  3. Embedding → Dense → Sigmoid model trained in-browser
 *  4. Loss/accuracy chart drawn on canvas each epoch
 *  5. Model exported to window.sentimentModel for Assignment 2 to consume
 */

// ── Dataset ──────────────────────────────────────────────────────
const DATASET = {
  positive: [
    "I absolutely love this product, it is fantastic",
    "What a wonderful and joyful experience today",
    "This is the best day of my life, feeling great",
    "Amazing results, I am so happy with the outcome",
    "Excellent quality and superb customer service",
    "I feel incredibly excited and optimistic about this",
    "This made me smile, brilliant work everyone",
    "Outstanding performance, exceeded all expectations",
    "Delightful and heartwarming, highly recommend",
    "Wonderful news, this is truly remarkable and inspiring",
    "I am thrilled with these results, simply perfect",
    "Such a positive and uplifting experience overall",
    "Great job team, impressive and well done",
    "This is beautiful, I enjoy it very much",
    "Fantastic opportunity, feeling motivated and energized",
    "So grateful and appreciative of this kind gesture",
    "Loved every moment, pure joy and satisfaction",
    "Superb craftsmanship and outstanding attention to detail",
    "Genuinely pleased and proud of this achievement",
    "Brilliant solution, makes everything so much easier"
  ],
  negative: [
    "I hate this, it is absolutely terrible and awful",
    "What a horrible and dreadful experience today",
    "This is the worst thing I have ever seen",
    "Terrible results, I am so disappointed and upset",
    "Poor quality and dreadful customer service received",
    "I feel incredibly frustrated and pessimistic about this",
    "This made me angry, pathetic work by everyone",
    "Dreadful performance, failed all expectations completely",
    "Disgusting and upsetting, would never recommend this",
    "Awful news, this is truly shocking and depressing",
    "I am disgusted with these results, simply broken",
    "Such a negative and distressing experience overall",
    "Bad job team, unimpressive and poorly done indeed",
    "This is ugly, I dislike it very much",
    "Terrible opportunity, feeling demotivated and drained",
    "So ungrateful and dismissive of this important matter",
    "Hated every moment, pure misery and dissatisfaction",
    "Poor craftsmanship and dreadful lack of attention",
    "Genuinely annoyed and embarrassed by this failure",
    "Awful solution, makes everything so much harder"
  ]
};

// ── State ─────────────────────────────────────────────────────────
let vocab = {};
let sentences = [];
let labels   = [];
let lossHistory = [];
let accHistory  = [];
let trainedModel = null;

const SEQ_LEN = () => parseInt(document.getElementById('cfg-seqlen').value) || 20;

// ── Tokenizer ─────────────────────────────────────────────────────
function buildVocab(texts) {
  const counts = {};
  texts.forEach(t => t.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).forEach(w => {
    if (w) counts[w] = (counts[w] || 0) + 1;
  }));
  const sorted = Object.entries(counts).sort((a,b) => b[1]-a[1]);
  vocab = { '<PAD>': 0, '<UNK>': 1 };
  sorted.forEach(([w], i) => { vocab[w] = i + 2; });
  return vocab;
}

function encode(text) {
  return text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/)
    .map(w => vocab[w] || 1);
}

function pad(seq, len) {
  if (seq.length >= len) return seq.slice(0, len);
  return [...seq, ...Array(len - seq.length).fill(0)];
}

// ── Prepare Data ──────────────────────────────────────────────────
function prepareData() {
  sentences = [...DATASET.positive, ...DATASET.negative];
  labels    = [
    ...Array(DATASET.positive.length).fill(1),
    ...Array(DATASET.negative.length).fill(0)
  ];
  buildVocab(sentences);
  document.getElementById('cfg-vocab').textContent = Object.keys(vocab).length;
}

// ── Build Model ───────────────────────────────────────────────────
function buildModel(vocabSize, embedDim, hiddenUnits) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim, inputLength: SEQ_LEN(), name: 'Embedding' }));
  model.add(tf.layers.globalAveragePooling1d({ name: 'GlobalAvgPool' }));
  model.add(tf.layers.dense({ units: hiddenUnits, activation: 'relu', name: 'Dense_ReLU' }));
  model.add(tf.layers.dropout({ rate: 0.3, name: 'Dropout' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'Output_Sigmoid' }));
  return model;
}

// ── Train ─────────────────────────────────────────────────────────
async function train() {
  const btnTrain  = document.getElementById('btn-train');
  const statusEl  = document.getElementById('train-status');
  const epochBadge = document.getElementById('epoch-badge');
  const progressFill = document.getElementById('progress-fill');
  const progressPct  = document.getElementById('progress-pct');

  btnTrain.disabled = true;
  statusEl.textContent = 'Preparing data…';
  statusEl.className = 'train-status running';
  lossHistory = []; accHistory = [];

  const epochs     = parseInt(document.getElementById('cfg-epochs').value) || 60;
  const embedDim   = parseInt(document.getElementById('cfg-embed').value)  || 16;
  const hiddenUnits = parseInt(document.getElementById('cfg-units').value) || 32;
  const lr         = parseFloat(document.getElementById('cfg-lr').value)   || 0.005;
  const seqLen     = SEQ_LEN();

  prepareData();
  const vocabSize = Object.keys(vocab).length;

  // Encode + pad
  const xs = sentences.map(s => pad(encode(s), seqLen));
  const xTensor = tf.tensor2d(xs, [xs.length, seqLen], 'int32');
  const yTensor = tf.tensor1d(labels, 'float32');

  const model = buildModel(vocabSize, embedDim, hiddenUnits);
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  renderArch(model, vocabSize, embedDim, hiddenUnits, seqLen);
  document.getElementById('mm-params').textContent = model.countParams().toLocaleString();

  const t0 = performance.now();
  statusEl.textContent = 'Training…';

  await model.fit(xTensor, yTensor, {
    epochs,
    batchSize: 8,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossHistory.push(logs.loss);
        accHistory.push(logs.acc || logs.accuracy || 0);
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        progressFill.style.width = pct + '%';
        progressPct.textContent  = pct + '%';
        epochBadge.textContent   = `Epoch ${epoch + 1} / ${epochs}`;
        drawChart();
        await tf.nextFrame();
      }
    }
  });

  const trainTime = ((performance.now() - t0) / 1000).toFixed(1);
  const finalLoss = lossHistory[lossHistory.length - 1].toFixed(4);
  const finalAcc  = (accHistory[accHistory.length - 1] * 100).toFixed(1);

  document.getElementById('mm-loss').textContent = finalLoss;
  document.getElementById('mm-acc').textContent  = finalAcc + '%';
  document.getElementById('mm-time').textContent = trainTime + 's';

  trainedModel = model;
  window.sentimentModel = { model, vocab, seqLen, encode, pad };

  xTensor.dispose(); yTensor.dispose();

  statusEl.textContent = `✓ Trained — ${finalAcc}% accuracy`;
  statusEl.className = 'train-status ok';
  btnTrain.disabled = false;
}

// ── Loss Chart ────────────────────────────────────────────────────
function drawChart() {
  const canvas = document.getElementById('loss-chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 700;
  const H = 180;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  if (lossHistory.length < 2) return;

  const pad2 = { top: 16, right: 20, bottom: 32, left: 48 };
  const cW = W - pad2.left - pad2.right;
  const cH = H - pad2.top - pad2.bottom;
  const maxLoss = Math.max(...lossHistory, 1);
  const n = lossHistory.length;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const y = pad2.top + cH * (1 - t);
    ctx.beginPath(); ctx.moveTo(pad2.left, y); ctx.lineTo(pad2.left + cW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText((maxLoss * t).toFixed(2), 4, y + 4);
  });

  // Loss line
  function drawLine(data, color) {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad2.left + (i / (n - 1)) * cW;
      const y = pad2.top  + cH * (1 - v / maxLoss);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawLine(lossHistory, '#f43f5e');
  drawLine(accHistory.map(v => v * maxLoss), '#10b981');

  // Legend
  ctx.fillStyle = '#f43f5e'; ctx.fillRect(pad2.left, H - 14, 10, 3);
  ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '10px Sora, sans-serif';
  ctx.fillText('Loss', pad2.left + 14, H - 10);
  ctx.fillStyle = '#10b981'; ctx.fillRect(pad2.left + 60, H - 14, 10, 3);
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.fillText('Accuracy (scaled)', pad2.left + 74, H - 10);
}

// ── Architecture Visualizer ───────────────────────────────────────
function renderArch(model, vocabSize, embedDim, hiddenUnits, seqLen) {
  const container = document.getElementById('arch-viz');
  const layers = [
    { name: 'Input', info: `shape: [${seqLen}]  (integer token IDs)` },
    { name: 'Embedding', info: `${vocabSize} tokens → ${embedDim}D  |  params: ${vocabSize * embedDim}` },
    { name: 'GlobalAvgPool1D', info: `[${seqLen}, ${embedDim}] → [${embedDim}]` },
    { name: 'Dense + ReLU', info: `${embedDim} → ${hiddenUnits} units  |  params: ${embedDim * hiddenUnits + hiddenUnits}` },
    { name: 'Dropout 30%', info: 'regularisation — disabled at inference' },
    { name: 'Dense + Sigmoid', info: `${hiddenUnits} → 1  |  output: P(positive)` },
  ];

  container.innerHTML = '<div class="arch-layers">' +
    layers.map((l, i) => `
      <div class="arch-layer"><span class="arch-layer-name">${l.name}</span><span class="arch-layer-info">${l.info}</span></div>
      ${i < layers.length - 1 ? '<div class="arch-arrow">▼</div>' : ''}
    `).join('') +
  '</div>';
}

// ── Render Dataset Lists ──────────────────────────────────────────
function renderLists() {
  const posList = document.getElementById('pos-list');
  const negList = document.getElementById('neg-list');
  posList.innerHTML = DATASET.positive.map(s => `<li class="pos-item">${s}</li>`).join('');
  negList.innerHTML = DATASET.negative.map(s => `<li class="neg-item">${s}</li>`).join('');
  document.getElementById('dataset-count').textContent =
    `${DATASET.positive.length + DATASET.negative.length} sentences`;
}

// ── Add Custom Sentence ───────────────────────────────────────────
document.getElementById('btn-add').addEventListener('click', () => {
  const input  = document.getElementById('new-sentence');
  const select = document.getElementById('new-label');
  const text   = input.value.trim();
  if (!text) return;
  const label = parseInt(select.value);
  if (label === 1) DATASET.positive.push(text);
  else             DATASET.negative.push(text);
  input.value = '';
  renderLists();
});

document.getElementById('btn-train').addEventListener('click', train);

// ── Init ──────────────────────────────────────────────────────────
renderLists();
prepareData();