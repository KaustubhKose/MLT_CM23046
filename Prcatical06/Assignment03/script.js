/**
 * Assignment 3: Compare RNN (LSTM) vs Dense Network for Sentiment Classification
 *
 * Both models use:
 *  - Same dataset (40 sentences, pos/neg)
 *  - Same tokenizer and vocabulary
 *  - Same training procedure
 *
 * Dense: Embedding → GlobalAvgPool1D → Dense(ReLU) → Dropout → Sigmoid
 * LSTM:  Embedding → LSTM            → Dense(ReLU) → Dropout → Sigmoid
 *
 * Outputs:
 *  - Individual training progress bars per model
 *  - Overlay loss chart (Dense amber vs LSTM teal)
 *  - Head-to-head comparison table (accuracy, loss, time, params)
 *  - Live side-by-side inference on custom sentences
 */

// ── Dataset (same as Assignment 1) ────────────────────────────────
const DATASET = {
  positive: [
    "I absolutely love this product it is fantastic","What a wonderful and joyful experience today",
    "This is the best day of my life feeling great","Amazing results I am so happy with the outcome",
    "Excellent quality and superb customer service","I feel incredibly excited and optimistic about this",
    "This made me smile brilliant work everyone","Outstanding performance exceeded all expectations",
    "Delightful and heartwarming highly recommend","Wonderful news this is truly remarkable and inspiring",
    "I am thrilled with these results simply perfect","Such a positive and uplifting experience overall",
    "Great job team impressive and well done","This is beautiful I enjoy it very much",
    "Fantastic opportunity feeling motivated and energized","So grateful and appreciative of this kind gesture",
    "Loved every moment pure joy and satisfaction","Superb craftsmanship and outstanding attention to detail",
    "Genuinely pleased and proud of this achievement","Brilliant solution makes everything so much easier"
  ],
  negative: [
    "I hate this it is absolutely terrible and awful","What a horrible and dreadful experience today",
    "This is the worst thing I have ever seen","Terrible results I am so disappointed and upset",
    "Poor quality and dreadful customer service received","I feel incredibly frustrated and pessimistic about this",
    "This made me angry pathetic work by everyone","Dreadful performance failed all expectations completely",
    "Disgusting and upsetting would never recommend this","Awful news this is truly shocking and depressing",
    "I am disgusted with these results simply broken","Such a negative and distressing experience overall",
    "Bad job team unimpressive and poorly done indeed","This is ugly I dislike it very much",
    "Terrible opportunity feeling demotivated and drained","So ungrateful and dismissive of this important matter",
    "Hated every moment pure misery and dissatisfaction","Poor craftsmanship and dreadful lack of attention",
    "Genuinely annoyed and embarrassed by this failure","Awful solution makes everything so much harder"
  ]
};

// ── Tokenizer ─────────────────────────────────────────────────────
let vocab = {};
const SEQ_LEN = 20;

function buildVocab(texts) {
  const counts = {};
  texts.forEach(t => t.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).forEach(w => {
    if (w) counts[w] = (counts[w]||0)+1;
  }));
  const sorted = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  vocab = {'<PAD>':0,'<UNK>':1};
  sorted.forEach(([w],i) => { vocab[w] = i+2; });
}

function encode(text) {
  return text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).map(w=>vocab[w]||1);
}
function pad(seq,len) {
  if (seq.length>=len) return seq.slice(0,len);
  return [...seq,...Array(len-seq.length).fill(0)];
}

// ── Prepare tensors ───────────────────────────────────────────────
let xTensor, yTensor, sentences, labels;
let vocabSize;

function prepareData() {
  sentences = [...DATASET.positive,...DATASET.negative];
  labels    = [...Array(DATASET.positive.length).fill(1),...Array(DATASET.negative.length).fill(0)];
  buildVocab(sentences);
  vocabSize = Object.keys(vocab).length;
  const xs = sentences.map(s=>pad(encode(s),SEQ_LEN));
  xTensor = tf.tensor2d(xs,[xs.length,SEQ_LEN],'int32');
  yTensor = tf.tensor1d(labels,'float32');
}

// ── Model builders ────────────────────────────────────────────────
function buildDenseModel(embedDim, hiddenUnits) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({inputDim:vocabSize,outputDim:embedDim,inputLength:SEQ_LEN,name:'emb'}));
  m.add(tf.layers.globalAveragePooling1d({name:'gap'}));
  m.add(tf.layers.dense({units:hiddenUnits,activation:'relu',name:'dense_relu'}));
  m.add(tf.layers.dropout({rate:0.3,name:'drop'}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid',name:'out'}));
  return m;
}

function buildRNNModel(embedDim, lstmUnits) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({inputDim:vocabSize,outputDim:embedDim,inputLength:SEQ_LEN,name:'emb_rnn'}));
  m.add(tf.layers.lstm({units:lstmUnits,name:'lstm'}));
  m.add(tf.layers.dense({units:Math.max(8,lstmUnits/2),activation:'relu',name:'dense_relu_rnn'}));
  m.add(tf.layers.dropout({rate:0.3,name:'drop_rnn'}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid',name:'out_rnn'}));
  return m;
}

// ── Results store ─────────────────────────────────────────────────
const results = { dense: null, rnn: null };
const lossHist = { dense: [], rnn: [] };
const accHist  = { dense: [], rnn: [] };
window.trainedModels = {};

// ── Train a model ─────────────────────────────────────────────────
async function trainModel(type) {
  const isRNN  = type === 'rnn';
  const fillId = isRNN ? 'rnn-fill' : 'dense-fill';
  const pctId  = isRNN ? 'rnn-pct'  : 'dense-pct';
  const btnId  = isRNN ? 'btn-train-rnn' : 'btn-train-dense';
  const epochs  = parseInt(document.getElementById(isRNN ? 'rnn-epochs' : 'dense-epochs').value) || 60;
  const embed   = parseInt(document.getElementById(isRNN ? 'rnn-embed'  : 'dense-embed').value)  || 16;
  const units   = parseInt(document.getElementById(isRNN ? 'rnn-units'  : 'dense-units').value)  || 32;
  const lr      = 0.005;

  document.getElementById(btnId).disabled = true;
  lossHist[type] = []; accHist[type] = [];

  if (!xTensor) prepareData();

  const model = isRNN ? buildRNNModel(embed,units) : buildDenseModel(embed,units);
  model.compile({ optimizer:tf.train.adam(lr), loss:'binaryCrossentropy', metrics:['accuracy'] });

  document.getElementById(isRNN?'r-params':'d-params').textContent = model.countParams().toLocaleString();

  const t0 = performance.now();

  await model.fit(xTensor, yTensor, {
    epochs, batchSize: 8, shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossHist[type].push(logs.loss);
        accHist[type].push(logs.acc||logs.accuracy||0);
        const pct = Math.round(((epoch+1)/epochs)*100);
        document.getElementById(fillId).style.width = pct+'%';
        document.getElementById(pctId).textContent  = pct+'%';
        drawCompareChart();
        await tf.nextFrame();
      }
    }
  });

  const elapsed = ((performance.now()-t0)/1000).toFixed(1);
  const finalAcc  = (accHist[type][accHist[type].length-1]*100).toFixed(1);
  const finalLoss = lossHist[type][lossHist[type].length-1].toFixed(4);

  if (isRNN) {
    document.getElementById('r-acc').textContent  = finalAcc+'%';
    document.getElementById('r-loss').textContent = finalLoss;
    document.getElementById('r-time').textContent = elapsed+'s';
  } else {
    document.getElementById('d-acc').textContent  = finalAcc+'%';
    document.getElementById('d-loss').textContent = finalLoss;
    document.getElementById('d-time').textContent = elapsed+'s';
  }

  results[type] = { acc:parseFloat(finalAcc), loss:parseFloat(finalLoss), time:parseFloat(elapsed), params:model.countParams() };
  window.trainedModels[type] = { model, vocab, seqLen:SEQ_LEN, encode, pad };

  document.getElementById(btnId).disabled = false;
  updateCompareTable();

  if (results.dense && results.rnn) {
    document.getElementById('btn-live-test').disabled = false;
    document.getElementById('both-status').textContent = '✓ Both models trained — live test enabled';
  }
}

// ── Loss Chart ────────────────────────────────────────────────────
function drawCompareChart() {
  const canvas = document.getElementById('compare-chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth||800, H=200;
  canvas.width=W; canvas.height=H;
  ctx.clearRect(0,0,W,H);

  const allLoss = [...lossHist.dense,...lossHist.rnn];
  if (allLoss.length < 2) return;
  const maxLoss = Math.max(...allLoss,1);
  const n = Math.max(lossHist.dense.length,lossHist.rnn.length,1);
  const pad2 = {top:14,right:18,bottom:28,left:44};
  const cW=W-pad2.left-pad2.right, cH=H-pad2.top-pad2.bottom;

  // Grid
  ctx.strokeStyle='rgba(255,255,255,0.05)'; ctx.lineWidth=1;
  [0,.25,.5,.75,1].forEach(t=>{
    const y=pad2.top+cH*(1-t);
    ctx.beginPath();ctx.moveTo(pad2.left,y);ctx.lineTo(pad2.left+cW,y);ctx.stroke();
    ctx.fillStyle='rgba(255,255,255,0.25)';ctx.font='10px JetBrains Mono,monospace';
    ctx.fillText((maxLoss*t).toFixed(2),4,y+4);
  });

  function drawLine(data,grad) {
    if (data.length<2) return;
    const grd = ctx.createLinearGradient(pad2.left,0,pad2.left+cW,0);
    grd.addColorStop(0,grad[0]); grd.addColorStop(1,grad[1]);
    ctx.strokeStyle=grd; ctx.lineWidth=2.5;
    ctx.beginPath();
    data.forEach((v,i)=>{
      const x=pad2.left+(i/(n-1||1))*cW;
      const y=pad2.top+cH*(1-v/maxLoss);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
  }

  drawLine(lossHist.dense,['#f59e0b','#fb923c']);
  drawLine(lossHist.rnn,  ['#14b8a6','#38bdf8']);
}

// ── Compare Table ─────────────────────────────────────────────────
function updateCompareTable() {
  const d = results.dense, r = results.rnn;
  if (!d && !r) return;
  if (!d || !r) return;

  const rows = [
    {
      label:'Final Accuracy',
      dVal: d.acc+'%', rVal: r.acc+'%',
      winner: d.acc>r.acc?'dense':d.acc<r.acc?'rnn':'tie',
      winnerLabel: d.acc>r.acc?'Dense':d.acc<r.acc?'LSTM':'Tie'
    },
    {
      label:'Final Loss',
      dVal: d.loss.toFixed(4), rVal: r.loss.toFixed(4),
      winner: d.loss<r.loss?'dense':d.loss>r.loss?'rnn':'tie',
      winnerLabel: d.loss<r.loss?'Dense':d.loss>r.loss?'LSTM':'Tie'
    },
    {
      label:'Train Time',
      dVal: d.time+'s', rVal: r.time+'s',
      winner: d.time<r.time?'dense':d.time>r.time?'rnn':'tie',
      winnerLabel: d.time<r.time?'Dense (faster)':d.time>r.time?'LSTM (faster)':'Tie'
    },
    {
      label:'Parameter Count',
      dVal: d.params.toLocaleString(), rVal: r.params.toLocaleString(),
      winner: d.params<r.params?'dense':d.params>r.params?'rnn':'tie',
      winnerLabel: d.params<r.params?'Dense (smaller)':d.params>r.params?'LSTM (smaller)':'Tie'
    },
  ];

  document.getElementById('compare-body').innerHTML = rows.map(row=>`
    <tr>
      <td>${row.label}</td>
      <td style="font-family:'JetBrains Mono',monospace">${row.dVal}</td>
      <td style="font-family:'JetBrains Mono',monospace">${row.rVal}</td>
      <td class="${row.winner==='tie'?'winner-tie':row.winner==='dense'?'winner-dense':'winner-rnn'}">${row.winnerLabel}</td>
    </tr>
  `).join('');
}

// ── Live Comparison ───────────────────────────────────────────────
document.getElementById('btn-live-test').addEventListener('click', async () => {
  const text = document.getElementById('live-input').value.trim();
  if (!text) return;
  const dm = window.trainedModels.dense;
  const rm = window.trainedModels.rnn;
  if (!dm || !rm) return;

  async function predict(tm) {
    const encoded = tm.encode(text.toLowerCase().replace(/[^a-z\s]/g,''));
    const padded  = tm.pad(encoded, tm.seqLen);
    const xT = tf.tensor2d([padded],[1,tm.seqLen],'int32');
    const pred = tm.model.predict(xT);
    const score = (await pred.data())[0];
    xT.dispose(); pred.dispose();
    return score;
  }

  const [dScore, rScore] = await Promise.all([predict(dm), predict(rm)]);

  const liveResults = document.getElementById('live-results');
  liveResults.style.display = 'grid';

  function setResult(barId, scoreId, verdictId, score) {
    document.getElementById(barId).style.width = (score*100).toFixed(1)+'%';
    document.getElementById(scoreId).textContent = (score*100).toFixed(1)+'%';
    const vEl = document.getElementById(verdictId);
    vEl.textContent = score>=0.5 ? '✦ Positive' : '✦ Negative';
    vEl.className = 'live-verdict '+(score>=0.5?'pos':'neg');
  }

  setResult('live-dense-bar','live-dense-score','live-dense-verdict',dScore);
  setResult('live-rnn-bar',  'live-rnn-score',  'live-rnn-verdict',  rScore);
});

// ── Button wiring ─────────────────────────────────────────────────
document.getElementById('btn-train-dense').addEventListener('click',()=>trainModel('dense'));
document.getElementById('btn-train-rnn').addEventListener('click',  ()=>trainModel('rnn'));
document.getElementById('btn-train-both').addEventListener('click', async ()=>{
  document.getElementById('btn-train-both').disabled=true;
  document.getElementById('both-status').textContent='Training Dense…';
  await trainModel('dense');
  document.getElementById('both-status').textContent='Training LSTM…';
  await trainModel('rnn');
  document.getElementById('btn-train-both').disabled=false;
});

// Enter key for live test
document.getElementById('live-input').addEventListener('keydown', e=>{
  if (e.key==='Enter') document.getElementById('btn-live-test').click();
});