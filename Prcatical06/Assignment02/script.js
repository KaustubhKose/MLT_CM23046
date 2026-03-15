/**
 * Assignment 2: Test the Model on Custom Sentences & Interpret Confidence Scores
 *
 * - Automatically trains the Assignment 1 model on load (reuses script.js)
 * - Allows custom sentence input and shows:
 *     • Animated semicircle gauge (0 = negative, 1 = positive)
 *     • Dual confidence bars (positive / negative %)
 *     • Token heatmap: each word chip coloured by its embedding L2 norm
 * - Keeps a running history log of all analyses
 */

// ── DOM ────────────────────────────────────────────────────────────
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const trainProgWrap = document.getElementById('train-prog-wrap');
const trainProgFill = document.getElementById('train-prog-fill');
const btnAnalyse   = document.getElementById('btn-analyse');
const inputEl      = document.getElementById('input-sentence');
const resultCard   = document.getElementById('result-card');
const historyList  = document.getElementById('history-list');

// ── Auto-train on load ─────────────────────────────────────────────
window.addEventListener('load', async () => {
  statusDot.className = 'status-dot loading';
  statusText.textContent = 'Training model…';
  trainProgWrap.style.display = 'block';

  // Patch the progress fill to also update our mini bar
  const origFill = document.getElementById('progress-fill');
  if (origFill) {
    const observer = new MutationObserver(() => {
      trainProgFill.style.width = origFill.style.width;
    });
    observer.observe(origFill, { attributes: true, attributeFilter: ['style'] });
  }

  await train(); // from assignment1/script.js

  trainProgWrap.style.display = 'none';
  statusDot.className = 'status-dot ready';
  statusText.textContent = 'Model ready — start analysing sentences';
  inputEl.disabled = false;
  btnAnalyse.disabled = false;
});

// ── Analyse ────────────────────────────────────────────────────────
btnAnalyse.addEventListener('click', analyse);
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); analyse(); } });

// Example chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    inputEl.value = chip.dataset.text;
    if (!btnAnalyse.disabled) analyse();
  });
});

async function analyse() {
  const sm = window.sentimentModel;
  if (!sm) return;

  const sentence = inputEl.value.trim();
  if (!sentence) return;

  const { model, vocab, seqLen, encode, pad } = sm;
  const tokens   = sentence.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).filter(Boolean);
  const encoded  = tokens.map(w => vocab[w] || 1);
  const padded   = pad(encoded, seqLen);
  const xTensor  = tf.tensor2d([padded], [1, seqLen], 'int32');
  const prediction = model.predict(xTensor);
  const score    = (await prediction.data())[0];
  xTensor.dispose(); prediction.dispose();

  const isPos    = score >= 0.5;
  const posScore = (score * 100).toFixed(1);
  const negScore = ((1 - score) * 100).toFixed(1);

  renderResult(sentence, score, isPos, posScore, negScore, tokens, encoded, sm);
  addHistory(sentence, isPos, score);
}

// ── Render Result ──────────────────────────────────────────────────
async function renderResult(sentence, score, isPos, posScore, negScore, tokens, encoded, sm) {
  resultCard.style.display = 'block';

  // Label badge
  const badge = document.getElementById('result-label-badge');
  if (isPos) {
    badge.textContent = '✦ Positive';
    badge.style.cssText = 'color:#6ee7b7;border-color:rgba(110,231,183,0.35);background:rgba(110,231,183,0.08)';
  } else {
    badge.textContent = '✦ Negative';
    badge.style.cssText = 'color:#fda4af;border-color:rgba(253,164,175,0.35);background:rgba(253,164,175,0.08)';
  }

  // Gauge — needle rotates from -90° (left=neg) to +90° (right=pos)
  const angle = -90 + (score * 180);
  document.getElementById('gauge-needle').setAttribute('transform', `rotate(${angle}, 100, 100)`);
  document.getElementById('gauge-value').textContent = (score * 100).toFixed(0) + '%';

  // Gauge arc fill
  const arcLen = 283; // π * r (approx)
  const filled = score * arcLen;
  const fillEl = document.getElementById('gauge-fill');
  fillEl.setAttribute('stroke-dashoffset', arcLen - filled);
  fillEl.setAttribute('stroke', isPos ? 'url(#pos-grad)' : 'url(#neg-grad)');

  // Confidence bars
  document.getElementById('pos-bar').style.width = posScore + '%';
  document.getElementById('neg-bar').style.width = negScore + '%';
  document.getElementById('pos-pct').textContent = posScore + '%';
  document.getElementById('neg-pct').textContent = negScore + '%';

  // Token heatmap
  await renderTokenHeatmap(tokens, encoded, sm, isPos);
}

// ── Token Heatmap ──────────────────────────────────────────────────
async function renderTokenHeatmap(tokens, encoded, sm, isPos) {
  const { model, seqLen } = sm;
  const tokenMap = document.getElementById('token-map');
  tokenMap.innerHTML = '';

  // Get embedding weights
  const embLayer = model.layers.find(l => l.name === 'Embedding');
  if (!embLayer) return;
  const embWeights = embLayer.getWeights()[0]; // shape [vocabSize, embedDim]

  // For each token, get its embedding vector and compute L2 norm
  const norms = [];
  for (const id of encoded.slice(0, seqLen)) {
    const vec  = tf.slice(embWeights, [id, 0], [1, -1]);
    const norm = (await tf.norm(vec).data())[0];
    norms.push(norm);
    vec.dispose();
  }
  embWeights.dispose();

  const maxNorm = Math.max(...norms, 1);

  tokens.slice(0, seqLen).forEach((token, i) => {
    const intensity = norms[i] / maxNorm;
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.textContent = token;

    if (isPos) {
      const g = Math.round(80 + intensity * 150);
      chip.style.background = `rgba(16,${g},100,${0.12 + intensity * 0.3})`;
      chip.style.borderColor = `rgba(52,211,153,${0.15 + intensity * 0.5})`;
      chip.style.color = `rgba(110,231,183,${0.4 + intensity * 0.6})`;
    } else {
      const r = Math.round(150 + intensity * 100);
      chip.style.background = `rgba(${r},30,60,${0.12 + intensity * 0.3})`;
      chip.style.borderColor = `rgba(244,63,94,${0.15 + intensity * 0.5})`;
      chip.style.color = `rgba(253,164,175,${0.4 + intensity * 0.6})`;
    }

    chip.title = `"${token}" · norm: ${norms[i].toFixed(3)}`;
    tokenMap.appendChild(chip);
  });
}

// ── History ────────────────────────────────────────────────────────
function addHistory(sentence, isPos, score) {
  const placeholder = historyList.querySelector('.placeholder-text');
  if (placeholder) placeholder.remove();

  const item = document.createElement('div');
  item.className = 'hist-item';
  item.innerHTML = `
    <span class="hist-tag ${isPos ? 'pos' : 'neg'}">${isPos ? 'POS' : 'NEG'}</span>
    <span class="hist-sentence">${sentence.length > 80 ? sentence.slice(0,80) + '…' : sentence}</span>
    <span class="hist-conf">${(score * 100).toFixed(1)}%</span>
  `;
  historyList.prepend(item);
}

document.getElementById('btn-clear-hist').addEventListener('click', () => {
  historyList.innerHTML = '<p class="placeholder-text">History cleared.</p>';
});