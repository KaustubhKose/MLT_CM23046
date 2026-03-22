/**
 * Assignment 2 — main.js
 * CDNs: GSAP 3 + ScrollTrigger · Chart.js 4 · Highlight.js 11
 */
'use strict';

gsap.registerPlugin(ScrollTrigger);

/* ── 1. Syntax highlighting ── */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* ── 2. Hero entrance ── */
gsap.timeline({ defaults: { ease: 'power3.out' } })
  .from('.a-tag',        { y: 18, opacity: 0, duration: .5 })
  .from('.hero-left h1', { y: 55, opacity: 0, duration: .75 }, '-=.3')
  .from('.hero-p',       { y: 20, opacity: 0, duration: .55 }, '-=.35')
  .from('.met-box',      { y: 20, opacity: 0, duration: .4, stagger: .1, ease: 'back.out(1.4)' }, '-=.25')
  .from('.matrix-art',   { x: 40, opacity: 0, duration: .6 }, '-=.55');

/* ── 3. Navbar ── */
ScrollTrigger.create({
  start: 70,
  onEnter:     () => document.getElementById('nav').classList.add('scrolled'),
  onLeaveBack: () => document.getElementById('nav').classList.remove('scrolled'),
});

/* ── 4. Code card reveals ── */
gsap.from('.code-card', {
  scrollTrigger: { trigger: '#metrics-code', start: 'top 80%' },
  y: 30, opacity: 0, duration: .6, stagger: .15,
});

/* ── 5. Interpretation cards ── */
gsap.from('.ig-card', {
  scrollTrigger: { trigger: '.interp-grid', start: 'top 80%' },
  y: 26, opacity: 0, duration: .5, stagger: .1,
});

/* ── 6. Copy buttons ── */
document.querySelectorAll('.cpbtn').forEach(btn => {
  btn.addEventListener('click', () => {
    const pre  = document.getElementById(btn.dataset.sid);
    const code = pre?.querySelector('code');
    if (!code) return;
    navigator.clipboard.writeText(code.innerText).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
    });
  });
});

/* ── 7. Live confusion matrix ── */
const CLASSES   = ['Apple', 'Banana', 'Mango'];
const TOTAL_PER = 100;

function buildMatrix(accPct, hardIdx) {
  const diag = Math.round(accPct / 100 * TOTAL_PER);
  const off  = TOTAL_PER - diag;
  return CLASSES.map((_, r) =>
    CLASSES.map((__, c) => {
      if (r === c) return diag;
      const factor = r === hardIdx ? 1.5 : 0.7;
      return Math.max(0, Math.round((off / 2) * factor + (Math.random() - .5) * 3));
    })
  );
}

function renderMatrix(matrix) {
  const body = document.getElementById('cmBody');
  body.innerHTML = '';
  CLASSES.forEach((rowLabel, r) => {
    const rl = document.createElement('div');
    rl.className = 'cmv-rl-cell';
    rl.textContent = rowLabel;
    body.appendChild(rl);
    matrix[r].forEach((val, c) => {
      const cell = document.createElement('div');
      cell.className = 'cm-cell ' + (r === c ? 'diag' : 'off');
      if (r === c) {
        const intensity = Math.min(1, val / TOTAL_PER);
        cell.style.background   = `rgba(171,71,188,${.1 + intensity * .35})`;
        cell.style.borderColor  = `rgba(171,71,188,${.2 + intensity * .55})`;
      } else {
        const intensity = Math.min(1, val / 20);
        cell.style.background   = `rgba(255,255,255,${intensity * .05})`;
        cell.style.color        = `rgba(206,147,216,${.2 + intensity * .65})`;
      }
      cell.textContent = val;
      body.appendChild(cell);
    });
  });
}

function calcMetrics(matrix) {
  const n  = matrix.length;
  const tp = CLASSES.map((_, i) => matrix[i][i]);
  const fp = CLASSES.map((_, i) => matrix.reduce((s, r) => s + r[i], 0) - matrix[i][i]);
  const fn = CLASSES.map((_, i) => matrix[i].reduce((a, b) => a + b, 0) - matrix[i][i]);
  const total   = matrix.flat().reduce((a, b) => a + b, 0);
  const correct = tp.reduce((a, b) => a + b, 0);
  const acc  = (correct / total * 100).toFixed(1);
  const prec = (tp.map((t, i) => t / (t + fp[i] + 1e-9)).reduce((a, b) => a + b, 0) / n * 100).toFixed(1);
  const rec  = (tp.map((t, i) => t / (t + fn[i] + 1e-9)).reduce((a, b) => a + b, 0) / n * 100).toFixed(1);
  const f1   = (2 * +prec * +rec / (+prec + +rec + 1e-9)).toFixed(1);
  const perF1 = tp.map((t, i) => (2 * t / (2 * t + fp[i] + fn[i] + 1e-9) * 100).toFixed(1));
  return { acc, prec, rec, f1, perF1 };
}

/* ── 8. F1 Bar Chart ── */
const f1Ctx = document.getElementById('f1Bar').getContext('2d');
const f1Chart = new Chart(f1Ctx, {
  type: 'bar',
  data: {
    labels: CLASSES,
    datasets: [{
      label: 'F1 Score (%)',
      data: [93, 91, 94],
      backgroundColor: ['rgba(206,147,216,.7)', 'rgba(244,143,177,.7)', 'rgba(255,213,79,.7)'],
      borderColor:     ['#ce93d8', '#f48fb1', '#ffd54f'],
      borderWidth: 1.5,
      borderRadius: 7,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Per-Class F1 Score', color: '#ce93d8', font: { size: 14, weight: '700' } }
    },
    scales: {
      y: { min: 50, max: 100, ticks: { color: '#4a1a5e', callback: v => v + '%' }, grid: { color: 'rgba(171,71,188,.08)' } },
      x: { ticks: { color: '#ce93d8', font: { weight: '600' } }, grid: { display: false } }
    }
  }
});

function updateLiveMetrics(m) {
  document.getElementById('liveMetrics').innerHTML = [
    ['Accuracy',  m.acc  + '%'],
    ['Precision', m.prec + '%'],
    ['Recall',    m.rec  + '%'],
    ['F1 Score',  m.f1   + '%'],
  ].map(([l, v]) => `<div class="lm-row"><span class="lm-label">${l}</span><span class="lm-val">${v}</span></div>`).join('');

  f1Chart.data.datasets[0].data = m.perF1.map(Number);
  f1Chart.update();

  // Animate hero pills
  gsap.to('#m-acc',  { textContent: m.acc,  duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#m-prec', { textContent: m.prec, duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#m-rec',  { textContent: m.rec,  duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#m-f1',   { textContent: m.f1,   duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
}

function updateCM() {
  const accPct  = +document.getElementById('r-acc').value;
  const hardIdx = +document.getElementById('sel-hard').value;
  document.getElementById('o-acc').value = accPct + '%';
  const matrix = buildMatrix(accPct, hardIdx);
  renderMatrix(matrix);
  updateLiveMetrics(calcMetrics(matrix));
}

document.getElementById('r-acc').addEventListener('input', updateCM);
document.getElementById('sel-hard').addEventListener('change', updateCM);
updateCM();