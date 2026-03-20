/**
 * Assignment 2 — main.js
 * CDNs: GSAP 3 + ScrollTrigger, Chart.js 4, Highlight.js 11
 */
'use strict';

gsap.registerPlugin(ScrollTrigger);

/* 1. Syntax highlighting */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* 2. GSAP Hero entrance */
const htl = gsap.timeline({ defaults: { ease: 'power3.out' } });
htl.from('.a-label',     { y: 18, opacity: 0, duration: .5 })
   .from('h1',           { y: 50, opacity: 0, duration: .7 }, '-=.25')
   .from('.sub',         { y: 20, opacity: 0, duration: .55 }, '-=.35')
   .from('.m-pill',      { y: 20, opacity: 0, duration: .4, stagger: .1, ease: 'back.out(1.4)' }, '-=.2')
   .from('.hero-art',    { x: 40, opacity: 0, duration: .6 }, '-=.5');

/* 3. Code cards scroll reveal */
gsap.from('.code-block', {
  scrollTrigger: { trigger: '#evaluate', start: 'top 80%' },
  y: 30, opacity: 0, duration: .6, stagger: .15
});

gsap.from('.icard', {
  scrollTrigger: { trigger: '.interp-grid', start: 'top 80%' },
  y: 24, opacity: 0, duration: .5, stagger: .1
});

/* 4. Copy buttons */
document.querySelectorAll('.cb-copy').forEach(btn => {
  btn.addEventListener('click', () => {
    const pre  = document.getElementById(btn.dataset.id);
    const code = pre?.querySelector('code');
    if (!code) return;
    navigator.clipboard.writeText(code.innerText).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
    });
  });
});

/* 5. Live confusion matrix simulator */
const CLASS_NAMES = ['Apple', 'Banana', 'Mango'];
const TOTAL_PER_CLASS = 100;

function buildMatrix(accPct, hardIdx) {
  const acc   = accPct / 100;
  const diag  = Math.round(acc * TOTAL_PER_CLASS);
  const off   = TOTAL_PER_CLASS - diag;
  const hard  = Math.ceil(off * 1.4);
  const easy  = Math.floor(off * 0.6);

  const matrix = Array.from({ length: 3 }, (_, r) =>
    Array.from({ length: 3 }, (_, c) => {
      if (r === c) return diag;
      const isHardRow = r === hardIdx;
      const split     = isHardRow ? hard : easy;
      return Math.max(0, Math.round(split / 2 + (Math.random() - .5) * 2));
    })
  );
  return matrix;
}

function renderCM(matrix) {
  const body = document.getElementById('cmBody');
  body.innerHTML = '';
  CLASS_NAMES.forEach((rowLabel, r) => {
    // Row label
    const rl = document.createElement('div');
    rl.className = 'cm-row-l';
    rl.textContent = rowLabel;
    body.appendChild(rl);

    matrix[r].forEach((val, c) => {
      const cell = document.createElement('div');
      cell.className = 'cm-cell ' + (r === c ? 'diag' : 'off');
      // Intensity for diagonal
      if (r === c) {
        const intensity = Math.min(1, val / TOTAL_PER_CLASS);
        cell.style.background = `rgba(233,30,99,${0.1 + intensity * .4})`;
        cell.style.borderColor = `rgba(233,30,99,${0.2 + intensity * .6})`;
      } else {
        const intensity = Math.min(1, val / 20);
        cell.style.background = `rgba(255,255,255,${intensity * .06})`;
        cell.style.color       = `rgba(244,143,177,${0.2 + intensity * .6})`;
      }
      cell.textContent = val;
      body.appendChild(cell);
    });
  });
}

function calcMetrics(matrix) {
  const n = matrix.length;
  let tp = [], fp = [], fn = [];
  for (let i = 0; i < n; i++) {
    tp.push(matrix[i][i]);
    fp.push(matrix.reduce((s, r) => s + r[i], 0) - matrix[i][i]);
    fn.push(matrix[i].reduce((a, b) => a + b, 0) - matrix[i][i]);
  }
  const total   = matrix.flat().reduce((a, b) => a + b, 0);
  const correct = tp.reduce((a, b) => a + b, 0);
  const acc     = (correct / total * 100).toFixed(1);
  const prec    = (tp.map((t, i) => t / (t + fp[i] + 1e-9)).reduce((a, b) => a + b, 0) / n * 100).toFixed(1);
  const rec     = (tp.map((t, i) => t / (t + fn[i] + 1e-9)).reduce((a, b) => a + b, 0) / n * 100).toFixed(1);
  const f1      = (2 * +prec * +rec / (+prec + +rec + 1e-9)).toFixed(1);
  const perF1   = tp.map((t, i) => (2 * t / (2 * t + fp[i] + fn[i] + 1e-9) * 100).toFixed(1));
  return { acc, prec, rec, f1, perF1 };
}

/* F1 Bar chart */
const f1Ctx = document.getElementById('f1Chart').getContext('2d');
const f1Chart = new Chart(f1Ctx, {
  type: 'bar',
  data: {
    labels: CLASS_NAMES,
    datasets: [{
      label: 'F1 Score (%)',
      data: [93, 91, 94],
      backgroundColor: ['rgba(244,143,177,.7)', 'rgba(206,147,216,.7)', 'rgba(255,171,145,.7)'],
      borderColor:     ['#f48fb1', '#ce93d8', '#ffab91'],
      borderWidth: 1.5,
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Per-Class F1 Score', color: '#f48fb1', font: { size: 14, weight: '700' } }
    },
    scales: {
      y: {
        min: 50, max: 100,
        ticks: { color: '#7b1f3a', callback: v => v + '%' },
        grid: { color: 'rgba(233,30,99,.08)' }
      },
      x: { ticks: { color: '#f48fb1' }, grid: { display: false } }
    }
  }
});

function updateMetricsUI(m) {
  const list = document.getElementById('metricsList');
  list.innerHTML = [
    { l: 'Accuracy',  v: m.acc  + '%' },
    { l: 'Precision', v: m.prec + '%' },
    { l: 'Recall',    v: m.rec  + '%' },
    { l: 'F1 Score',  v: m.f1   + '%' },
  ].map(({ l, v }) => `
    <div class="ml-row">
      <span class="ml-label">${l}</span>
      <span class="ml-val">${v}</span>
    </div>
  `).join('');

  f1Chart.data.datasets[0].data = m.perF1.map(Number);
  f1Chart.update();
}

let currentMatrix = [];

function update() {
  const accPct  = +document.getElementById('r-acc').value;
  const hardIdx = +document.getElementById('sel-hard').value;
  document.getElementById('o-acc2').value = accPct;

  currentMatrix = buildMatrix(accPct, hardIdx);
  renderCM(currentMatrix);
  const m = calcMetrics(currentMatrix);
  updateMetricsUI(m);

  // Animate hero metric pills with GSAP
  gsap.to('#mp-acc',  { textContent: m.acc,  duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#mp-prec', { textContent: m.prec, duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#mp-rec',  { textContent: m.rec,  duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
  gsap.to('#mp-f1',   { textContent: m.f1,   duration: .4, snap: { textContent: .1 }, ease: 'power1.out' });
}

document.getElementById('r-acc').addEventListener('input', update);
document.getElementById('sel-hard').addEventListener('change', update);
update();

/* 6. Navbar scroll */
ScrollTrigger.create({
  start: 60,
  onEnter:     () => document.querySelector('.navbar').style.background = 'rgba(13,4,8,.95)',
  onLeaveBack: () => document.querySelector('.navbar').style.background = 'rgba(13,4,8,.8)',
});