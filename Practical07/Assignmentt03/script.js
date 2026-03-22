/**
 * Assignment 3 — main.js
 * CDNs: GSAP 3 + ScrollTrigger · Chart.js 4 · Highlight.js 11
 */
'use strict';

gsap.registerPlugin(ScrollTrigger);

/* ── 1. Syntax highlighting ── */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* ── 2. Hero entrance ── */
gsap.timeline({ defaults: { ease: 'power3.out' } })
  .from('.h-eyebrow', { y: 16, opacity: 0, duration: .5 })
  .from('.h-title',   { y: 55, opacity: 0, duration: .75 }, '-=.3')
  .from('.h-desc',    { y: 22, opacity: 0, duration: .55 }, '-=.35')
  .from('.h-btns',    { y: 16, opacity: 0, duration: .45 }, '-=.3')
  .from('.lad-item',  { x: 40, opacity: 0, duration: .45, stagger: .14 }, '-=.45')
  .from('.lad-connector', { opacity: 0, duration: .3, stagger: .1 }, '-=.3');

/* ── 3. Ladder fill animation ── */
ScrollTrigger.create({
  trigger: '.ladder',
  start: 'top 80%',
  onEnter: () => {
    document.querySelectorAll('.lad-fill').forEach((fill, i) => {
      const target = fill.style.width;
      fill.style.width = '0%';
      setTimeout(() => { fill.style.width = target; }, i * 200 + 300);
    });
  }
});

/* ── 4. Insights reveal ── */
gsap.from('.ins-card', {
  scrollTrigger: { trigger: '.ins-grid', start: 'top 80%' },
  y: 26, opacity: 0, duration: .5, stagger: .1,
});

/* ── 5. Navbar scroll ── */
ScrollTrigger.create({
  start: 60,
  onEnter:     () => document.getElementById('nav').classList.add('scrolled'),
  onLeaveBack: () => document.getElementById('nav').classList.remove('scrolled'),
});

/* ── 6. Phase tabs ── */
document.querySelectorAll('.ptab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.ptab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.ph-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panel = document.getElementById(tab.dataset.phase);
    panel.classList.add('active');
    gsap.from(panel, { opacity: 0, y: 10, duration: .25, ease: 'power2.out' });
  });
});

/* ── 7. Performance Chart ── */
const BASE_ACC = [94.2, 91.7, 89.4];
const BASE_F1  = [93.8, 90.9, 88.6];
const PHASES   = ['3 Classes', '5 Classes', '7 Classes'];

const perfCtx = document.getElementById('perfChart').getContext('2d');
const perfChart = new Chart(perfCtx, {
  type: 'line',
  data: {
    labels: PHASES,
    datasets: [
      {
        label: 'Val Accuracy (%)',
        data: [...BASE_ACC],
        borderColor: '#ffb74d',
        backgroundColor: 'rgba(255,183,77,.1)',
        borderWidth: 2.5,
        tension: .35,
        pointBackgroundColor: '#f57c00',
        pointRadius: 7,
        pointHoverRadius: 10,
        fill: true,
      },
      {
        label: 'Macro F1 (%)',
        data: [...BASE_F1],
        borderColor: '#ff8a65',
        backgroundColor: 'rgba(255,138,101,.07)',
        borderWidth: 2.5,
        borderDash: [6, 4],
        tension: .35,
        pointBackgroundColor: '#e64a19',
        pointRadius: 7,
        pointHoverRadius: 10,
        fill: true,
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: { labels: { color: '#ffcc80', font: { weight: '600', size: 13 } } },
      tooltip: {
        backgroundColor: '#120b04',
        borderColor: 'rgba(245,124,0,.3)',
        borderWidth: 1,
        titleColor: '#fff8e1',
        bodyColor: '#ffcc80',
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` }
      }
    },
    scales: {
      y: {
        min: 80, max: 100,
        ticks: { color: '#4e2b00', callback: v => v + '%', stepSize: 5 },
        grid:  { color: 'rgba(245,124,0,.07)' }
      },
      x: {
        ticks: { color: '#ffb74d', font: { weight: '700' } },
        grid:  { color: 'rgba(245,124,0,.05)' }
      }
    }
  }
});

/* ── 8. Simulator controls ── */
function computeAccuracies(ipp, overlap, carry) {
  const dm = Math.min(1, ipp / 300);
  const cb = carry ? .025 : 0;
  const op = overlap * .004;
  const a3 = Math.min(.98, .78 + dm * .14 + cb);
  const a5 = Math.min(a3, a3 - .022 - op + (carry ? .008 : 0));
  const a7 = Math.min(a5, a5 - .020 - op * .8 + (carry ? .006 : 0));
  return [a3, a5, a7].map(v => Math.max(.56, v) * 100);
}

function refreshSummary(accs) {
  document.getElementById('summGrid').innerHTML = [
    ['3-Class Acc', accs[0].toFixed(1) + '%'],
    ['5-Class Acc', accs[1].toFixed(1) + '%'],
    ['7-Class Acc', accs[2].toFixed(1) + '%'],
    ['Drop 3→5', `<span style="color:#f87171">−${(accs[0]-accs[1]).toFixed(1)}%</span>`],
    ['Drop 5→7', `<span style="color:#f87171">−${(accs[1]-accs[2]).toFixed(1)}%</span>`],
  ].map(([l, v]) => `<div class="sg-row"><span class="sg-l">${l}</span><span class="sg-v">${v}</span></div>`).join('');
}

function updateChart() {
  const ipp   = +document.getElementById('r-ipp').value;
  const olp   = +document.getElementById('r-olp').value;
  const carry = document.getElementById('ck-carry').checked;
  document.getElementById('o-ipp').value = ipp;

  const accs = computeAccuracies(ipp, olp, carry);
  const f1s  = accs.map(a => +(a - .4 - Math.random() * .7).toFixed(1));

  // Animate chart data with GSAP-style approach
  const animAcc = perfChart.data.datasets[0].data;
  const animF1  = perfChart.data.datasets[1].data;
  accs.forEach((v, i) => { animAcc[i] = v; });
  f1s.forEach((v,  i) => { animF1[i]  = v; });
  perfChart.update('active');

  refreshSummary(accs);
}

['r-ipp', 'r-olp', 'ck-carry'].forEach(id => {
  document.getElementById(id).addEventListener('input',  updateChart);
  document.getElementById(id).addEventListener('change', updateChart);
});
updateChart();