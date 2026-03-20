/**
 * Assignment 3 — main.js
 * CDNs: GSAP 3 + ScrollTrigger, Chart.js 4, Highlight.js 11
 */
'use strict';

gsap.registerPlugin(ScrollTrigger);

/* 1. Syntax highlighting */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* 2. Hero entrance */
gsap.timeline({ defaults: { ease: 'power3.out' } })
  .from('.hero-pill',    { y: 18, opacity: 0, duration: .5 })
  .from('.hero-left h1', { y: 50, opacity: 0, duration: .7 }, '-=.3')
  .from('.hdesc',        { y: 20, opacity: 0, duration: .55 }, '-=.35')
  .from('.hero-btn-row', { y: 16, opacity: 0, duration: .4 }, '-=.3')
  .from('.pl-step',      { x: 40, opacity: 0, duration: .45, stagger: .12 }, '-=.4')
  .from('.pl-arrow',     { opacity: 0, duration: .3, stagger: .1 }, '-=.3');

/* 3. Insights cards */
gsap.from('.ins-card', {
  scrollTrigger: { trigger: '.insights-grid', start: 'top 80%' },
  y: 28, opacity: 0, duration: .5, stagger: .1, ease: 'power2.out'
});

/* 4. Phase tabs */
document.querySelectorAll('.ptab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.ptab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.phase-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panel = document.getElementById(tab.dataset.phase);
    panel.classList.add('active');
    gsap.from(panel, { opacity: 0, y: 12, duration: .25, ease: 'power2.out' });
  });
});

/* 5. Navbar */
ScrollTrigger.create({
  start: 60,
  onEnter:     () => document.querySelector('.nav').style.background = 'rgba(5,3,15,.95)',
  onLeaveBack: () => document.querySelector('.nav').style.background = 'rgba(5,3,15,.78)',
});

/* 6. Performance Chart (Chart.js) */
const PHASES     = ['3 Classes', '5 Classes', '7 Classes'];
const BASE_ACC   = [94.2, 91.7, 89.4];
const BASE_F1    = [93.8, 90.9, 88.6];

const perfCtx = document.getElementById('perfChart').getContext('2d');
const perfChart = new Chart(perfCtx, {
  type: 'line',
  data: {
    labels: PHASES,
    datasets: [
      {
        label: 'Val Accuracy (%)',
        data: [...BASE_ACC],
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167,139,250,.12)',
        borderWidth: 2.5,
        tension: .35,
        pointBackgroundColor: '#7c3aed',
        pointRadius: 7,
        pointHoverRadius: 10,
        fill: true,
      },
      {
        label: 'Macro F1 (%)',
        data: [...BASE_F1],
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,.08)',
        borderWidth: 2.5,
        tension: .35,
        borderDash: [6, 4],
        pointBackgroundColor: '#f59e0b',
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
      legend: {
        labels: { color: '#a78bfa', font: { weight: '600', size: 13 } }
      },
      tooltip: {
        backgroundColor: '#0a0720',
        borderColor: 'rgba(124,58,237,.3)',
        borderWidth: 1,
        titleColor: '#ede9fe',
        bodyColor: '#a78bfa',
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` }
      }
    },
    scales: {
      y: {
        min: 80, max: 100,
        ticks: { color: '#3730a3', callback: v => v + '%', stepSize: 5 },
        grid:  { color: 'rgba(124,58,237,.08)' },
      },
      x: {
        ticks: { color: '#a78bfa', font: { weight: '600' } },
        grid:  { color: 'rgba(124,58,237,.06)' },
      }
    }
  }
});

/* 7. Simulation controls update */
function computeAccuracies(ipp, overlap, carry) {
  const dataMult   = Math.min(1, ipp / 300);
  const carryBonus = carry ? 0.025 : 0;
  const acc3 = Math.min(.98, .78 + dataMult * .14 + carryBonus);
  const penA  = overlap * 0.004;
  const acc5 = Math.min(acc3, acc3 - .022 - penA + (carry ? .008 : 0));
  const acc7 = Math.min(acc5, acc5 - .020 - penA * .8 + (carry ? .006 : 0));
  return [acc3, acc5, acc7].map(v => Math.max(.55, v) * 100);
}

function updateSummary(accs) {
  const drops = [
    (accs[0] - accs[1]).toFixed(1),
    (accs[1] - accs[2]).toFixed(1),
  ];
  document.getElementById('summaryBox').innerHTML = `
    <div class="sb-row"><span class="sb-l">3-Class Acc</span><span class="sb-v">${accs[0].toFixed(1)}%</span></div>
    <div class="sb-row"><span class="sb-l">5-Class Acc</span><span class="sb-v">${accs[1].toFixed(1)}%</span></div>
    <div class="sb-row"><span class="sb-l">7-Class Acc</span><span class="sb-v">${accs[2].toFixed(1)}%</span></div>
    <div class="sb-row"><span class="sb-l">Drop 3→5</span><span class="sb-v" style="color:#f87171">−${drops[0]}%</span></div>
    <div class="sb-row"><span class="sb-l">Drop 5→7</span><span class="sb-v" style="color:#f87171">−${drops[1]}%</span></div>
  `;
}

function updateChart() {
  const ipp     = +document.getElementById('r-ipp').value;
  const overlap = +document.getElementById('r-overlap').value;
  const carry   = document.getElementById('ck-carry').checked;

  document.getElementById('o-ipp').value = ipp;

  const accs = computeAccuracies(ipp, overlap, carry);
  const f1s  = accs.map(a => +(a - 0.4 - Math.random() * 0.8).toFixed(1));

  gsap.to(perfChart.data.datasets[0].data, {
    endArray: accs,
    duration: .5,
    ease: 'power2.out',
    onUpdate: () => perfChart.update('none'),
  });

  gsap.to(perfChart.data.datasets[1].data, {
    endArray: f1s,
    duration: .5,
    ease: 'power2.out',
    onUpdate: () => perfChart.update('none'),
  });

  updateSummary(accs);
}

['r-ipp','r-overlap','ck-carry'].forEach(id => {
  document.getElementById(id).addEventListener('input',  updateChart);
  document.getElementById(id).addEventListener('change', updateChart);
});
updateChart();

/* 8. Phase ladder hover */
document.querySelectorAll('.pl-step').forEach((step, i) => {
  step.addEventListener('mouseenter', () => {
    gsap.to(step, { x: 6, duration: .2, ease: 'power2.out' });
  });
  step.addEventListener('mouseleave', () => {
    gsap.to(step, { x: 0, duration: .3, ease: 'power2.out' });
  });
});