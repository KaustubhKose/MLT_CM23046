/**
 * Assignment 1 — main.js
 * CDNs: GSAP 3, Chart.js 4, Highlight.js 11
 */
'use strict';

/* 1. Syntax highlighting */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* 2. GSAP Hero entrance */
gsap.registerPlugin(ScrollTrigger);

gsap.timeline({ defaults: { ease: 'power3.out' } })
  .from('.hero-eyebrow', { y: 20, opacity: 0, duration: .6 })
  .from('.hero-title',   { y: 50, opacity: 0, duration: .7 }, '-=.3')
  .from('.hero-desc',    { y: 24, opacity: 0, duration: .6 }, '-=.35')
  .from('.hero-chips',   { y: 20, opacity: 0, duration: .5 }, '-=.3')
  .from('.stats-row',    { y: 16, opacity: 0, duration: .5 }, '-=.3');

/* 3. Step cards scroll reveal */
gsap.from('.step-card', {
  scrollTrigger: { trigger: '.steps-list', start: 'top 80%' },
  x: -30, opacity: 0, duration: .5, stagger: .12, ease: 'power2.out'
});

/* 4. Info cards reveal */
gsap.from('.info-card', {
  scrollTrigger: { trigger: '.cards-grid', start: 'top 80%' },
  y: 30, opacity: 0, duration: .5, stagger: .1, ease: 'power2.out'
});

/* 5. Navbar scroll glass */
ScrollTrigger.create({
  start: 60,
  onEnter:     () => document.querySelector('.navbar').style.background = 'rgba(3,13,8,.95)',
  onLeaveBack: () => document.querySelector('.navbar').style.background = 'rgba(3,13,8,.75)',
});

/* 6. Chart.js Gauge */
const gaugeCtx = document.getElementById('gauge1').getContext('2d');
const gaugeChart = new Chart(gaugeCtx, {
  type: 'doughnut',
  data: {
    datasets: [{
      data: [0, 100],
      backgroundColor: ['rgba(0,230,118,1)', 'rgba(255,255,255,.05)'],
      borderWidth: 0,
      borderRadius: 6
    }]
  },
  options: {
    cutout: '78%',
    rotation: -90,
    circumference: 180,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    animation: { duration: 700, easing: 'easeInOutQuart' }
  }
});

/* 7. Simulator */
const classes = [
  { name: '🍎 Apple',  color: '#ef9a9a' },
  { name: '🍌 Banana', color: '#ffe082' },
  { name: '🥭 Mango',  color: '#ffcc80' },
];

const classScoresEl = document.getElementById('classScores');
classScoresEl.innerHTML = classes.map((c, i) => `
  <div class="cs-row">
    <span class="cs-name">${c.name}</span>
    <div class="cs-bar-bg"><div class="cs-bar-fill" id="csb-${i}" style="background:${c.color}"></div></div>
    <span class="cs-pct" id="csp-${i}">0%</span>
  </div>
`).join('');

function estAcc(imgs, epochs, ft, bn) {
  const d = Math.min(1, imgs / 280);
  const e = Math.min(1, epochs / 20);
  const f = ft ? .06 : 0;
  const b = bn ? .02 : 0;
  return Math.min(.98, Math.max(.58, .74 + d*.12 + e*.07 + f + b));
}

function estTime(imgs, epochs, ft) {
  const steps = Math.ceil(imgs * 3 * .8 / 32);
  const sec = steps * epochs * .013 + (ft ? steps * 8 * .019 : 0);
  const m = sec / 60;
  return m < 1 ? '< 1 min' : `~${Math.round(m)} min`;
}

let curAcc = 0;

function update() {
  const imgs   = +document.getElementById('r-imgs').value;
  const ep     = +document.getElementById('r-ep').value;
  const ft     = document.getElementById('ck-ft').checked;
  const bn     = document.getElementById('ck-bn').checked;

  document.getElementById('o-imgs').value = imgs;
  document.getElementById('o-ep').value   = ep;

  const acc    = estAcc(imgs, ep, ft, bn);
  const accPct = Math.round(acc * 100);
  const loss   = (1 - acc + .05 + Math.random() * .03).toFixed(3);

  gsap.to({ v: curAcc }, {
    v: accPct, duration: .6, ease: 'power2.out',
    onUpdate: function () {
      document.getElementById('g-acc').textContent = Math.round(this.targets()[0].v);
    }
  });
  curAcc = accPct;

  // gauge colour
  const col = accPct >= 88 ? 'rgba(0,230,118,1)' : accPct >= 78 ? 'rgba(255,193,7,1)' : 'rgba(244,67,54,1)';
  gaugeChart.data.datasets[0].data = [accPct, 100 - accPct];
  gaugeChart.data.datasets[0].backgroundColor[0] = col;
  gaugeChart.update();

  document.getElementById('r-time').textContent  = estTime(imgs, ep, ft);
  document.getElementById('r-loss').textContent  = loss;
  document.getElementById('r-phase').textContent = ft ? 'P1 + Fine-tune' : 'Phase 1 only';

  // Per-class bars
  const scores = classes.map((_, i) => {
    if (i === 0) return acc * 100;
    const spread = (100 - acc * 100) / (classes.length - 1);
    return Math.max(3, spread + (Math.random() - .5) * 8);
  });
  classes.forEach((_, i) => {
    const v = scores[i].toFixed(1);
    document.getElementById(`csb-${i}`).style.width = v + '%';
    document.getElementById(`csp-${i}`).textContent = v + '%';
  });
}

['r-imgs','r-ep','ck-ft','ck-bn'].forEach(id => {
  document.getElementById(id).addEventListener('input', update);
  document.getElementById(id).addEventListener('change', update);
});
update();

/* 8. Counter animation on stat boxes */
document.querySelectorAll('.stat-n').forEach(el => {
  const id   = el.id;
  const targets = { 'cnt-classes': 3, 'cnt-acc': 94, 'cnt-params': 3.4 };
  const t = targets[id];
  if (!t) return;
  const isDecimal = String(t).includes('.');
  gsap.from({ v: 0 }, {
    v: t, duration: 2, delay: .8, ease: 'power2.out',
    onUpdate: function () {
      el.textContent = isDecimal
        ? this.targets()[0].v.toFixed(1)
        : Math.round(this.targets()[0].v);
    }
  });
});