/**
 * Assignment 1 — main.js
 * CDNs: GSAP 3 + ScrollTrigger · Chart.js 4 · Highlight.js 11
 */
'use strict';

gsap.registerPlugin(ScrollTrigger);

/* ── 1. Syntax highlighting ── */
document.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));

/* ── 2. Hero entrance ── */
gsap.timeline({ defaults: { ease: 'power3.out' } })
  .from('.hero-badge',   { y: 20, opacity: 0, duration: .55 })
  .from('.hero-title',   { y: 50, opacity: 0, duration: .7 }, '-=.3')
  .from('.hero-sub',     { y: 22, opacity: 0, duration: .55 }, '-=.35')
  .from('.hero-actions', { y: 18, opacity: 0, duration: .45 }, '-=.3')
  .from('.hero-chips',   { y: 16, opacity: 0, duration: .4 }, '-=.25')
  .from('.stat-card',    { x: 30, opacity: 0, duration: .45, stagger: .1 }, '-=.4');

/* ── 3. Navbar scroll ── */
ScrollTrigger.create({
  start: 70,
  onEnter:     () => document.getElementById('nav').classList.add('scrolled'),
  onLeaveBack: () => document.getElementById('nav').classList.remove('scrolled'),
});

/* ── 4. Pipeline nodes reveal ── */
gsap.from('.pipe-node', {
  scrollTrigger: { trigger: '.pipeline-flow', start: 'top 78%' },
  y: 35, opacity: 0, duration: .55, stagger: .1, ease: 'power2.out',
});

/* ── 5. Concept cards reveal ── */
gsap.from('.concept-card', {
  scrollTrigger: { trigger: '.concepts-grid', start: 'top 80%' },
  y: 28, opacity: 0, duration: .5, stagger: .08, ease: 'power2.out',
});

/* ── 6. Hero stat counters ── */
const statTargets = { sv1: 3, sv2: 94, sv3: 3.4, sv4: 8 };
Object.entries(statTargets).forEach(([id, target]) => {
  const el = document.getElementById(id);
  if (!el) return;
  const isDecimal = String(target).includes('.');
  gsap.to({ v: 0 }, {
    v: target, duration: 2.2, delay: .7, ease: 'power2.out',
    onUpdate: function () {
      el.textContent = isDecimal
        ? this.targets()[0].v.toFixed(1)
        : Math.round(this.targets()[0].v);
    }
  });
});

/* ── 7. Code tabs ── */
document.querySelectorAll('.stab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.stab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.code-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(btn.dataset.tab);
    if (panel) {
      panel.classList.add('active');
      gsap.from(panel, { opacity: 0, y: 10, duration: .25, ease: 'power2.out' });
    }
  });
});

/* ── 8. Copy buttons ── */
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const panel = document.getElementById(btn.dataset.src);
    const code  = panel?.querySelector('code');
    if (!code) return;
    navigator.clipboard.writeText(code.innerText).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
    });
  });
});

/* ── 9. Chart.js gauge ── */
const gaugeCtx = document.getElementById('gaugeChart').getContext('2d');
const gaugeChart = new Chart(gaugeCtx, {
  type: 'doughnut',
  data: {
    datasets: [{
      data: [0, 100],
      backgroundColor: ['rgba(0,229,200,1)', 'rgba(255,255,255,.04)'],
      borderWidth: 0,
      borderRadius: 6,
    }]
  },
  options: {
    cutout: '78%', rotation: -90, circumference: 180,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    animation: { duration: 700, easing: 'easeInOutQuart' },
  }
});

/* ── 10. Simulator logic ── */
const classes = [
  { name: '🍎 Apple',  color: '#ef9a9a' },
  { name: '🍌 Banana', color: '#ffe082' },
  { name: '🥭 Mango',  color: '#ffcc80' },
];

const pcBars = document.getElementById('pcBars');
pcBars.innerHTML = classes.map((c, i) => `
  <div class="pcb-row">
    <span class="pcb-name">${c.name}</span>
    <div class="pcb-bg"><div class="pcb-fill" id="pcf-${i}" style="background:${c.color}"></div></div>
    <span class="pcb-pct" id="pcp-${i}">0%</span>
  </div>`).join('');

function estAcc(img, ep, drop, ft, bn, aug) {
  const d = Math.min(1, img / 300);
  const e = Math.min(1, ep / 25);
  const dr = drop > .55 ? -.03 : drop < .2 ? -.02 : 0;
  return Math.min(.98, Math.max(.56,
    .73 + d * .13 + e * .08 + (ft ? .055 : 0) + (bn ? .018 : 0) + (aug ? .022 : 0) + dr
  ));
}

function estTime(img, ep, ft) {
  const steps = Math.ceil(img * 3 * .8 / 32);
  const sec = steps * ep * .013 + (ft ? steps * 8 * .019 : 0);
  const m = sec / 60;
  return m < 1 ? '< 1 min' : `~${Math.round(m)} min`;
}

let curAcc = 0;

function runSim() {
  const img  = +document.getElementById('r-img').value;
  const ep   = +document.getElementById('r-ep').value;
  const drop = +document.getElementById('r-drop').value;
  const ft   = document.getElementById('ck-ft').checked;
  const bn   = document.getElementById('ck-bn').checked;
  const aug  = document.getElementById('ck-aug').checked;

  document.getElementById('o-img').value  = img;
  document.getElementById('o-ep').value   = ep;
  document.getElementById('o-drop').value = drop.toFixed(2);

  const acc    = estAcc(img, ep, drop, ft, bn, aug);
  const accPct = Math.round(acc * 100);
  const loss   = (1 - acc + .04 + Math.random() * .025).toFixed(3);

  gsap.to({ v: curAcc }, {
    v: accPct, duration: .65, ease: 'power2.out',
    onUpdate: function () {
      document.getElementById('gaugeVal').textContent = Math.round(this.targets()[0].v);
    }
  });
  curAcc = accPct;

  const col = accPct >= 88 ? 'rgba(0,229,200,1)' : accPct >= 76 ? 'rgba(255,193,7,1)' : 'rgba(239,83,80,1)';
  gaugeChart.data.datasets[0].data = [accPct, 100 - accPct];
  gaugeChart.data.datasets[0].backgroundColor[0] = col;
  gaugeChart.update();

  document.getElementById('rm-time').textContent  = estTime(img, ep, ft);
  document.getElementById('rm-loss').textContent  = loss;
  document.getElementById('rm-phase').textContent = ft ? 'Phase 1 + Fine-tune' : 'Phase 1 only';

  const dominant = acc * 100;
  const spread   = (100 - dominant) / (classes.length - 1);
  classes.forEach((_, i) => {
    const v = i === 0 ? dominant : Math.max(2, spread + (Math.random() - .5) * 7);
    document.getElementById(`pcf-${i}`).style.width = v.toFixed(1) + '%';
    document.getElementById(`pcp-${i}`).textContent = v.toFixed(1) + '%';
  });
}

['r-img', 'r-ep', 'r-drop', 'ck-ft', 'ck-bn', 'ck-aug'].forEach(id => {
  document.getElementById(id).addEventListener('input', runSim);
  document.getElementById(id).addEventListener('change', runSim);
});
runSim();