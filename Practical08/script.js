/**
 * Transfer Learning Guide — main.js
 * CDNs used: GSAP 3, Chart.js 4, Highlight.js 11, AOS 2
 */

'use strict';

/* ═══════════════════════════════════════════
   1. INIT: AOS scroll-reveal
═══════════════════════════════════════════ */
AOS.init({
  duration: 600,
  easing: 'ease-out-cubic',
  once: true,
  offset: 60,
});


/* ═══════════════════════════════════════════
   2. GSAP: Hero entrance animation
═══════════════════════════════════════════ */
(function heroAnimation() {
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

  tl.from('#heroBadge',   { y: 24, opacity: 0, duration: 0.6 })
    .from('#heroTitle',   { y: 40, opacity: 0, duration: 0.7 }, '-=0.3')
    .from('#heroSub',     { y: 24, opacity: 0, duration: 0.6 }, '-=0.4')
    .from('#heroActions', { y: 20, opacity: 0, duration: 0.5 }, '-=0.35')
    .from('#heroStats',   { y: 16, opacity: 0, duration: 0.5 }, '-=0.3')
    .from('.fruit-pill',  {
        x: 40, opacity: 0, duration: 0.5,
        stagger: 0.1, ease: 'back.out(1.4)'
    }, '-=0.4');
})();


/* ═══════════════════════════════════════════
   3. GSAP: Counter animation for hero stats
═══════════════════════════════════════════ */
(function counterAnimation() {
  document.querySelectorAll('.stat__num').forEach(el => {
    const target = parseInt(el.dataset.target, 10);
    gsap.to({ val: 0 }, {
      val: target,
      duration: 2,
      delay: 0.8,
      ease: 'power2.out',
      onUpdate: function () {
        el.textContent = Math.round(this.targets()[0].val);
      }
    });
  });
})();


/* ═══════════════════════════════════════════
   4. GSAP: Navbar scroll effect
═══════════════════════════════════════════ */
(function navbarScroll() {
  const navbar = document.getElementById('navbar');
  if (!navbar) return;

  ScrollTrigger.create({
    start: 80,
    onEnter:     () => navbar.classList.add('scrolled'),
    onLeaveBack: () => navbar.classList.remove('scrolled'),
  });
})();


/* ═══════════════════════════════════════════
   5. GSAP: Pipeline cards stagger on scroll
═══════════════════════════════════════════ */
(function pipelineReveal() {
  gsap.from('.pipe-card', {
    scrollTrigger: {
      trigger: '.pipeline-grid',
      start: 'top 80%',
    },
    y: 40,
    opacity: 0,
    duration: 0.6,
    stagger: 0.12,
    ease: 'power2.out',
  });
})();


/* ═══════════════════════════════════════════
   6. CODE TABS with GSAP fade transition
═══════════════════════════════════════════ */
(function codeTabs() {
  const stabs  = document.querySelectorAll('.stab');
  const panels = document.querySelectorAll('.code-panel');

  stabs.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = 'tab-' + btn.dataset.tab;

      stabs.forEach(s => s.classList.remove('active'));
      btn.classList.add('active');

      panels.forEach(p => {
        if (p.id === targetId) {
          p.classList.add('active');
          gsap.from(p, { opacity: 0, y: 10, duration: 0.25, ease: 'power2.out' });
        } else {
          p.classList.remove('active');
        }
      });
    });
  });
})();


/* ═══════════════════════════════════════════
   7. Highlight.js: syntax highlighting
═══════════════════════════════════════════ */
(function syntaxHighlight() {
  document.querySelectorAll('pre code').forEach(block => {
    hljs.highlightElement(block);
  });
})();


/* ═══════════════════════════════════════════
   8. Copy-to-clipboard for code panels
═══════════════════════════════════════════ */
(function copyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const panelId = btn.dataset.target;
      const panel   = document.getElementById(panelId);
      if (!panel) return;

      const code = panel.querySelector('code');
      const text = code ? code.innerText : '';

      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {
        btn.textContent = 'Error';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      });
    });
  });
})();


/* ═══════════════════════════════════════════
   9. Chart.js: Donut gauge + class bars simulator
═══════════════════════════════════════════ */
(function simulator() {
  const imgSlider   = document.getElementById('imgCount');
  const epochSlider = document.getElementById('epochCount');
  const ftCheck     = document.getElementById('ftCheck');
  const imgOut      = document.getElementById('imgOut');
  const epochOut    = document.getElementById('epochOut');
  const accNum      = document.getElementById('accNum');
  const timeVal     = document.getElementById('timeVal');
  const phaseVal    = document.getElementById('phaseVal');
  const classBars   = document.getElementById('classBars');

  if (!imgSlider) return;

  // ── Donut gauge ──
  const ctx = document.getElementById('accGauge').getContext('2d');
  const gauge = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: [
          'rgba(102,126,234,1)',
          'rgba(255,255,255,0.05)',
        ],
        borderWidth: 0,
        borderRadius: 6,
      }]
    },
    options: {
      cutout: '78%',
      rotation: -90,
      circumference: 180,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 600, easing: 'easeInOutQuart' },
    }
  });

  // ── Class names and gradient colours ──
  const classes = [
    { name: '🍎 Apple',  color: '#f87171' },
    { name: '🍌 Banana', color: '#fbbf24' },
    { name: '🍇 Grape',  color: '#a78bfa' },
    { name: '🥭 Mango',  color: '#fb923c' },
    { name: '🍊 Orange', color: '#34d399' },
  ];

  // Build class bar rows once
  classBars.innerHTML = classes.map((c, i) => `
    <div class="class-bar-row">
      <span class="class-name">${c.name}</span>
      <div class="class-bar-bg">
        <div class="class-bar-fill" id="bar-${i}" style="width:0%; background:${c.color};"></div>
      </div>
      <span class="class-pct" id="pct-${i}">0%</span>
    </div>
  `).join('');

  // ── Heuristic accuracy model ──
  function estimateAccuracy(imgs, epochs, ft) {
    const dataFactor  = Math.min(1, imgs / 300);
    const epochFactor = Math.min(1, epochs / 15);
    const ftBonus     = ft ? 0.07 : 0;
    const base        = 0.72 + dataFactor * 0.14 + epochFactor * 0.08 + ftBonus;
    return Math.min(0.995, Math.max(0.55, base));
  }

  // ── Training time estimate ──
  function estimateTime(imgs, epochs, ft) {
    const samples     = imgs * 5 * 0.8;
    const steps       = Math.ceil(samples / 32);
    const phase1Sec   = steps * epochs * 0.012;
    const phase2Sec   = ft ? steps * Math.min(epochs, 8) * 0.018 : 0;
    const totalMin    = (phase1Sec + phase2Sec) / 60;
    return totalMin < 1 ? '< 1 min' : `~${Math.round(totalMin)} min`;
  }

  // ── Simulate class-level scores ──
  function fakeClassScores(overallAcc) {
    // One dominant class, spread remainder
    const dominant = overallAcc * 100;
    return classes.map((_, i) => {
      if (i === 0) return dominant;
      const spread = (100 - dominant) / (classes.length - 1);
      return Math.max(2, spread + (Math.random() - 0.5) * 6);
    }).map(v => Math.min(99.9, Math.max(1, v)));
  }

  let currentAcc = 0;

  function update() {
    const imgs   = parseInt(imgSlider.value, 10);
    const epochs = parseInt(epochSlider.value, 10);
    const ft     = ftCheck.checked;

    imgOut.value   = imgs;
    epochOut.value = epochs;

    const acc = estimateAccuracy(imgs, epochs, ft);
    const accPct = Math.round(acc * 100);

    // Animate the number counter
    gsap.to({ val: currentAcc }, {
      val: accPct,
      duration: 0.6,
      ease: 'power2.out',
      onUpdate: function () {
        accNum.textContent = Math.round(this.targets()[0].val);
      },
      onComplete: () => { currentAcc = accPct; }
    });

    // Update gauge chart
    const remaining = 100 - accPct;
    const gaugeColor = accPct >= 88
      ? 'rgba(56,239,125,1)'
      : accPct >= 78
        ? 'rgba(251,191,36,1)'
        : 'rgba(248,113,113,1)';

    gauge.data.datasets[0].data = [accPct, remaining];
    gauge.data.datasets[0].backgroundColor[0] = gaugeColor;
    gauge.update();

    // Update meta
    timeVal.textContent  = estimateTime(imgs, epochs, ft);
    phaseVal.textContent = ft ? 'Phase 1 + Fine-tune' : 'Phase 1 only';

    // Update class bars
    const scores = fakeClassScores(acc);
    classes.forEach((_, i) => {
      const bar = document.getElementById(`bar-${i}`);
      const pct = document.getElementById(`pct-${i}`);
      if (bar && pct) {
        const val = scores[i].toFixed(1);
        bar.style.width = val + '%';
        pct.textContent = val + '%';
      }
    });
  }

  imgSlider.addEventListener('input',   update);
  epochSlider.addEventListener('input', update);
  ftCheck.addEventListener('change',   update);
  update(); // initial render
})();


/* ═══════════════════════════════════════════
   10. GSAP: Concept cards hover magnetic effect
═══════════════════════════════════════════ */
(function magneticCards() {
  document.querySelectorAll('.concept-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width  - 0.5) * 8;
      const y = ((e.clientY - rect.top)  / rect.height - 0.5) * 8;
      gsap.to(card, { rotateX: -y, rotateY: x, duration: 0.4, ease: 'power2.out', transformPerspective: 800 });
    });
    card.addEventListener('mouseleave', () => {
      gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.5, ease: 'elastic.out(1, 0.5)' });
    });
  });
})();


/* ═══════════════════════════════════════════
   11. Active nav link highlight on scroll
═══════════════════════════════════════════ */
(function activeNavLinks() {
  const sections = ['pipeline', 'code', 'simulator', 'concepts'];
  sections.forEach(id => {
    const el = document.getElementById(id);
    const link = document.querySelector(`.nav-links a[href="#${id}"]`);
    if (!el || !link) return;

    ScrollTrigger.create({
      trigger: el,
      start: 'top 60%',
      end: 'bottom 40%',
      onEnter:      () => link.style.color = '#fff',
      onLeave:      () => link.style.color = '',
      onEnterBack:  () => link.style.color = '#fff',
      onLeaveBack:  () => link.style.color = '',
    });
  });
})();