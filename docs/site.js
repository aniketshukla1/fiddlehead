/*
 * Page behaviour for the Fiddlehead site: the theme toggle (with a view transition),
 * moments that play once as a section arrives ([data-play] gets .play; [data-count-to]
 * counts up), the statement that fills in as you read it ([data-fill]), copy buttons
 * ([data-copy]), the launch video that plays muted while it is on screen ([data-autoplay]),
 * and pointer light on cards (.spot) and the footer wordmark ([data-glow]).
 * Everything only adds to a page that is already complete; the particles live in
 * particle-scene.js. The inline boot script in <head> sets data-theme from storage
 * and adds the .js class.
 */
(() => {
  'use strict';

  const root = document.documentElement;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const hasObserver = 'IntersectionObserver' in window;
  const THEME_KEY = 'fiddlehead-site-theme'; // match the boot script in <head>

  /* ---------- Theme ---------- */
  const themeNow = () => root.dataset.theme || (darkQuery.matches ? 'dark' : 'light');
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  function syncTheme() {
    const mode = themeNow();
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
      button.dataset.mode = mode;
      button.setAttribute('aria-label', mode === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    });
    if (themeMeta) themeMeta.setAttribute('content', mode === 'dark' ? '#030605' : '#e3f1e9');
  }
  function setTheme(next) {
    const apply = () => { root.dataset.theme = next; syncTheme(); };
    if (document.startViewTransition && !reduce.matches) document.startViewTransition(apply);
    else apply();
    try { localStorage.setItem(THEME_KEY, next); } catch (error) { /* storage can be blocked; the choice lasts for this visit */ }
  }
  document.addEventListener('click', (event) => {
    const button = event.target.closest && event.target.closest('[data-theme-toggle]');
    if (button) setTheme(themeNow() === 'dark' ? 'light' : 'dark');
  });
  if (darkQuery.addEventListener) darkQuery.addEventListener('change', syncTheme);
  new MutationObserver(syncTheme).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  syncTheme();

  /* ---------- Counters: the resting number is the real one; on entry it counts up to it ---------- */
  document.querySelectorAll('[data-count-to]').forEach((el) => {
    const host = el.closest('[data-play]');
    if (!host || !hasObserver || reduce.matches) return;
    const to = Number(el.dataset.countTo);
    const final = el.textContent;
    host.addEventListener('scene:play', () => {
      const start = performance.now();
      const tick = (now) => {
        const k = Math.min(1, (now - start) / 1400);
        el.textContent = k < 1 ? String(Math.round(to * (1 - Math.pow(1 - k, 3)))) : final;
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, { once: true });
  });

  /* ---------- Moments that play once, when their section arrives ---------- */
  function onEnter(selector, className, options) {
    const els = document.querySelectorAll(selector);
    const fire = (el) => { el.classList.add(className); el.dispatchEvent(new CustomEvent('scene:play')); };
    if (!hasObserver) { els.forEach(fire); return; }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        fire(entry.target);
      });
    }, options);
    els.forEach((el) => observer.observe(el));
  }
  onEnter('[data-play]', 'play', { rootMargin: '0px 0px -12% 0px', threshold: 0.2 });

  /* ---------- The statement fills in as you read it ---------- */
  const fills = Array.from(document.querySelectorAll('[data-fill]'));
  fills.forEach((el) => {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const texts = [];
    while (walker.nextNode()) texts.push(walker.currentNode);
    texts.forEach((node) => {
      const frag = document.createDocumentFragment();
      node.textContent.split(/(\s+)/).forEach((part) => {
        if (part === '') return;
        if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(part)); return; }
        const span = document.createElement('span');
        span.className = 'word';
        span.textContent = part;
        frag.appendChild(span);
      });
      node.parentNode.replaceChild(frag, node);
    });
    el.words = Array.from(el.querySelectorAll('.word'));
    el.lit = 0;
    el.classList.add('filling');
  });
  let queued = false;
  function onScroll() {
    queued = false;
    const vh = window.innerHeight || 1;
    fills.forEach((el) => {
      const r = el.getBoundingClientRect();
      const progress = Math.min(1, Math.max(0, (vh * 0.86 - r.top) / (r.height + vh * 0.36)));
      const count = reduce.matches ? el.words.length : Math.round(progress * el.words.length);
      for (; el.lit < count; el.lit++) el.words[el.lit].classList.add('on');
      for (; el.lit > count; el.lit--) el.words[el.lit - 1].classList.remove('on');
    });
  }
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(onScroll); } };
  window.addEventListener('scroll', queue, { passive: true });
  window.addEventListener('resize', queue);
  onScroll();

  /* ---------- Copy buttons ---------- */
  document.querySelectorAll('[data-copy]').forEach((button) => {
    button.hidden = !(navigator.clipboard && window.isSecureContext);
    button.addEventListener('click', async () => {
      const source = document.getElementById(button.dataset.copy);
      if (!source) return;
      try {
        await navigator.clipboard.writeText(source.innerText.trim());
        button.textContent = 'Copied';
      } catch (error) {
        button.textContent = 'Press ⌘C';
      }
      setTimeout(() => { button.textContent = 'Copy'; }, 1600);
    });
  });

  /* ---------- The launch video plays muted while it is on screen ---------- */
  document.querySelectorAll('video[data-autoplay]').forEach((video) => {
    const sound = document.querySelector(`[data-sound="${video.id}"]`);
    if (sound) {
      sound.hidden = false;
      sound.addEventListener('click', () => {
        video.muted = false;
        video.currentTime = 0;
        video.controls = true;
        video.play().catch(() => {});
        sound.hidden = true;
      });
    }
    if (reduce.matches || !hasObserver) { video.controls = true; return; }
    new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) video.play().catch(() => { video.controls = true; });
        else video.pause();
      });
    }, { threshold: 0.35 }).observe(video);
  });

  /* ---------- Light that follows the pointer: card spotlights and the footer mark ---------- */
  if (!reduce.matches && window.matchMedia('(pointer: fine)').matches) {
    document.querySelectorAll('.card').forEach((el) => el.classList.add('spot'));
    let pending = null;
    let frame = 0;
    const paint = () => {
      frame = 0;
      const e = pending;
      const target = e.target instanceof Element ? e.target : null;
      if (!target) return;
      const card = target.closest('.spot');
      if (card) {
        const r = card.getBoundingClientRect();
        card.style.setProperty('--sx', `${e.clientX - r.left}px`);
        card.style.setProperty('--sy', `${e.clientY - r.top}px`);
      }
      const footer = target.closest('.site-footer');
      const glow = footer && footer.querySelector('[data-glow]');
      if (glow) {
        const r = glow.getBoundingClientRect();
        glow.style.setProperty('--fx', `${e.clientX - r.left}px`);
        glow.style.setProperty('--fy', `${e.clientY - r.top}px`);
      }
    };
    document.addEventListener('pointermove', (event) => {
      pending = event;
      if (!frame) frame = requestAnimationFrame(paint);
    }, { passive: true });
  }
})();
