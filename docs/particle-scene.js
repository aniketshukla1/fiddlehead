/*
 * Particle scene: one cloud of particles, fixed behind the page, that takes a
 * different shape for each section and moves between them as you scroll.
 *
 * Every element with data-shape is an anchor, and belongs to the <section> it
 * sits in. While the edge between two sections is on screen, each keeps its
 * own shape and particles hop across in proportion to how much of the next
 * section shows; each hop takes about 0.6 s, so nothing is left hanging in the
 * air when scrolling stops. A shape in play stays inside the visible part of
 * its section. Particles dim behind text blocks so they never fight reading.
 * A low-resolution fog layer drifts underneath, scaled up by CSS.
 *
 * Anchor attributes:
 *   data-shape    logo | path | curve | matrix | frond | stars
 *   data-palette  a key of PALETTES (defaults to the shape's own)
 *   data-fit      contain | width | height | cover | long (how the shape is sized to the anchor)
 *   data-scale    multiplier on that size
 *   data-hold     "anchor" keeps the shape on its anchor (it scrolls away with it);
 *                 the default waits on screen while its section is visible
 * data-quiet on any text block: particles dim behind it (common text classes
 * are included by default, see quietEls).
 * ?debug-scene in the URL puts the current pair, frames and sections on window.__scene.
 *
 * Page contract: <div class="scene-layer" data-scene><canvas class="scene-fog"></canvas>
 * <canvas class="scene-dots"></canvas></div> as the LAST child of <body>,
 * position: fixed; z-index: 0, with section content at z-index: 1. The page
 * must be complete without this file; it adds .has-scene to <html> once the
 * first frame is on screen. Edit LOGO and PALETTES; leave the rest alone.
 */
(() => {
  'use strict';

  const root = document.documentElement;
  const layer = document.querySelector('[data-scene]');
  // With reduced motion the shapes still appear, held still, and change only as you scroll.
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!layer) return;
  const dots = layer.querySelector('.scene-dots');
  const fogCanvas = layer.querySelector('.scene-fog');
  const anchorEls = Array.from(document.querySelectorAll('[data-shape]'));
  if (!dots || anchorEls.length === 0) return;

  const glOptions = { alpha: true, antialias: false, premultipliedAlpha: true, powerPreference: 'high-performance' };
  const gl = dots.getContext('webgl2', glOptions) || dots.getContext('webgl', glOptions);
  if (!gl || gl.getParameter(gl.MAX_VERTEX_ATTRIBS) < 10) return;


  /* ---------- Your brand: edit LOGO here and PALETTES below ---------- */

  // The mark: a fern fiddlehead, a stem curling into a coil (the same path as the app's
  // logo). It is stroked, not filled, in a VIEW x VIEW box; `accent` circles take the
  // palette's accent colour (here the coil's heart). centre and span place the mark:
  // span is half its height in box units.
  const LOGO = {
    view: 24,
    paths: [
      'M8.8 21.2C7.3 18.4 6.6 14.6 6.6 9.6L7.0 8.4L7.6 7.4L8.3 6.5L9.2 5.8L10.1 5.3L11.1 5.1L12.1 5.0' +
      'L13.0 5.1L13.8 5.4L14.5 5.8L15.2 6.3L15.6 6.9L16.0 7.6L16.2 8.2L16.2 8.9L16.2 9.6L16.0 10.1' +
      'L15.7 10.7L15.3 11.1L14.9 11.4L14.4 11.7L14.0 11.8L13.5 11.9L13.0 11.8L12.6 11.7L12.3 11.5' +
      'L12.0 11.2L11.7 10.9L11.6 10.6L11.5 10.3L11.4 10.0L11.4 9.6L11.5 9.4L11.7 9.1L11.8 8.9L12.1 8.7',
    ],
    stroke: 2.3,
    accent: [{ cx: 12.9, cy: 9.6, r: 1.5 }],
    centre: [11.4, 13.1],
    span: 9.3,
  };

  const SHAPES = ['logo', 'path', 'curve', 'matrix', 'frond', 'stars'];
  const small = Math.min(window.innerWidth, window.innerHeight) < 700 || (navigator.hardwareConcurrency || 8) <= 4;
  const N = small ? 6500 : 14000;

  // A seeded random keeps every shape the same on every visit.
  let seed = 20261101;
  const random = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    let u = 0;
    while (u === 0) u = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  };

  /* ---------- Shapes: x, y, z in roughly -1..1, plus one value per shape in w ---------- */
  // w = 1 takes the palette's accent colour in every shape.

  // The fiddlehead, sampled from its own stroke. w marks the coil's heart.
  function logo(out) {
    const S = 220;
    const k = S / LOGO.view;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = S;
    const g = canvas.getContext('2d');
    g.scale(k, k);
    g.lineWidth = LOGO.stroke;
    g.lineCap = 'round';
    g.lineJoin = 'round';
    LOGO.paths.forEach((d) => g.stroke(new Path2D(d)));
    const px = g.getImageData(0, 0, S, S).data;
    const filled = [];
    for (let i = 0; i < S * S; i++) if (px[i * 4 + 3] > 140) filled.push(i);
    if (filled.length === 0) return stars(out);
    for (let i = 0; i < N; i++) {
      const f = filled[(random() * filled.length) | 0];
      const sx = ((f % S) + random()) / k;
      const sy = (((f / S) | 0) + random()) / k;
      const x = (sx - LOGO.centre[0]) / LOGO.span;
      const y = -(sy - LOGO.centre[1]) / LOGO.span;
      const o = i * 4;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = -0.3 * (x * x + 0.25 * y * y) + (random() - 0.5) * 0.12;
      out[o + 3] = LOGO.accent.some((c) => (sx - c.cx) ** 2 + (sy - c.cy) ** 2 < c.r * c.r) ? 1 : 0;
    }
  }

  // Four steps: a braided trail through four stops. w marks the stops.
  function path(out) {
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      if (random() < 0.3) {
        const stop = (random() * 4) | 0;
        const r = 0.055 * Math.cbrt(random());
        const th = random() * Math.PI * 2;
        const ph = Math.acos(2 * random() - 1);
        out[o] = -1 + stop * (2 / 3) + r * Math.sin(ph) * Math.cos(th);
        out[o + 1] = r * Math.cos(ph);
        out[o + 2] = r * Math.sin(ph) * Math.sin(th);
        out[o + 3] = 1;
      } else {
        const x = -1.12 + random() * 2.24;
        const strand = (random() * 3) | 0;
        const amp = 0.035 + 0.02 * strand;
        const phase = strand * 2.1;
        out[o] = x;
        out[o + 1] = Math.sin(x * 4.2 + phase) * amp + gauss() * 0.008;
        out[o + 2] = Math.cos(x * 4.2 + phase) * amp + gauss() * 0.01;
        out[o + 3] = 0;
      }
    }
  }

  // A training curve: noisy and flat while the weights are random, then the loss drops
  // as the model learns and settles. w: 0 the curve, 0.5 the axes, 1 the marker where
  // the drop begins (in the studio, where the induction circuit appears).
  const DROP = 0.36;
  const lossAt = (u) => {
    const fall = u < DROP ? 0 : 1 - Math.exp(-(u - DROP) * 6.5);
    return 0.62 - fall * 1.12;
  };
  function curve(out) {
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      const r = random();
      if (r < 0.1) {
        // the axes: a faint L
        const onX = random() < 0.62;
        out[o] = onX ? -1.25 + random() * 2.5 : -1.25;
        out[o + 1] = onX ? -0.72 : -0.72 + random() * 1.6;
        out[o + 2] = 0;
        out[o + 3] = 0.5;
      } else if (r < 0.16) {
        // the marker: a dashed vertical line where the drop begins
        let y = -0.72 + random() * 1.6;
        if (((y + 0.72) * 14) % 1 > 0.55) y -= 0.05;
        out[o] = -1.25 + DROP * 2.5 + gauss() * 0.004;
        out[o + 1] = y;
        out[o + 2] = 0;
        out[o + 3] = 1;
      } else {
        const u = random();
        const calm = Math.min(1, Math.max(0, (u - DROP) * 3.2));
        out[o] = -1.25 + u * 2.5;
        out[o + 1] = lossAt(u) + gauss() * (0.075 * (1 - calm) + 0.012) + Math.sin(u * 61) * 0.02 * (1 - calm);
        out[o + 2] = gauss() * 0.035;
        out[o + 3] = 0;
      }
    }
  }

  // Four attention heads as small causal heatmaps, one per behaviour the studio labels:
  // previous-token, positional (self), attention sink, and induction, the copy circuit
  // (w = 1). Strong cells get more particles; faint cells carry w = 0.15.
  function matrix(out) {
    const n = 12;
    const cell = 0.072;
    const gap = 0.2;
    const size = n * cell;
    const heads = [
      { ox: -size - gap / 2, oy: size + gap / 2, kind: 'prev' },
      { ox: gap / 2, oy: size + gap / 2, kind: 'pos' },
      { ox: -size - gap / 2, oy: -gap / 2, kind: 'sink' },
      { ox: gap / 2, oy: -gap / 2, kind: 'induction' },
    ];
    // A sequence that repeats every 5 tokens: an induction head looks at the token after
    // the previous occurrence of the current one, a stripe 4 below the diagonal.
    const strong = (kind, r, c) => (kind === 'prev' ? c === r - 1 : kind === 'pos' ? c === r
      : kind === 'sink' ? c === 0 && r > 0 : r >= 5 && c === r - 4);
    const cells = [];
    let total = 0;
    heads.forEach((h) => {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c <= r; c++) {
          const s = strong(h.kind, r, c);
          const weight = s ? 5 : 1;
          total += weight;
          cells.push({ x: h.ox + (c + 0.5) * cell, y: h.oy - (r + 0.5) * cell, weight, total,
            w: s ? (h.kind === 'induction' ? 1 : 0.6) : 0.15 });
        }
      }
    });
    for (let i = 0; i < N; i++) {
      const pick = random() * total;
      let lo = 0;
      let hi = cells.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cells[mid].total < pick) lo = mid + 1; else hi = mid; }
      const c = cells[lo];
      const o = i * 4;
      out[o] = c.x + (random() - 0.5) * cell * 0.84;
      out[o + 1] = c.y + (random() - 0.5) * cell * 0.84;
      out[o + 2] = gauss() * 0.02;
      out[o + 3] = c.w;
    }
  }

  // The fern, grown: a curving stem with paired leaflets that shorten toward the top,
  // and the tip still curled. w marks the leaflet tips and the curl.
  const stemAt = (t) => [0.42 * t * t - 0.06 * t, -1 + 1.86 * t];
  function frond(out) {
    const pairs = 17;
    const tip = stemAt(1);
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      const r = random();
      if (r < 0.13) {
        const [x, y] = stemAt(random());
        out[o] = x + gauss() * 0.006;
        out[o + 1] = y;
        out[o + 2] = gauss() * 0.01;
        out[o + 3] = 0;
      } else if (r < 0.2) {
        const a = random() * Math.PI * 3.4;
        const rad = 0.1 * Math.exp(-a * 0.16);
        out[o] = tip[0] - 0.08 + Math.cos(a) * rad;
        out[o + 1] = tip[1] + Math.sin(a) * rad;
        out[o + 2] = gauss() * 0.01;
        out[o + 3] = 1;
      } else {
        const k = (random() * pairs) | 0;
        const side = random() < 0.5 ? -1 : 1;
        const t = 0.05 + (k / pairs) * 0.84;
        const [bx, by] = stemAt(t);
        const len = 0.07 + 0.5 * Math.pow(Math.sin(Math.PI * Math.min(1, (t - 0.02) / 0.9)), 0.85);
        const up = 0.55 + t * 0.35;
        const dx = side * Math.cos(up);
        const dy = Math.sin(up);
        const u = random();
        const half = len * 0.17 * Math.sin(Math.PI * Math.pow(u, 0.75));
        const v = (random() * 2 - 1) * half;
        out[o] = bx + dx * u * len - dy * v * side;
        out[o + 1] = by + dy * u * len + dx * v * side;
        out[o + 2] = gauss() * 0.02 + side * 0.06 * u;
        out[o + 3] = u > 0.88 ? 1 : 0;
      }
    }
  }

  // Noise: random weights before training, or faint dust behind long sections. w marks the bright points.
  function stars(out) {
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      out[o] = (random() * 2 - 1) * 2.4;
      out[o + 1] = (random() * 2 - 1) * 1.5;
      out[o + 2] = -1.6 + random() * 2;
      out[o + 3] = random() < 0.04 ? 1 : 0;
    }
  }

  /* ---------- Colours: the brand's tokens, a light and a dark set per palette ---------- */

  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  // base, accent, alpha, glow (0 sits on the sky like ink, 1 adds light), fog colour, fog amount
  const P = (base, accent, alpha, glow, fog, fogAmount) => ({ base: hex(base), accent: hex(accent), alpha, glow, fog: hex(fog), fogAmount });
  const PALETTES = {
    fern: { light: P('#0b7e54', '#b45309', 1, 0.04, '#ffffff', 0.3), dark: P('#3dd68c', '#f5c26b', 0.95, 0.9, '#1f6b4a', 0.14) },
    noise: { light: P('#7d8a83', '#3dd68c', 0.95, 1, '#1f6b4a', 0.08), dark: P('#a0a4ab', '#3dd68c', 0.9, 1, '#1f6b4a', 0.08) },
    dawn: { light: P('#0b7e54', '#b45309', 0.95, 0.03, '#ffffff', 0.18), dark: P('#7ee2b0', '#f5c26b', 0.9, 0.9, '#1f6b4a', 0.1) },
    signal: { light: P('#0b7e54', '#b45309', 0.95, 0.04, '#ffffff', 0.14), dark: P('#3dd68c', '#f5a524', 0.92, 0.9, '#1f6b4a', 0.1) },
    heads: { light: P('#4b5a52', '#0b7e54', 0.95, 0.03, '#ffffff', 0.12), dark: P('#9aa59f', '#3dd68c', 0.9, 0.9, '#1f6b4a', 0.1) },
    frond: { light: P('#0b7e54', '#4d7c0f', 1, 0.04, '#ffffff', 0.2), dark: P('#3dd68c', '#d9f99d', 0.95, 0.95, '#1f6b4a', 0.12) },
    dust: { light: P('#5b6068', '#0b7e54', 0.35, 0, '#ffffff', 0.16), dark: P('#80858d', '#3dd68c', 0.34, 0.75, '#1f6b4a', 0.08) },
    night: { light: P('#3dd68c', '#f5c26b', 0.95, 1, '#1f6b4a', 0.1), dark: P('#3dd68c', '#f5c26b', 0.95, 1, '#1f6b4a', 0.1) },
  };
  const DEFAULT_PALETTE = { logo: 'fern', path: 'dawn', curve: 'signal', matrix: 'heads', frond: 'frond', stars: 'dust' };
  const isDark = () => (root.dataset.theme || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';

  /* ---------- GPU ---------- */

  const VERT = `
precision highp float;
attribute vec4 aLogo; attribute vec4 aPath; attribute vec4 aCurve; attribute vec4 aMatrix;
attribute vec4 aFrond; attribute vec4 aStars;
attribute vec4 aRand;
uniform vec2 uRes; uniform float uDpr; uniform float uTime; uniform float uSize;
uniform float uFrom; uniform float uTo;
attribute float aHop;
uniform vec4 uQuiet[4];
uniform vec3 uFromFrame; uniform vec3 uToFrame; uniform vec2 uRot;
uniform vec3 uFromBase; uniform vec3 uFromAccent; uniform vec3 uToBase; uniform vec3 uToAccent;
uniform vec2 uAlpha; uniform vec2 uTilt; uniform vec2 uPointer;
varying vec3 vColor; varying float vAlpha;

vec3 rotX(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z); }
vec3 rotY(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }
vec2 rot2(vec2 p, float a) { float c = cos(a), s = sin(a); return vec2(c * p.x - s * p.y, s * p.x + c * p.y); }

vec4 shape(float i) {
  if (i < 0.5) return aLogo;
  if (i < 1.5) return aPath;
  if (i < 2.5) return aCurve;
  if (i < 3.5) return aMatrix;
  if (i < 4.5) return aFrond;
  return aStars;
}

// The oscilloscope beam that sweeps the training curve left to right.
float sweep(vec4 s) { return exp(-pow((s.x - (mod(uTime * 0.42, 3.4) - 1.7)) * 6.0, 2.0)); }

vec3 pose(float i, vec4 s, float rz) {
  vec3 p = s.xyz;
  float yaw = 0.0;
  float pitch = 0.0;
  if (i < 0.5) {
    // The fiddlehead sways slowly and bobs, a coil holding still before it unfurls.
    yaw = sin(uTime * 0.32) * 0.5;
    p.y += sin(uTime * 1.1) * 0.025;
  } else if (i < 1.5) {
    pitch = 0.0;
  } else if (i < 2.5) {
    yaw = sin(uTime * 0.2) * 0.16;
  } else if (i < 3.5) {
    yaw = sin(uTime * 0.18) * 0.3 + 0.08;
    pitch = 0.16;
  } else if (i < 4.5) {
    // The frond sways from its base.
    p.x += sin(uTime * 0.7 + p.y * 1.6) * 0.03 * (p.y + 1.0);
    yaw = sin(uTime * 0.22) * 0.55;
  } else {
    yaw = uTime * 0.015;
  }
  p.xy = rot2(p.xy, rz);
  p = rotY(p, yaw + uTilt.x);
  return rotX(p, pitch + uTilt.y);
}

float accent(float i, vec4 s) { return step(0.9, s.w); }
float shine(float i, vec4 s) {
  if (i < 0.5) return s.w * 0.9;
  if (i < 1.5) return s.w;
  if (i < 2.5) return s.w < 0.25 ? sweep(s) * 1.3 : 0.0;
  if (i < 3.5) return step(0.5, s.w) * 0.45;
  if (i < 4.5) return s.w * 0.4;
  return s.w * 1.4;
}
float lit(float i, vec4 s) {
  if (i > 1.5 && i < 2.5) {
    if (s.w > 0.25 && s.w < 0.75) return 0.32;
    return s.w > 0.75 ? 0.9 : 0.66 + 0.34 * sweep(s);
  }
  if (i > 2.5 && i < 3.5) {
    if (s.w < 0.3) return 0.5;
    if (s.w > 0.9) return 0.72 + 0.28 * sin(uTime * 2.4 - (s.x - s.y) * 6.0);
    return 1.0;
  }
  if (i > 4.5) return 0.55 + 0.45 * sin(uTime * (1.0 + aRand.y * 2.0) + aRand.y * 40.0);
  return 1.0;
}

void main() {
  float r = aRand.x;
  float m = smoothstep(0.0, 1.0, aHop);
  vec4 sa = shape(uFrom);
  vec4 sb = shape(uTo);
  vec3 p = mix(pose(uFrom, sa, uRot.x), pose(uTo, sb, uRot.y), m);
  // On the way between shapes each particle loops out a little.
  p += sin(m * 3.14159) * vec3(sin(r * 61.0 + uTime * 0.7), cos(r * 47.0 + uTime * 0.6), sin(r * 29.0)) * 0.45;
  // A few particles drift loosely around every shape.
  p += step(0.96, aRand.z) * vec3(sin(uTime * 0.31 + r * 50.0), cos(uTime * 0.27 + r * 70.0), sin(uTime * 0.21 + r * 90.0)) * 0.22;

  vec3 frame = mix(uFromFrame, uToFrame, m);
  float persp = 2.8 / (2.8 - clamp(p.z, -2.0, 1.6));
  vec2 screen = frame.xy + vec2(p.x, -p.y) * frame.z * persp;
  vec2 d = screen - uPointer;
  screen += normalize(d + vec2(0.0001)) * exp(-dot(d, d) / 9000.0) * 26.0;
  gl_Position = vec4(screen.x / uRes.x * 2.0 - 1.0, 1.0 - screen.y / uRes.y * 2.0, 0.0, 1.0);

  float g = mix(shine(uFrom, sa), shine(uTo, sb), m);
  gl_PointSize = uSize * uDpr * persp * (0.6 + aRand.w * 0.8) * (1.0 + g * 1.4);
  vColor = mix(mix(uFromBase, uFromAccent, accent(uFrom, sa)), mix(uToBase, uToAccent, accent(uTo, sb)), m);
  // Particles fade behind text, so they never compete with reading.
  float hush = 1.0;
  for (int q = 0; q < 4; q++) {
    vec4 box = uQuiet[q];
    vec2 inside = min(screen - box.xy, box.zw - screen);
    hush = min(hush, mix(1.0, 0.28, smoothstep(-24.0, 16.0, min(inside.x, inside.y))));
  }
  vAlpha = mix(uAlpha.x * lit(uFrom, sa), uAlpha.y * lit(uTo, sb), m) * (0.6 + 0.4 * aRand.y) * hush;
}`;


  const FRAG = `
precision mediump float;
varying vec3 vColor; varying float vAlpha;
uniform float uGlow;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c) * 4.0;
  // Glowing particles fade softly; ink particles on a light sky keep a crisp edge.
  float soft = 1.0 - smoothstep(0.0, 1.0, d);
  float a = mix(1.0 - smoothstep(0.45, 1.0, d), soft * soft, uGlow) * vAlpha;
  if (a < 0.003) discard;
  // Always valid premultiplied colour (colour never brighter than alpha). A GPU compositor
  // drops pixels whose alpha is below their colour, so glow comes from additive blending
  // inside the canvas instead (see draw).
  gl_FragColor = vec4(vColor * a, a);
}`;

  function compile(context, vert, frag) {
    const program = context.createProgram();
    [[context.VERTEX_SHADER, vert], [context.FRAGMENT_SHADER, frag]].forEach(([type, source]) => {
      const shader = context.createShader(type);
      context.shaderSource(shader, source);
      context.compileShader(shader);
      if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) throw new Error(context.getShaderInfoLog(shader) || 'shader');
      context.attachShader(program, shader);
    });
    context.linkProgram(program);
    if (!context.getProgramParameter(program, context.LINK_STATUS)) throw new Error(context.getProgramInfoLog(program) || 'link');
    return program;
  }

  let program;
  try {
    program = compile(gl, VERT, FRAG);
  } catch (error) {
    return;
  }
  gl.useProgram(program);

  const builders = { logo, path, curve, matrix, frond, stars };
  const attrNames = { logo: 'aLogo', path: 'aPath', curve: 'aCurve', matrix: 'aMatrix', frond: 'aFrond', stars: 'aStars' };
  const upload = (name, data) => {
    const loc = gl.getAttribLocation(program, name);
    if (loc < 0) return;
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
  };
  SHAPES.forEach((name) => {
    const data = new Float32Array(N * 4);
    builders[name](data);
    upload(attrNames[name], data);
  });
  const rand = new Float32Array(N * 4);
  for (let i = 0; i < rand.length; i++) rand[i] = random();
  upload('aRand', rand);

  // Each particle's place between the two shapes in play: 0 is the first, 1 the second.
  // It moves toward its side over about 0.6 s, so particles finish their hop and rest in a
  // shape instead of hanging between sections when the scrolling stops.
  const hop = new Float32Array(N);
  const hopBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, hopBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, hop, gl.DYNAMIC_DRAW);
  const hopLoc = gl.getAttribLocation(program, 'aHop');
  if (hopLoc >= 0) {
    gl.enableVertexAttribArray(hopLoc);
    gl.vertexAttribPointer(hopLoc, 1, gl.FLOAT, false, 0, 0);
  }

  const u = {};
  ['uRes', 'uDpr', 'uTime', 'uSize', 'uFrom', 'uTo', 'uQuiet', 'uFromFrame', 'uToFrame', 'uRot', 'uFromBase', 'uFromAccent', 'uToBase', 'uToAccent', 'uAlpha', 'uTilt', 'uPointer', 'uGlow']
    .forEach((name) => { u[name] = gl.getUniformLocation(program, name); });
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);

  /* ---------- Fog: a few thousand pixels of noise, stretched by CSS ---------- */

  let fog = null;
  if (fogCanvas) {
    const fgl = fogCanvas.getContext('webgl', { alpha: true, antialias: false, premultipliedAlpha: true });
    if (fgl) {
      try {
        const fogProgram = compile(fgl, `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`, `
precision mediump float;
uniform vec2 uRes; uniform float uTime; uniform vec3 uColor; uniform float uAmount; uniform float uScroll;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p); vec2 w = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), w.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), w.x), w.y);
}
float fbm(vec2 p) {
  float v = 0.0; float a = 0.5;
  for (int k = 0; k < 5; k++) { v += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5; }
  return v;
}
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p = vec2(uv.x * uRes.x / uRes.y, uv.y) * 2.4;
  p.y -= uScroll * 0.6;
  float t = uTime * 0.03;
  float n = fbm(p + vec2(t, 0.0) + fbm(p * 0.9 - vec2(0.0, t)) * 0.8);
  float a = smoothstep(0.42, 0.9, n) * uAmount;
  gl_FragColor = vec4(uColor * a, a);
}`);
        fgl.useProgram(fogProgram);
        const quad = fgl.createBuffer();
        fgl.bindBuffer(fgl.ARRAY_BUFFER, quad);
        fgl.bufferData(fgl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), fgl.STATIC_DRAW);
        const loc = fgl.getAttribLocation(fogProgram, 'aPos');
        fgl.enableVertexAttribArray(loc);
        fgl.vertexAttribPointer(loc, 2, fgl.FLOAT, false, 0, 0);
        fog = { gl: fgl, u: {} };
        ['uRes', 'uTime', 'uColor', 'uAmount', 'uScroll'].forEach((name) => { fog.u[name] = fgl.getUniformLocation(fogProgram, name); });
      } catch (error) {
        fog = null;
      }
    }
  }

  /* ---------- Anchors ---------- */

  const anchors = anchorEls.map((el) => {
    const shape = Math.max(0, SHAPES.indexOf(el.dataset.shape));
    return {
      el,
      shape,
      palette: PALETTES[el.dataset.palette] ? el.dataset.palette : DEFAULT_PALETTE[SHAPES[shape]],
      fit: el.dataset.fit || 'contain',
      scale: Number(el.dataset.scale) || 1,
      section: el.closest('section, footer') || el,
      // data-hold="anchor" keeps a shape on its anchor (it scrolls away with it) instead of waiting on screen.
      hold: el.dataset.hold || 'screen',
    };
  });

  function frameOf(anchor, rect) {
    const w = rect.width;
    const h = rect.height;
    let size;
    let rotation = 0;
    if (anchor.fit === 'width') size = w / 2;
    else if (anchor.fit === 'height') size = h / 2;
    else if (anchor.fit === 'cover') size = Math.max(w / 4.8, h / 3);
    else if (anchor.fit === 'long') {
      if (h > w) { rotation = -Math.PI / 2; size = h / 2; } else size = w / 2;
    } else size = Math.min(w, h) / 2;
    return { x: rect.left + w / 2, y: rect.top + h / 2, size: size * anchor.scale, rotation };
  }

  /* ---------- Loop ---------- */

  let width = 0;
  let height = 0;
  let dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, small ? 1.5 : 1.75);
    width = window.innerWidth;
    height = window.innerHeight;
    dots.width = Math.round(width * dpr);
    dots.height = Math.round(height * dpr);
    gl.viewport(0, 0, dots.width, dots.height);
    if (fog) {
      fogCanvas.width = Math.ceil(width / 6);
      fogCanvas.height = Math.ceil(height / 6);
      fog.gl.viewport(0, 0, fogCanvas.width, fogCanvas.height);
    }
  }
  resize();
  window.addEventListener('resize', resize);

  const pointer = { x: -1e4, y: -1e4, tx: 0, ty: 0, sx: 0, sy: 0 };
  if (!still && window.matchMedia('(pointer: fine)').matches) {
    window.addEventListener('pointermove', (e) => {
      pointer.x = e.clientX;
      pointer.y = e.clientY;
      pointer.tx = (e.clientX / width - 0.5) * 0.5;
      pointer.ty = (e.clientY / height - 0.5) * 0.3;
    }, { passive: true });
    document.addEventListener('pointerleave', () => { pointer.x = pointer.y = -1e4; pointer.tx = pointer.ty = 0; });
  }

  let dark = isDark();
  const syncTheme = () => { dark = isDark(); };
  new MutationObserver(syncTheme).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
  if (darkQuery.addEventListener) darkQuery.addEventListener('change', syncTheme);

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerp3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

  let running = true;
  let shown = false;
  dots.addEventListener('webglcontextlost', (e) => { e.preventDefault(); running = false; root.classList.remove('has-scene'); });

  // How far each shape reaches above and below its centre, in shape units. Stars cover their
  // whole section and are left where they are.
  const REACH = [[-1.0, 1.0], [-0.16, 0.16], [-0.78, 0.9], [-0.96, 0.96], [-1.02, 0.98], null];
  // Text blocks the particles dim behind.
  const quietEls = Array.from(document.querySelectorAll('[data-quiet], .section-head, .story-text, .dusk-text, .statement-text, .cta-text, .band-text'));
  const quiet = new Float32Array(16);
  // ?debug-scene puts the current pair, frames and sections on window.__scene.
  const debug = /debug-scene/.test(location.search);
  let lastTime = 0;
  let pairFrom = -1;
  let pairTo = -1;
  let shownMix = 0;

  function draw(now) {
    if (!running) return;
    const t = still ? 4 : now / 1000;
    const dt = lastTime ? Math.min(0.1, (now - lastTime) / 1000) : 1 / 60;
    lastTime = now;
    // Anchors hidden at this width (display: none) have no box and are skipped.
    const rects = anchors.map((a) => a.el.getBoundingClientRect());
    const live = [];
    rects.forEach((r, k) => { if (r.width > 0 || r.height > 0) live.push(k); });
    if (live.length === 0) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      requestAnimationFrame(draw);
      return;
    }
    // Hand-over: while the edge between two sections is on screen, each keeps its own shape,
    // and particles cross to the next one in proportion to how much of it shows. Inside one
    // section with two anchors, the edge is halfway between them.
    const focus = height * 0.5;
    const band = height * 0.5;
    const centres = live.map((k) => rects[k].top + rects[k].height / 2);
    const sections = live.map((k) => anchors[k].section.getBoundingClientRect());
    const edges = [];
    for (let k = 0; k < live.length - 1; k++) {
      edges.push(anchors[live[k]].section === anchors[live[k + 1]].section ? (centres[k] + centres[k + 1]) / 2 : sections[k + 1].top);
    }
    let i = 0;
    while (i < edges.length && focus > edges[i] + band) i++;
    const j = Math.min(i + 1, live.length - 1);
    const target = i < edges.length ? clamp((focus - (edges[i] - band)) / (band * 2), 0, 1) : 0;
    const from = live[i];
    const to = live[j];
    let dirty = false;
    if (from !== pairFrom || to !== pairTo) {
      // A new pair shares one anchor with the old one, so particles resting there stay put.
      if (from === pairTo) hop.fill(0);
      else if (to === pairFrom) hop.fill(1);
      else for (let k = 0; k < N; k++) hop[k] = rand[k * 4] < target ? 1 : 0;
      pairFrom = from;
      pairTo = to;
      shownMix = target;
      dirty = true;
    }
    const step = still ? 1 : dt * 1.7;
    for (let k = 0; k < N; k++) {
      const side = rand[k * 4] < target ? 1 : 0;
      const h = hop[k];
      if (h !== side) {
        hop[k] = side > h ? Math.min(1, h + step) : Math.max(0, h - step);
        dirty = true;
      }
    }
    if (dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, hopBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, hop);
    }
    shownMix += (target - shownMix) * (still ? 1 : Math.min(1, dt * 5));
    const mix = shownMix;
    const a = anchors[from];
    const b = anchors[to];
    // A shape stays inside the visible part of its own section while it is in play, so it
    // waits on screen instead of sliding off with an anchor near the section's edge.
    const settle = (k, anchor, frame) => {
      const reach = REACH[anchor.shape];
      if (!reach) return frame;
      const low = frame.rotation ? -1.15 : reach[0];
      const high = frame.rotation ? 1.15 : reach[1];
      const s = sections[k];
      const pad = 12;
      const top = s.top + pad + high * frame.size;
      const bottom = s.bottom - pad + low * frame.size;
      // In a single-column layout a waiting shape would drift over the text stacked around
      // its stage, so below 900px every shape rides its own anchor.
      if (anchor.hold === 'anchor' || width < 900) {
        if (top <= bottom) frame.y = clamp(frame.y, top, bottom);
        return frame;
      }
      let lo = Math.max(top, pad + high * frame.size);
      let hi = Math.min(bottom, height - pad + low * frame.size);
      if (lo > hi) {
        // Taller than the screen: keep at least a band of it showing, as far as the section allows.
        lo = Math.max(top, 90 + low * frame.size);
        hi = Math.min(bottom, height - 90 + high * frame.size);
      }
      if (lo <= hi) frame.y = clamp(frame.y, lo, hi);
      else if (top <= bottom) frame.y = clamp(frame.y, top, bottom);
      return frame;
    };
    const fa = settle(i, a, frameOf(a, rects[from]));
    const fb = settle(j, b, frameOf(b, rects[to]));
    if (debug) window.__scene = { from: SHAPES[a.shape], to: SHAPES[b.shape], target: +target.toFixed(3), a: [Math.round(fa.x), Math.round(fa.y), Math.round(fa.size)], b: [Math.round(fb.x), Math.round(fb.y), Math.round(fb.size)], sa: [Math.round(sections[i].top), Math.round(sections[i].bottom)], sb: [Math.round(sections[j].top), Math.round(sections[j].bottom)] };
    quiet.fill(-1e5);
    let q = 0;
    for (const el of quietEls) {
      if (q === 4) break;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.bottom < 0 || r.top > height) continue;
      quiet[q * 4] = r.left;
      quiet[q * 4 + 1] = r.top;
      quiet[q * 4 + 2] = r.right;
      quiet[q * 4 + 3] = r.bottom;
      q++;
    }
    const pa = PALETTES[a.palette][dark ? 'dark' : 'light'];
    const pb = PALETTES[b.palette][dark ? 'dark' : 'light'];

    pointer.sx += (pointer.tx - pointer.sx) * 0.05;
    pointer.sy += (pointer.ty - pointer.sy) * 0.05;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(u.uRes, width, height);
    gl.uniform1f(u.uDpr, dpr);
    gl.uniform1f(u.uTime, t);
    gl.uniform1f(u.uSize, small ? 2.2 : 2.0);
    gl.uniform1f(u.uFrom, a.shape);
    gl.uniform1f(u.uTo, b.shape);
    gl.uniform4fv(u.uQuiet, quiet);
    gl.uniform3f(u.uFromFrame, fa.x, fa.y, fa.size);
    gl.uniform3f(u.uToFrame, fb.x, fb.y, fb.size);
    gl.uniform2f(u.uRot, fa.rotation, fb.rotation);
    gl.uniform3fv(u.uFromBase, pa.base);
    gl.uniform3fv(u.uFromAccent, pa.accent);
    gl.uniform3fv(u.uToBase, pb.base);
    gl.uniform3fv(u.uToAccent, pb.accent);
    gl.uniform2f(u.uAlpha, pa.alpha, pb.alpha);
    gl.uniform2f(u.uTilt, pointer.sx, pointer.sy);
    gl.uniform2f(u.uPointer, pointer.x, pointer.y);
    const glow = lerp(pa.glow, pb.glow, mix);
    gl.uniform1f(u.uGlow, glow);
    // On dark skies particles add light where they overlap; on light skies they lay colour over.
    if (glow > 0.5) gl.blendFuncSeparate(gl.ONE, gl.ONE, gl.ONE, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, 0, N);

    if (fog) {
      const f = fog.gl;
      f.clearColor(0, 0, 0, 0);
      f.clear(f.COLOR_BUFFER_BIT);
      f.uniform2f(fog.u.uRes, fogCanvas.width, fogCanvas.height);
      f.uniform1f(fog.u.uTime, t);
      f.uniform3fv(fog.u.uColor, lerp3(pa.fog, pb.fog, mix));
      f.uniform1f(fog.u.uAmount, lerp(pa.fogAmount, pb.fogAmount, mix));
      f.uniform1f(fog.u.uScroll, window.scrollY / Math.max(1, height));
      f.drawArrays(f.TRIANGLES, 0, 3);
    }

    if (!shown) {
      shown = true;
      root.classList.add('has-scene');
    }
    if (!still) requestAnimationFrame(draw);
  }
  if (still) {
    let queued = false;
    const redraw = () => { if (!queued) { queued = true; requestAnimationFrame((now) => { queued = false; draw(now); }); } };
    window.addEventListener('scroll', redraw, { passive: true });
    window.addEventListener('resize', redraw);
    new MutationObserver(redraw).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  }
  requestAnimationFrame(draw);
})();
