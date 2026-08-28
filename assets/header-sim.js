// Homepage hero: particles undergoing underdamped Langevin dynamics in the
// score field (∇log density) of "LET'S MODEL IT OUT".
//
// Everything static — the magma density heatmap and the white score arrows —
// is rendered by Makie in examples/export_score_field.jl and shipped as
// assets/hero-field.png. That PNG's pixel box is exactly the field's data
// limits (no axis margins), so mapping data coordinates onto it is a plain
// linear rescale. This file only integrates and draws the particles.
//
// assets/score-field.bin carries the field itself; see the export script for
// the binary schema.
(function () {
  "use strict";

  const container = document.getElementById("hero-sim");
  const bgImage = document.getElementById("hero-sim-bg");
  const fgCanvas = document.getElementById("hero-sim-particles");
  if (!container || !bgImage || !fgCanvas) return;

  // Tunables — picked by eye against the rendered background.
  const PARTICLE_COUNT = 500;
  const GAMMA = 1.1; // friction
  const TEMPERATURE_INITIAL = 0.7; // starting noise scale
  const TEMPERATURE_FINAL = 0.05; // frozen noise scale
  const COOLING_TIME = 30.0; // seconds over which temperature drops
  const SCORE_FORCE_MAX = 90; // just above the exported field's max (~77.5)
  const PARTICLE_COLOR = "#5b8def"; // Makie's :cornflowerblue
  // Particle diameter in the background PNG's own pixels, scaled with the
  // image as it is laid out. 8px there matches Makie's markersize = 4.
  const PARTICLE_DIAMETER = 10;
  const MOUSE_STRENGTH = 6.0;
  const MOUSE_EPS = 0.02;
  const MOUSE_RADIUS = 0.9; // data-space units; beyond this, no repulsion
  const EXPLOSION_STRENGTH = 9.0; // click impulse scale
  const EXPLOSION_HORIZONTAL_SPREAD = 0.0; // sideways fan (0 = purely vertical)
  const EXPLOSION_VERTICAL_BOOST = 1.0; // how hard particles shoot to top/bottom
  const EXPLOSION_EPS = 0.1; // softening for very close particles

  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // The field's data aspect, (xmax-xmin):(ymax-ymin). Fixed by the export
  // script; read out of the header once the field loads, but needed for
  // layout before then.
  let dataAspect = 5.0;

  // Where the particle canvas is on screen, in CSS pixels.
  let display = { dw: 0, dh: 0, radius: 1 };

  // Quarto renders the hero inside <main class="content">, one cell of a grid
  // that also holds the margin sidebar — so a full-bleed hero runs underneath
  // the listing's "Categories" panel. Lift it out to sit above the whole grid,
  // where it shares its row with nothing.
  const quartoContent = document.getElementById("quarto-content");
  if (quartoContent && quartoContent.parentNode) {
    quartoContent.parentNode.insertBefore(container, quartoContent);
  }

  // --- layout ---
  //
  // Measure the container's own offset rather than assuming it, and size to
  // documentElement.clientWidth so a scrollbar doesn't push the page into
  // horizontal overflow the way 100vw would.
  function fullBleed() {
    container.style.marginLeft = "0px";
    container.style.width = "auto";
    const left = container.getBoundingClientRect().left;
    container.style.marginLeft = -left + "px";
    container.style.width = document.documentElement.clientWidth + "px";
  }

  // "Cover"-fit the field into the (wider) hero box, cropping rather than
  // stretching. The image and the canvas get identical boxes, so data
  // coordinates land on the same pixels in both.
  function layout() {
    fullBleed();

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    let dw = cw;
    let dh = dw / dataAspect;
    if (dh < ch) {
      dh = ch;
      dw = dh * dataAspect;
    }

    bgImage.style.width = dw + "px";
    bgImage.style.height = dh + "px";
    fgCanvas.style.width = dw + "px";
    fgCanvas.style.height = dh + "px";

    // Back the canvas at device resolution so the dots stay round, and work
    // in CSS pixels by scaling the context once.
    const dpr = window.devicePixelRatio || 1;
    fgCanvas.width = Math.round(dw * dpr);
    fgCanvas.height = Math.round(dh * dpr);
    const ctx = fgCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const nativeWidth = bgImage.naturalWidth || dw;
    display = {
      dw: dw,
      dh: dh,
      radius: Math.max(1, (PARTICLE_DIAMETER / 2) * (dw / nativeWidth)),
    };
  }

  layout();
  if (window.ResizeObserver) {
    new ResizeObserver(layout).observe(document.documentElement);
  } else {
    window.addEventListener("resize", layout);
  }
  // naturalWidth is 0 until the PNG decodes; re-derive the dot size then.
  if (!bgImage.complete) bgImage.addEventListener("load", layout);

  // Floor-mod (JS `%` returns negative results for negative operands, unlike
  // Julia's `mod`, which `reflect_position` relies on).
  function floorMod(x, m) {
    return ((x % m) + m) % m;
  }

  function reflect(x, lower, upper) {
    const width = upper - lower;
    return lower + width - Math.abs(floorMod(x - lower, 2 * width) - width);
  }

  let elapsed = 0.0;

  function currentTemperature(dt) {
    // Keep the simulation at a constant temperature; no cooling over time.
    return TEMPERATURE_INITIAL;
  }

  let gaussSpare = null;
  function gaussian() {
    if (gaussSpare !== null) {
      const v = gaussSpare;
      gaussSpare = null;
      return v;
    }
    let u1, u2;
    do {
      u1 = Math.random();
    } while (u1 <= Number.EPSILON);
    u2 = Math.random();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    gaussSpare = mag * Math.sin(2.0 * Math.PI * u2);
    return mag * Math.cos(2.0 * Math.PI * u2);
  }

  fetch("assets/score-field.bin")
    .then((r) => r.arrayBuffer())
    .then(init)
    .catch(() => {
      // Field failed to load (offline build preview, blocked fetch, etc).
      // The background PNG still stands on its own; just skip the particles.
    });

  function init(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(
      view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
    );
    if (magic !== "SFLD") return;

    const nx = view.getInt32(4, true);
    const ny = view.getInt32(8, true);
    const xmin = view.getFloat32(12, true);
    const xmax = view.getFloat32(16, true);
    const ymin = view.getFloat32(20, true);
    const ymax = view.getFloat32(24, true);
    // Offsets 28 (uniform_floor) and the u block below it are what the export
    // script colours the background with; the browser no longer needs either.

    const n = nx * ny;
    let offset = 32 + n * 4; // skip u
    const sx = new Float32Array(buffer, offset, n); offset += n * 4;
    const sy = new Float32Array(buffer, offset, n);

    const dgrid = (xmax - xmin) / (nx - 1);
    dataAspect = (xmax - xmin) / (ymax - ymin);
    layout();

    // field[j*nx+i], i = x-index, j = y-index (see export script).
    function sample(field, x, y) {
      let i = Math.floor((x - xmin) / dgrid);
      let j = Math.floor((y - ymin) / dgrid);
      i = Math.min(Math.max(i, 0), nx - 2);
      j = Math.min(Math.max(j, 0), ny - 2);
      const a = (x - (xmin + i * dgrid)) / dgrid;
      const b = (y - (ymin + j * dgrid)) / dgrid;
      const f00 = field[j * nx + i];
      const f10 = field[j * nx + i + 1];
      const f01 = field[(j + 1) * nx + i];
      const f11 = field[(j + 1) * nx + i + 1];
      return (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 +
        (1 - a) * b * f01 + a * b * f11;
    }

    const fgCtx = fgCanvas.getContext("2d");

    // --- pointer repulsion ---
    let pointerX = null, pointerY = null;
    container.addEventListener("pointermove", (evt) => {
      const rect = container.getBoundingClientRect();
      const px = evt.clientX - rect.left - (rect.width - display.dw) / 2;
      const py = evt.clientY - rect.top - (rect.height - display.dh) / 2;
      pointerX = xmin + (px / display.dw) * (xmax - xmin);
      pointerY = ymax - (py / display.dh) * (ymax - ymin);
    });
    container.addEventListener("pointerleave", () => {
      pointerX = null;
      pointerY = null;
    });

    container.addEventListener("click", (evt) => {
      const rect = container.getBoundingClientRect();
      const cx = evt.clientX - rect.left - (rect.width - display.dw) / 2;
      const cy = evt.clientY - rect.top - (rect.height - display.dh) / 2;
      const clickX = xmin + (cx / display.dw) * (xmax - xmin);
      const clickY = ymax - (cy / display.dh) * (ymax - ymin);

      // Reheat the system.
      elapsed = 0.0;

      // Vertical-biased radial explosion: particles shoot mostly toward the
      // top and bottom of the hero, clearing the center wordmark, while still
      // fanning out a little sideways.
      for (let k = 0; k < PARTICLE_COUNT; k++) {
        const dx = px[k] - clickX;
        const dy = py[k] - clickY;
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r < EXPLOSION_EPS) continue;
        const ux = dx / r;
        const uy = dy / r;
        const bx = ux * EXPLOSION_HORIZONTAL_SPREAD;
        const by = uy * EXPLOSION_VERTICAL_BOOST + (dy >= 0 ? 0.1 : -0.1);
        const bMag = Math.sqrt(bx * bx + by * by);
        const scale = EXPLOSION_STRENGTH / bMag;
        vx[k] += bx * scale;
        vy[k] += by * scale;
      }
    });

    // --- particles (structure-of-arrays, reused every frame) ---
    const px = new Float32Array(PARTICLE_COUNT);
    const py = new Float32Array(PARTICLE_COUNT);
    const vx = new Float32Array(PARTICLE_COUNT);
    const vy = new Float32Array(PARTICLE_COUNT);
    for (let k = 0; k < PARTICLE_COUNT; k++) {
      px[k] = xmin + Math.random() * (xmax - xmin);
      py[k] = ymin + Math.random() * (ymax - ymin);
    }

    function step(dt) {
      const temperature = currentTemperature(dt);
      const noise = Math.sqrt(2 * GAMMA * temperature * dt);
      for (let k = 0; k < PARTICLE_COUNT; k++) {
        let fx = sample(sx, px[k], py[k]);
        let fy = sample(sy, px[k], py[k]);
        const fmag = Math.hypot(fx, fy);
        if (fmag > SCORE_FORCE_MAX) {
          const s = SCORE_FORCE_MAX / fmag;
          fx *= s;
          fy *= s;
        }
        if (pointerX !== null) {
          const dx = px[k] - pointerX;
          const dy = py[k] - pointerY;
          const r2 = dx * dx + dy * dy;
          if (r2 < MOUSE_RADIUS * MOUSE_RADIUS) {
            const inv = MOUSE_STRENGTH / (r2 + MOUSE_EPS);
            fx += dx * inv;
            fy += dy * inv;
          }
        }
        vx[k] += (fx - GAMMA * vx[k]) * dt + noise * gaussian();
        vy[k] += (fy - GAMMA * vy[k]) * dt + noise * gaussian();
        px[k] = reflect(px[k] + vx[k] * dt, xmin, xmax);
        py[k] = reflect(py[k] + vy[k] * dt, ymin, ymax);
      }
    }

    function draw() {
      const { dw, dh, radius } = display;
      fgCtx.clearRect(0, 0, dw, dh);
      fgCtx.fillStyle = PARTICLE_COLOR;
      const tau = 2 * Math.PI;
      for (let k = 0; k < PARTICLE_COUNT; k++) {
        const cx = ((px[k] - xmin) / (xmax - xmin)) * dw;
        const cy = dh - ((py[k] - ymin) / (ymax - ymin)) * dh;
        fgCtx.beginPath();
        fgCtx.arc(cx, cy, radius, 0, tau);
        fgCtx.fill();
      }
    }

    if (reduceMotion) {
      // Settle the particles onto the wordmark, then hold them there.
      for (let k = 0; k < 600; k++) step(1 / 60);
      draw();
      return;
    }

    let last = null;
    function frame(t) {
      if (last === null) last = t;
      const dt = Math.min((t - last) / 1000, 1 / 30);
      last = t;
      step(dt);
      draw();
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
})();
