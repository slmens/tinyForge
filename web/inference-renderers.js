/* ═══════════════════════════════════════════════════════
   TinyForge — inference-renderers.js

   Canvas animation renderers for each ONNX op type.
   Each renderer takes real activation data from the
   inference run and visualizes it step-by-step.
   ═══════════════════════════════════════════════════════ */

// ── Viridis colormap (purple→blue→teal→green→yellow) ──

const INF_VIRIDIS = (function () {
  const stops = [
    [68,1,84],[72,20,103],[67,44,122],[57,67,133],
    [45,89,141],[36,109,146],[28,128,149],[22,145,152],
    [21,163,149],[30,179,139],[53,191,123],[85,199,103],
    [120,207,81],[160,213,58],[199,216,38],[234,214,26],
    [253,206,20],[253,183,20],[246,157,29],[234,129,47],
  ];
  const out = [];
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    out.push([
      Math.round(stops[lo][0] + f * (stops[hi][0] - stops[lo][0])),
      Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1])),
      Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2])),
    ]);
  }
  return out;
})();

// ── Easing helpers ────────────────────────────────────

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function easeInOutCubic(t) { return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }
function lerp(a, b, t) { return a + (b - a) * t; }

// ── Normalize array to [0,1] ──────────────────────────

function normalizeArr(arr) {
  let mn = arr[0], mx = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
  }
  const range = mx - mn || 1;
  return { norm: Array.from(arr).map(v => (v - mn) / range), mn, mx };
}

// ── Canvas helpers ────────────────────────────────────

function clearCanvas(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060610';
  ctx.fillRect(0, 0, w, h);
}

// Draw a sampling note in the bottom-right corner
function drawSamplingNote(ctx, W, H, text) {
  ctx.save();
  ctx.font = `400 10px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = 'rgba(232,232,240,0.45)';
  ctx.fillText(text, W - 12, H - 8);
  ctx.restore();
}

// Format a number with commas
function fmtNum(n) {
  return n.toLocaleString();
}

function viridisColor(v) {
  const idx = Math.max(0, Math.min(255, Math.round(v * 255)));
  const c = INF_VIRIDIS[idx];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function viridisColorAlpha(v, a) {
  const idx = Math.max(0, Math.min(255, Math.round(v * 255)));
  const c = INF_VIRIDIS[idx];
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

// Average tensor across channel dimension → 2D float array
function avgChannels(data, C, H, W) {
  const out = new Float32Array(H * W);
  for (let c = 0; c < C; c++) {
    const base = c * H * W;
    for (let i = 0; i < H * W; i++) out[i] += data[base + i];
  }
  for (let i = 0; i < H * W; i++) out[i] /= C;
  return out;
}

// Extract a centered patch from a 2D spatial array
function centerPatch(spatial, H, W, size) {
  const s = Math.min(size, H, W);
  const sy = Math.floor((H - s) / 2);
  const sx = Math.floor((W - s) / 2);
  const patch = new Float32Array(s * s);
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      patch[y * s + x] = spatial[(sy + y) * W + (sx + x)];
    }
  }
  return { patch, s };
}

// ── Base renderer ─────────────────────────────────────

class BaseRenderer {
  constructor(canvas, speedGetter) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.W = canvas.width;
    this.H = canvas.height;
    this.speedGetter = speedGetter;
    this._raf = null;
    this._resolve = null;
    this._reject = null;
    this._startTime = null;
    this._cancelled = false;
  }

  // Override in subclasses — duration before speed adjustment
  get baseDuration() { return 2000; }

  get duration() { return this.baseDuration / (this.speedGetter ? this.speedGetter() : 1); }

  run() {
    return new Promise((resolve, reject) => {
      this._resolve = resolve;
      this._reject = reject;
      this._cancelled = false;
      this._startTime = null;
      this._tick = (ts) => {
        if (this._cancelled) return;
        if (this._startTime === null) this._startTime = ts;
        const elapsed = ts - this._startTime;
        const dur = this.baseDuration / (this.speedGetter ? this.speedGetter() : 1);
        const p = Math.min(1, elapsed / dur);
        this.draw(p);
        if (p < 1) {
          this._raf = requestAnimationFrame(this._tick);
        } else {
          this.draw(1);
          resolve();
        }
      };
      this._raf = requestAnimationFrame(this._tick);
    });
  }

  cancel() {
    this._cancelled = true;
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._resolve) { this.draw(1); this._resolve(); }
  }

  draw(progress) { /* override */ }
}

// ── ReshapeRenderer ───────────────────────────────────
// Minimal: just shows the shape transform text

class ReshapeRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);
    this.inputShape = inputs[0] ? inputs[0].shape : [];
    this.outputShape = output.shape;
  }

  get baseDuration() { return 800; }

  draw(p) {
    const { ctx, W, H } = this;
    clearCanvas(ctx, W, H);

    const cx = W / 2, cy = H / 2;
    const t = easeOutCubic(p);

    // Input shape
    const inStr = '[' + this.inputShape.join(', ') + ']';
    const outStr = '[' + this.outputShape.join(', ') + ']';

    ctx.font = `400 18px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Fade out input shape
    const inAlpha = p < 0.5 ? 1 : 1 - (p - 0.5) * 2;
    ctx.fillStyle = `rgba(232,232,240,${inAlpha * 0.6})`;
    ctx.fillText(inStr, cx, cy - 20);

    // Arrow
    ctx.fillStyle = `rgba(148,163,184,${Math.min(1, p * 3)})`;
    ctx.fillText('↓', cx, cy + 8);

    // Fade in output shape
    const outAlpha = p < 0.5 ? 0 : (p - 0.5) * 2;
    ctx.fillStyle = `rgba(232,232,240,${outAlpha * 0.9})`;
    ctx.fillText(outStr, cx, cy + 36);

    // Label
    ctx.font = `500 12px 'Syne', sans-serif`;
    ctx.fillStyle = `rgba(148,163,184,${Math.min(1, p * 2) * 0.7})`;
    ctx.fillText('reshape', cx, cy - 52);
  }
}

// ── ClipRenderer ──────────────────────────────────────
// Show a row of values being clamped to [0, 6]

class ClipRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);
    const data = inputs[0] ? inputs[0].data : new Float32Array(12);
    const total = data.length;

    // Sample 14 interesting values — prefer ones with out-of-range
    const candidates = [];
    for (let i = 0; i < total; i++) candidates.push({ v: data[i], i });
    // Sort: negative first, then >6, then rest
    candidates.sort((a, b) => {
      const aOOB = a.v < 0 || a.v > 6 ? 1 : 0;
      const bOOB = b.v < 0 || b.v > 6 ? 1 : 0;
      return bOOB - aOOB;
    });
    this.values = candidates.slice(0, 14).map(c => c.v);
    this.N = this.values.length;
    this.outData = output ? output.data : data;

    const sp = inputs[0] ? inputs[0].shape : [1, 1, 1, 1];
    const spatial = sp.length === 4 ? sp[2] : 1;
    this._spatial = spatial;
  }

  get baseDuration() {
    return this._spatial >= 56 ? 2200 : this._spatial >= 14 ? 1600 : 1000;
  }

  draw(p) {
    const { ctx, W, H, values, N } = this;
    clearCanvas(ctx, W, H);

    const cx = W / 2, cy = H / 2;
    const cellW = 54, cellH = 48, gap = 8;
    const totalW = N * cellW + (N - 1) * gap;
    const startX = cx - totalW / 2;
    const MIN_VAL = 0, MAX_VAL = 6;

    // Phase 1 (0-0.35): show raw values, highlight OOB ones red
    // Phase 2 (0.35-0.75): values animate toward clamped position
    // Phase 3 (0.75-1.0): show clamped values settled, green tint

    ctx.font = `400 11px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Title
    ctx.font = `500 11px 'Syne', sans-serif`;
    ctx.fillStyle = `rgba(232,232,240,0.25)`;
    ctx.fillText('ReLU6 — clamp to [0, 6]', cx, 28);

    // Range indicator
    ctx.font = `300 10px 'JetBrains Mono', monospace`;
    ctx.fillStyle = `rgba(232,232,240,0.15)`;
    ctx.fillText('min = 0                max = 6', cx, H - 28);

    for (let i = 0; i < N; i++) {
      const rawVal = values[i];
      const clampedVal = Math.max(MIN_VAL, Math.min(MAX_VAL, rawVal));
      const isOOB = rawVal < MIN_VAL || rawVal > MAX_VAL;
      const x = startX + i * (cellW + gap) + cellW / 2;

      // Animate the displayed value
      let displayVal;
      if (p < 0.35) {
        displayVal = rawVal;
      } else if (p < 0.75) {
        const t = easeOutCubic((p - 0.35) / 0.4);
        displayVal = lerp(rawVal, clampedVal, t);
      } else {
        displayVal = clampedVal;
      }

      // Cell background
      let cellAlpha = 0.06;
      let borderColor = 'rgba(255,255,255,0.08)';

      if (isOOB && p < 0.75) {
        const pulse = Math.sin(p * Math.PI * 6) * 0.5 + 0.5;
        const glowStrength = p < 0.35 ? 0.3 + pulse * 0.2 : 0.3 * (1 - (p - 0.35) / 0.4);
        cellAlpha = glowStrength * 0.15;
        borderColor = rawVal < 0
          ? `rgba(248,113,113,${glowStrength})`
          : `rgba(251,211,77,${glowStrength})`;
      } else if (p >= 0.75) {
        // Settled green
        cellAlpha = 0.08;
        borderColor = 'rgba(52,211,153,0.2)';
      }

      // Draw cell
      ctx.fillStyle = `rgba(255,255,255,${cellAlpha})`;
      roundRect(ctx, x - cellW/2, cy - cellH/2, cellW, cellH, 6);
      ctx.fill();
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1;
      roundRect(ctx, x - cellW/2, cy - cellH/2, cellW, cellH, 6);
      ctx.stroke();

      // Value text
      let textColor;
      if (isOOB && p < 0.35) {
        textColor = rawVal < 0 ? 'rgba(248,113,113,0.9)' : 'rgba(251,211,77,0.9)';
      } else if (p >= 0.75) {
        textColor = 'rgba(52,211,153,0.8)';
      } else {
        textColor = 'rgba(232,232,240,0.7)';
      }
      ctx.fillStyle = textColor;
      ctx.font = `400 11px 'JetBrains Mono', monospace`;
      ctx.fillText(displayVal.toFixed(2), x, cy - 4);

      // OOB arrow indicator in phase 2
      if (isOOB && p > 0.35 && p < 0.75) {
        const arrowAlpha = 1 - (p - 0.35) / 0.4;
        ctx.fillStyle = `rgba(232,232,240,${arrowAlpha * 0.5})`;
        ctx.font = `300 10px 'JetBrains Mono', monospace`;
        ctx.fillText(rawVal < 0 ? '▲0' : '▼6', x, cy + 14);
      }
    }

    // Sampling note
    const totalVals = this.outData.length;
    drawSamplingNote(ctx, W, H,
      `showing ${this.N} of ${fmtNum(totalVals)} values`
    );
  }
}

// ── AddRenderer ───────────────────────────────────────
// Two rows of numbers merge into one (residual addition)

class AddRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);
    const N = 8;
    const dataA = inputs[0] ? inputs[0].data : new Float32Array(N);
    const dataB = inputs[1] ? inputs[1].data : new Float32Array(N);
    const step = Math.max(1, Math.floor(dataA.length / N));
    this.valA = Array.from({length: N}, (_, i) => dataA[i * step] || 0);
    this.valB = Array.from({length: N}, (_, i) => dataB[i * step] || 0);
    this.valC = this.valA.map((a, i) => a + this.valB[i]);
    this.N = N;
    this._totalVals = dataA.length;

    const sp = inputs[0] ? inputs[0].shape : [1, 1, 1, 1];
    this._spatial = sp.length === 4 ? sp[2] : 1;
  }

  get baseDuration() {
    return this._spatial >= 56 ? 1600 : this._spatial >= 14 ? 1200 : 900;
  }

  draw(p) {
    const { ctx, W, H, valA, valB, valC, N } = this;
    clearCanvas(ctx, W, H);

    const cx = W / 2, cy = H / 2;
    const cellW = 60, gap = 10;
    const totalW = N * cellW + (N - 1) * gap;
    const startX = cx - totalW / 2;

    // Title
    ctx.font = `500 11px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(232,232,240,0.25)';
    ctx.fillText('residual add — shortcut + main path', cx, 24);

    // Phase 0-0.3: show both rows
    // Phase 0.3-0.7: rows slide toward each other
    // Phase 0.7-1.0: result row appears

    // Row Y positions
    const rowSpacing = 52;
    let yA, yB;
    if (p < 0.3) {
      yA = cy - rowSpacing;
      yB = cy + rowSpacing;
    } else if (p < 0.7) {
      const t = easeInOutCubic((p - 0.3) / 0.4);
      yA = lerp(cy - rowSpacing, cy - 10, t);
      yB = lerp(cy + rowSpacing, cy + 10, t);
    } else {
      yA = cy - 10;
      yB = cy + 10;
    }

    const rowAlpha = p < 0.7 ? 1 : 1 - easeOutCubic((p - 0.7) / 0.3);
    const resAlpha = p < 0.7 ? 0 : easeOutCubic((p - 0.7) / 0.3);

    ctx.font = `400 11px 'JetBrains Mono', monospace`;
    ctx.textBaseline = 'middle';

    // Labels
    if (p < 0.65) {
      ctx.font = `400 10px 'Syne', sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(248,113,113,${rowAlpha * 0.6})`;
      ctx.fillText('shortcut', 20, yA);
      ctx.fillStyle = `rgba(96,165,250,${rowAlpha * 0.6})`;
      ctx.fillText('main', 20, yB);
    }

    for (let i = 0; i < N; i++) {
      const x = startX + i * (cellW + gap) + cellW / 2;

      // Row A
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(248,113,113,${rowAlpha * 0.75})`;
      ctx.font = `400 11px 'JetBrains Mono', monospace`;
      ctx.fillText(valA[i].toFixed(2), x, yA);

      // Row B
      ctx.fillStyle = `rgba(96,165,250,${rowAlpha * 0.75})`;
      ctx.fillText(valB[i].toFixed(2), x, yB);

      // Plus sign between them (when close together)
      if (p > 0.4 && p < 0.7) {
        const plusAlpha = (p - 0.4) / 0.3 * rowAlpha;
        ctx.fillStyle = `rgba(232,232,240,${plusAlpha * 0.4})`;
        ctx.font = `300 10px 'JetBrains Mono', monospace`;
        ctx.fillText('+', x, (yA + yB) / 2);
      }

      // Result row
      if (resAlpha > 0) {
        ctx.fillStyle = `rgba(52,211,153,${resAlpha * 0.9})`;
        ctx.font = `500 12px 'JetBrains Mono', monospace`;
        ctx.fillText(valC[i].toFixed(2), x, cy);
      }
    }

    // Sampling note
    drawSamplingNote(ctx, W, H,
      `showing ${this.N} of ${fmtNum(this._totalVals)} values per tensor`
    );
  }
}

// ── ReduceMeanRenderer ────────────────────────────────
// Grid collapsing to single values per channel

class ReduceMeanRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);
    const inp = inputs[0];
    if (inp && inp.shape.length === 4) {
      const [, C, H, W] = inp.shape;
      this._totalC = C;
      this.C = Math.min(C, 8);
      this.H = H; this.W = W;
      // Sample values for display
      this.grid = [];
      const hw = H * W;
      for (let c = 0; c < this.C; c++) {
        const row = [];
        for (let i = 0; i < hw; i++) row.push(inp.data[c * hw + i]);
        this.grid.push(row);
      }
    } else {
      this._totalC = 4;
      this.C = 4; this.H = 7; this.W = 7;
      this.grid = Array.from({length: 4}, () => Array.from({length: 49}, () => Math.random()));
    }
    this.means = this.grid.map(row => row.reduce((a, b) => a + b, 0) / row.length);
    this._norms = this.grid.map(row => normalizeArr(row).norm);
    this._meanNorm = normalizeArr(this.means).norm;
  }

  get baseDuration() { return 3200; }

  draw(p) {
    const { ctx, W: CW, H: CH, C, H, Wgrid = this.W } = this;
    clearCanvas(ctx, CW, CH);

    // Phase 0-0.5: show grids with pulse effect
    // Phase 0.5-0.8: grids shrink / collapse to center
    // Phase 0.8-1.0: mean value appears

    const cx = CW / 2, cy = CH / 2;
    const cellC = Math.min(C, 6); // show max 6 channels
    const padding = 20;
    const availW = CW - padding * 2;
    const colW = Math.floor(availW / cellC);
    const colH = 90;
    const gridSize = Math.min(H, Wgrid, 7);
    const cellSize = Math.min(14, Math.floor(colW / gridSize) - 1);

    ctx.font = `500 11px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(232,232,240,0.25)';
    ctx.fillText('global average pool — spatial → scalar', cx, 24);

    for (let c = 0; c < cellC; c++) {
      const colX = padding + c * colW + colW / 2;
      const norm = this._norms[c];
      const gridW = gridSize * cellSize + (gridSize - 1);
      const gridStartX = colX - gridW / 2;
      const gridStartY = cy - colH / 2 + 10;

      // Collapse scale
      let scale = 1;
      if (p > 0.5) {
        scale = 1 - easeInOutCubic((p - 0.5) / 0.3) * 0.5;
      }
      if (p > 0.8) scale = Math.max(0.01, scale - easeOutCubic((p - 0.8) / 0.2) * 0.5);

      ctx.save();
      ctx.translate(colX, cy);
      ctx.scale(scale, scale);
      ctx.translate(-colX, -cy);

      // Draw mini grid
      const alpha = p < 0.8 ? 1 : 1 - (p - 0.8) / 0.2;
      for (let gy = 0; gy < gridSize; gy++) {
        for (let gx = 0; gx < gridSize; gx++) {
          const idx = gy * gridSize + gx;
          const v = idx < norm.length ? norm[idx] : 0.5;
          const gxPx = gridStartX + gx * (cellSize + 1);
          const gyPx = gridStartY + gy * (cellSize + 1);
          const viridis = INF_VIRIDIS[Math.round(v * 255)];
          ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${alpha})`;
          ctx.fillRect(gxPx, gyPx, cellSize, cellSize);
        }
      }
      ctx.restore();

      // Mean value appears
      if (p > 0.75) {
        const showAlpha = easeOutCubic((p - 0.75) / 0.25);
        const meanNorm = this._meanNorm[c] !== undefined ? this._meanNorm[c] : 0.5;
        const viridis = INF_VIRIDIS[Math.round(meanNorm * 255)];
        ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${showAlpha})`;
        ctx.font = `500 13px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(this.means[c].toFixed(3), colX, cy + 52);

        ctx.fillStyle = `rgba(232,232,240,${showAlpha * 0.3})`;
        ctx.font = `400 9px 'Syne', sans-serif`;
        ctx.fillText(`ch ${c}`, colX, cy + 66);
      }
    }

    // Sampling note
    const totalVals = this._totalC * this.H * this.W;
    drawSamplingNote(ctx, CW, CH,
      `showing ${this.C} of ${fmtNum(this._totalC)} channels · ${fmtNum(totalVals)} total values`
    );
  }
}

// ── GemmRenderer ──────────────────────────────────────
// Row × column dot product, cell-by-cell fill

class GemmRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);
    const A = inputs[0];
    const B = inputs[1];
    // A: [1, K=1280], B: [N=1000, K=1280]
    this.K = A ? A.shape[1] : 8;
    this.N = B ? B.shape[0] : 8;
    this.dataA = A ? A.data : new Float32Array(8);
    this.dataB = B ? B.data : new Float32Array(64);
    this.outputData = output ? output.data : new Float32Array(8);

    // Sample slices for display
    this.showK = 10;
    this.showN = 6;
    const kStep = Math.max(1, Math.floor(this.K / this.showK));
    const nStep = Math.max(1, Math.floor(this.N / this.showN));
    this.sampleA = Array.from({length: this.showK}, (_, i) => this.dataA[i * kStep] || 0);
    this.sampleB = Array.from({length: this.showN}, (_, n) =>
      Array.from({length: this.showK}, (_, k) => {
        const nIdx = n * nStep, kIdx = k * kStep;
        return this.dataB[nIdx * this.K + kIdx] || 0;
      })
    );
    this.sampleOut = Array.from({length: this.showN}, (_, i) => this.outputData[i * nStep] || 0);
    const { norm: outNorm } = normalizeArr(this.sampleOut);
    this.outNorm = outNorm;
  }

  get baseDuration() { return 4000; }

  draw(p) {
    const { ctx, W, H } = this;
    clearCanvas(ctx, W, H);

    const { showK, showN, sampleA, sampleB, sampleOut, outNorm } = this;
    const cx = W / 2, cy = H / 2;

    ctx.font = `500 11px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(232,232,240,0.25)';
    ctx.fillText('matrix multiply — 1280 features → 1000 class scores', cx, 24);

    // Layout
    const cellW = 46, cellH = 32, gap = 3;
    const aH = cellH, aW = showK * (cellW + gap);
    const bH = showN * (cellH + gap), bW = showK * (cellW + gap);

    const aX = cx - aW / 2;
    const aY = cy - 100;
    const bX = cx - bW / 2;
    const bY = aY + aH + 36;
    const outX = cx;
    const outY = bY + bH + 36;

    // Phase 0-0.25: show A row
    // Phase 0.25-0.65: show B columns, compute highlighted
    // Phase 0.65-1.0: output bar fills in

    const aAlpha = Math.min(1, p * 4);

    // ── Row A (input features)
    ctx.font = `400 10px 'JetBrains Mono', monospace`;
    ctx.textBaseline = 'middle';
    const { norm: aNorm } = normalizeArr(sampleA);
    for (let k = 0; k < showK; k++) {
      const x = aX + k * (cellW + gap);
      const v = aNorm[k];
      const viridis = INF_VIRIDIS[Math.round(v * 255)];
      ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${aAlpha * 0.8})`;
      roundRect(ctx, x, aY, cellW, aH, 4); ctx.fill();

      if (cellW > 30) {
        ctx.fillStyle = `rgba(232,232,240,${aAlpha * 0.7})`;
        ctx.textAlign = 'center';
        ctx.fillText(sampleA[k].toFixed(1), x + cellW/2, aY + aH/2);
      }
    }

    // Label
    ctx.fillStyle = `rgba(232,232,240,${aAlpha * 0.35})`;
    ctx.font = `400 9px 'Syne', sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(`A [1, ${this.K}]`, aX - 8, aY + aH/2);

    if (p < 0.25) return;

    // ── Matrix B (weights)
    const bAlpha = Math.min(1, (p - 0.25) * 4);
    const activeCol = Math.floor((p - 0.25) / 0.4 * showN);

    for (let n = 0; n < showN; n++) {
      for (let k = 0; k < showK; k++) {
        const x = bX + k * (cellW + gap);
        const y = bY + n * (cellH + gap);
        const v = (sampleB[n][k] - Math.min(...sampleB[n])) /
                  (Math.max(...sampleB[n]) - Math.min(...sampleB[n]) + 1e-6);

        const isActive = n === Math.min(activeCol, showN - 1);
        const viridis = INF_VIRIDIS[Math.round(v * 255)];
        const alpha = bAlpha * (isActive ? 0.9 : 0.45);
        ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${alpha})`;
        roundRect(ctx, x, y, cellW, cellH, 3); ctx.fill();
      }
    }

    ctx.fillStyle = `rgba(232,232,240,${bAlpha * 0.35})`;
    ctx.font = `400 9px 'Syne', sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(`B [${this.N}, ${this.K}]`, bX - 8, bY + bH/2);

    if (p < 0.65) return;

    // ── Output values
    const outAlpha = easeOutCubic((p - 0.65) / 0.35);
    const showCount = Math.ceil(outAlpha * showN);

    ctx.font = `500 12px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    for (let n = 0; n < showCount; n++) {
      const v = outNorm[n];
      const viridis = INF_VIRIDIS[Math.round(v * 255)];
      const barW = 60 + v * 80;
      const y = outY + n * (cellH + gap);
      ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${outAlpha * 0.7})`;
      roundRect(ctx, cx - barW/2, y, barW, cellH, 4); ctx.fill();

      ctx.fillStyle = `rgba(232,232,240,${outAlpha * 0.6})`;
      ctx.fillText(sampleOut[n].toFixed(2), cx, y + cellH/2);
    }

    ctx.fillStyle = `rgba(232,232,240,${outAlpha * 0.35})`;
    ctx.font = `400 9px 'Syne', sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(`output [1, ${this.N}]`, outX - 40 - 8, outY + (showN * (cellH + gap)) / 2);

    // Sampling note
    drawSamplingNote(ctx, W, H,
      `showing ${this.showK} of ${fmtNum(this.K)} features · ${this.showN} of ${fmtNum(this.N)} classes`);
  }
}

// ── ConvRenderer ──────────────────────────────────────
// The star. Spatial grid + sliding kernel + MAC text + output heatmap

class ConvRenderer extends BaseRenderer {
  constructor(canvas, node, inputs, output, speedGetter) {
    super(canvas, speedGetter);

    const inp = inputs[0];
    const weight = inputs[1];

    const inShape = inp ? inp.shape : [1, 1, 8, 8];
    const [, C_in, H_in, W_in] = inShape;
    const wShape = weight ? weight.shape : [1, 1, 3, 3];
    const [C_out, ICg, KH, KW] = wShape;

    this.C_in = C_in; this.H_in = H_in; this.W_in = W_in;
    this.KH = KH; this.KW = KW;
    this.C_out = C_out;

    const attrs = node.attrs || {};
    this.SH = attrs.strides ? attrs.strides[0] : 1;
    this.SW = attrs.strides ? attrs.strides[1] : 1;

    const outShape = output ? output.shape : [1, 1, 8, 8];
    const [, C_out2, H_out, W_out] = outShape;
    this.H_out = H_out; this.W_out = W_out;

    // Build display patch from input (channel-averaged, centered 8x8)
    const spatial2D = inp ? avgChannels(inp.data, C_in, H_in, W_in) : new Float32Array(64);
    const PATCH = Math.min(8, H_in, W_in);
    const { patch, s } = centerPatch(spatial2D, H_in, W_in, PATCH);
    this.patch = patch;
    this.patchSize = s;
    const { norm: patchNorm } = normalizeArr(patch);
    this.patchNorm = patchNorm;

    // Extract first kernel (averaged across input channels)
    const kernelFlat = new Float32Array(KH * KW);
    if (weight) {
      for (let ic = 0; ic < ICg; ic++) {
        for (let j = 0; j < KH * KW; j++) {
          kernelFlat[j] += weight.data[ic * KH * KW + j];
        }
      }
      for (let j = 0; j < KH * KW; j++) kernelFlat[j] /= ICg;
    }
    this.kernel = kernelFlat;
    const { norm: kernelNorm } = normalizeArr(kernelFlat);
    this.kernelNorm = kernelNorm;

    // Output activation for heatmap
    this.outputData = output ? output.data : new Float32Array(H_out * W_out);
    const outSpatial = output ? avgChannels(output.data, C_out2, H_out, W_out) : new Float32Array(H_out * W_out);
    const { norm: outNorm } = normalizeArr(outSpatial);
    this.outNorm = outNorm;
    this.outH = H_out; this.outW = W_out;

    // Positions to animate
    const numSlowPositions = 5;
    const positions = [];
    const maxPos = Math.min(s - KH, s - KW);
    for (let ky = 0; ky <= maxPos; ky++) {
      for (let kx = 0; kx <= maxPos; kx++) {
        positions.push([ky, kx]);
      }
    }
    this.positions = positions;
    this.numSlow = Math.min(numSlowPositions, positions.length);

    this._spatial = H_in;
  }

  get baseDuration() {
    const s = this._spatial;
    if (s >= 112) return 6000;
    if (s >= 56)  return 5000;
    if (s >= 28)  return 3500;
    if (s >= 14)  return 2800;
    return 2000;
  }

  draw(p) {
    const { ctx, W, H } = this;
    clearCanvas(ctx, W, H);

    const { patchSize, patchNorm, kernel, kernelNorm, KH, KW, positions, numSlow } = this;

    // Phase 1 (0-0.45): kernel slides over input, MAC text
    // Phase 2 (0.45-0.70): kernel zips across remaining
    // Phase 3 (0.70-1.00): output heatmap fades in

    // Layout
    const gridLeft = 40;
    const cellSize = Math.min(46, Math.floor((W * 0.5 - gridLeft - 20) / patchSize));
    const gridW = patchSize * cellSize;
    const gridH = patchSize * cellSize;
    const gridX = gridLeft;
    const gridY = (H - gridH) / 2 + 10;

    const textX = gridLeft + gridW + 40;
    const textW = W - textX - 24;

    // Draw input patch grid
    for (let gy = 0; gy < patchSize; gy++) {
      for (let gx = 0; gx < patchSize; gx++) {
        const v = patchNorm[gy * patchSize + gx];
        const viridis = INF_VIRIDIS[Math.round(v * 255)];
        ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},0.75)`;
        ctx.fillRect(gridX + gx * cellSize, gridY + gy * cellSize, cellSize - 1, cellSize - 1);
      }
    }

    // Input label
    ctx.fillStyle = 'rgba(232,232,240,0.25)';
    ctx.font = `400 9px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`input  [${this.H_in}×${this.W_in}, C=${this.C_in}]`, gridX + gridW/2, gridY - 14);

    // ── Phase 1 & 2: sliding kernel ──
    if (p < 0.70) {
      const phaseP = p < 0.45 ? p / 0.45 : 1;
      const phase2P = p > 0.45 ? (p - 0.45) / 0.25 : 0;

      // Which position are we at?
      let posIdx;
      if (p < 0.45) {
        // Slow phase: animate through numSlow positions
        posIdx = Math.min(numSlow - 1, Math.floor(phaseP * numSlow));
      } else {
        // Fast phase: zip through all remaining
        posIdx = numSlow + Math.floor(phase2P * (positions.length - numSlow));
      }
      posIdx = Math.min(posIdx, positions.length - 1);

      const [ky, kx] = positions[posIdx];

      // Kernel overlay on input grid
      ctx.strokeStyle = 'rgba(251,211,77,0.85)';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        gridX + kx * cellSize - 1,
        gridY + ky * cellSize - 1,
        KW * cellSize + 1,
        KH * cellSize + 1
      );

      // Highlight kernel cells with subtle glow
      for (let j = 0; j < KH; j++) {
        for (let i = 0; i < KW; i++) {
          ctx.fillStyle = 'rgba(251,211,77,0.12)';
          ctx.fillRect(
            gridX + (kx + i) * cellSize,
            gridY + (ky + j) * cellSize,
            cellSize - 1, cellSize - 1
          );
        }
      }

      // MAC text panel (slow phase only)
      if (p < 0.45 && textW > 80) {
        const showCount = Math.min(KH * KW, 4);
        ctx.font = `400 11px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(232,232,240,0.35)';
        ctx.fillText('multiply-accumulate:', textX, gridY + 8);

        let runSum = 0;
        for (let j = 0; j < showCount; j++) {
          const jy = Math.floor(j / KW), jx = j % KW;
          const inV = patchNorm[(ky + jy) * patchSize + (kx + jx)] || 0;
          const kV = kernelNorm[j] || 0;
          const inRaw = this.patch[(ky + jy) * patchSize + (kx + jx)] || 0;
          const kRaw = kernel[j] || 0;
          runSum += inRaw * kRaw;

          const lineY = gridY + 30 + j * 26;
          const entryP = Math.min(1, phaseP * numSlow * 3 - j * 0.5);
          if (entryP <= 0) continue;

          // Input value
          ctx.fillStyle = viridisColorAlpha(inV, entryP * 0.8);
          ctx.fillText(inRaw.toFixed(3), textX, lineY);

          // × sign
          ctx.fillStyle = `rgba(232,232,240,${entryP * 0.4})`;
          ctx.fillText(' × ', textX + 52, lineY);

          // Kernel value
          const kColor = kernelNorm[j] > 0.5 ? `rgba(96,165,250,${entryP * 0.8})` : `rgba(248,113,113,${entryP * 0.8})`;
          ctx.fillStyle = kColor;
          ctx.fillText((kRaw).toFixed(3), textX + 78, lineY);

          // = product
          ctx.fillStyle = `rgba(232,232,240,${entryP * 0.4})`;
          ctx.fillText(' = ', textX + 132, lineY);
          ctx.fillStyle = `rgba(52,211,153,${entryP * 0.7})`;
          ctx.fillText((inRaw * kRaw).toFixed(3), textX + 154, lineY);
        }

        if (showCount < KH * KW) {
          ctx.fillStyle = 'rgba(232,232,240,0.2)';
          ctx.fillText('...', textX, gridY + 30 + showCount * 26);
        }

        // Running sum
        const sumY = gridY + 30 + (showCount + 1) * 26 + 8;
        ctx.fillStyle = 'rgba(251,211,77,0.6)';
        ctx.font = `500 12px 'JetBrains Mono', monospace`;
        ctx.fillText(`sum = ${runSum.toFixed(3)}`, textX, sumY);

        // Position counter
        ctx.fillStyle = 'rgba(232,232,240,0.2)';
        ctx.font = `400 10px 'Syne', sans-serif`;
        ctx.fillText(`position ${posIdx + 1} / ${positions.length}`, textX, gridY + gridH - 10);
      } else if (p >= 0.45) {
        // Fast phase: just show position counter
        ctx.fillStyle = 'rgba(232,232,240,0.3)';
        ctx.font = `400 11px 'Syne', sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(`scanning... ${posIdx + 1} / ${positions.length}`, textX + textW/2, H/2);
      }
    }

    // ── Phase 3: output heatmap ──
    if (p >= 0.65) {
      const fadeIn = easeOutCubic((p - 0.65) / 0.35);
      const { outNorm, outH, outW } = this;
      const showSize = Math.min(outH, outW, patchSize);

      // Draw output heatmap alongside or replacing input
      const outCellSize = cellSize;
      const outGridX = p > 0.80 ? gridX : gridX + gridW + 20;
      const outGridY = gridY;

      if (p > 0.80) {
        // Cross-fade: dim the input
        ctx.fillStyle = `rgba(6,6,16,${(p - 0.80) / 0.20 * 0.7})`;
        ctx.fillRect(gridX, gridY, gridW, gridH);
      }

      // Output mini-heatmap
      const outPatchSize = Math.min(showSize, patchSize);
      for (let gy = 0; gy < outPatchSize; gy++) {
        for (let gx = 0; gx < outPatchSize; gx++) {
          const sy = Math.floor(gy / outPatchSize * showSize);
          const sx = Math.floor(gx / outPatchSize * showSize);
          const v = outNorm[sy * outW + sx] || 0;
          const viridis = INF_VIRIDIS[Math.round(v * 255)];
          ctx.fillStyle = `rgba(${viridis[0]},${viridis[1]},${viridis[2]},${fadeIn * 0.9})`;
          ctx.fillRect(
            outGridX + gx * outCellSize,
            outGridY + gy * outCellSize,
            outCellSize - 1, outCellSize - 1
          );
        }
      }

      if (p > 0.75) {
        ctx.fillStyle = `rgba(232,232,240,${fadeIn * 0.25})`;
        ctx.font = `400 9px 'Syne', sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(
          `output  [${this.H_out}×${this.W_out}, C=${this.C_out}]`,
          outGridX + (outPatchSize * outCellSize) / 2,
          outGridY + outPatchSize * outCellSize + 14
        );
      }
    }

    // Title at top
    ctx.fillStyle = 'rgba(232,232,240,0.2)';
    ctx.font = `400 10px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(
      `kernel ${this.KH}×${this.KW}  stride ${this.SH}  groups ${this.C_in === this.C_out && this.C_in > 1 ? this.C_in : 1}`,
      W / 2, H - 16
    );

    // Sampling note
    const totalVals = this.C_in * this.H_in * this.W_in;
    const shownVals = this.patchSize * this.patchSize;
    drawSamplingNote(ctx, W, H,
      `showing ${fmtNum(shownVals)} of ${fmtNum(totalVals)} values · channel avg · center crop`
    );
  }
}

// ── SoftmaxRenderer ───────────────────────────────────
// Logits morph into probabilities with bar chart reveal

class SoftmaxRenderer extends BaseRenderer {
  constructor(canvas, result, labels, speedGetter) {
    super(canvas, speedGetter);
    this.top5 = result.top5;
    this.probs = result.probs.data;
    this.labels = labels || [];
  }

  get baseDuration() { return 3200; }

  draw(p) {
    const { ctx, W, H, top5, labels } = this;
    clearCanvas(ctx, W, H);

    const cx = W / 2;

    // Title
    ctx.font = `500 12px 'Syne', sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = `rgba(232,232,240,${Math.min(1, p * 3) * 0.3})`;
    ctx.fillText('softmax — converting logits to probabilities', cx, 28);

    const N = Math.min(top5.length, 5);
    const barH = 28, gap = 12;
    const totalH = N * (barH + gap) - gap;
    const startY = (H - totalH) / 2;
    const labelW = 180;
    const barAreaX = cx - 160 + labelW + 16;
    const barMaxW = W - barAreaX - 80;

    for (let i = 0; i < N; i++) {
      const item = top5[i];
      const prob = item.probability;
      const label = labels[item.index] || `class ${item.index}`;
      const shortLabel = label.split(',')[0].trim();
      const y = startY + i * (barH + gap);

      // Stagger entry
      const entryDelay = i * 0.06;
      const entryP = Math.max(0, Math.min(1, (p - entryDelay) / 0.35));
      const barP = Math.max(0, Math.min(1, (p - 0.25 - entryDelay) / 0.5));

      if (entryP <= 0) continue;

      // Opacity and position slide-in
      const slideT = easeOutCubic(entryP);
      const alpha = slideT;
      const offsetX = (1 - slideT) * 20;

      // Label
      const isTop = i === 0;
      ctx.fillStyle = isTop
        ? `rgba(251,211,77,${alpha})`
        : `rgba(232,232,240,${alpha * 0.7})`;
      ctx.font = isTop
        ? `600 13px 'Syne', sans-serif`
        : `400 12px 'Syne', sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillText(shortLabel, cx - 160 + labelW + offsetX, y + barH / 2);

      // Bar track
      ctx.fillStyle = `rgba(255,255,255,${alpha * 0.06})`;
      roundRect(ctx, barAreaX + offsetX, y, barMaxW, barH, 4); ctx.fill();

      // Bar fill
      const fillW = barP * prob * barMaxW;
      if (fillW > 0) {
        const barColor = isTop ? '#FBD34D' : 'rgba(96,165,250,0.6)';
        ctx.fillStyle = barColor;
        roundRect(ctx, barAreaX + offsetX, y, fillW, barH, 4); ctx.fill();
      }

      // Percentage
      const pctAlpha = Math.max(0, Math.min(1, (barP - 0.2) / 0.3)) * alpha;
      if (pctAlpha > 0) {
        ctx.fillStyle = isTop
          ? `rgba(251,211,77,${pctAlpha})`
          : `rgba(232,232,240,${pctAlpha * 0.6})`;
        ctx.font = `400 12px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'left';
        ctx.fillText(`${(prob * 100).toFixed(1)}%`, barAreaX + barMaxW + 12 + offsetX, y + barH / 2);
      }
    }

    // Sampling note
    drawSamplingNote(ctx, W, H, `showing top 5 of 1,000 classes`);
  }
}

// ── Utility: roundRect ────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ── Renderer registry ─────────────────────────────────

const RENDERER_MAP = {
  Conv:       ConvRenderer,
  Clip:       ClipRenderer,
  Add:        AddRenderer,
  ReduceMean: ReduceMeanRenderer,
  Reshape:    ReshapeRenderer,
  Gemm:       GemmRenderer,
};
