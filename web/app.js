/* ═══════════════════════════════════════════════════════
   TinyForge — app.js

   Loads graph.json produced by `make json`, then renders
   all 100 nodes flat in a horizontal strip. No grouping —
   the user can see the whole network laid out linearly.
   ═══════════════════════════════════════════════════════ */

// ── Op colors ─────────────────────────────────────────
const OP_COLOR = {
  Conv:       'var(--op-conv)',
  Clip:       'var(--op-clip)',
  Add:        'var(--op-add)',
  Gemm:       'var(--op-gemm)',
  ReduceMean: 'var(--op-reduce)',
  Reshape:    'var(--op-reshape)',
};

function opColor(op) {
  return OP_COLOR[op] || 'var(--op-other)';
}

// Background tints for node inner section
const OP_BG = {
  Conv:       'var(--op-conv-bg)',
  Clip:       'var(--op-clip-bg)',
  Add:        'var(--op-add-bg)',
  Gemm:       'var(--op-gemm-bg)',
  ReduceMean: 'var(--op-reduce-bg)',
  Reshape:    'var(--op-reshape-bg)',
};

// ── Human-readable op descriptions ────────────────────
// People who don't know ML need plain-English names.
const OP_LABELS = {
  Conv:       'Convolution',
  Clip:       'Activation (ReLU6)',
  Add:        'Residual Add',
  Gemm:       'Matrix Multiply',
  ReduceMean: 'Average Pool',
  Reshape:    'Reshape',
  BatchNormalization: 'Batch Norm',
  Shape:      'Get Shape',
  Gather:     'Gather',
  Unsqueeze:  'Add Dimension',
  Concat:     'Concatenate',
  Mul:        'Multiply',
  Pad:        'Pad',
};

// Short plain-English tooltip for each op type
const OP_TOOLTIP = {
  Conv:       'Scans the image with small filters to detect patterns like edges and textures',
  Clip:       'Caps negative values to zero — decides which signals pass through',
  Add:        'Adds a shortcut connection so information can skip layers (prevents forgetting)',
  Gemm:       'Multiplies features by a weight matrix to produce the final class scores',
  ReduceMean: 'Averages the spatial dimensions down to a single value per channel',
  Reshape:    'Changes the shape of the data without changing the values',
};

// ── Legend colors for CSS ─────────────────────────────
const LEGEND_ITEMS = [
  { op: 'Conv',       label: 'Convolution',    desc: 'Pattern detection' },
  { op: 'Clip',       label: 'Activation',     desc: 'Signal gating' },
  { op: 'Add',        label: 'Residual',       desc: 'Skip connection' },
  { op: 'Gemm',       label: 'Classifier',     desc: 'Final prediction' },
  { op: 'ReduceMean', label: 'Pooling',        desc: 'Spatial averaging' },
  { op: 'Reshape',    label: 'Reshape',        desc: 'Data reformatting' },
];

// ── Blog markdown config per phase ────────────────────
const BLOG_PHASES = {
  parser: {
    eyebrow: 'Phase 01 — Parser',
    title: 'How we turn a model file into a graph',
    templateId: 'blog-md-parser',
  },
  inference: {
    eyebrow: 'Phase 02 — Inference',
    title: 'How the graph becomes a prediction',
    templateId: 'blog-md-inference',
  },
};

// ── Annotations ───────────────────────────────────────
const ANNOTATIONS = {
  0:  "Stem — first 3×3 conv with stride 2. Image shrinks: 224×224 → 112×112. Only 3 input channels (RGB).",
  2:  "First inverted residual block. 'Inverted' because it expands channels first, then compresses — opposite of regular ResNets.",
  5:  "Expansion factor t=6 kicks in here. Channels temporarily blow up to ×6 before being squeezed back down.",
  15: "↩ First skip connection! The Add node combines the block input with its own output. Lets gradients flow straight back during training.",
  44: "We're at 14×14 spatial resolution now. The network is getting deeper but the feature maps are smaller and richer.",
  61: "Block 12 — channels are 96 here. Notice the depthwise conv groups = 96, one filter per channel.",
  95: "Final feature conv — projects everything to 1280 dimensions. This is the 'embedding' of the image.",
  97: "Global average pool — collapses the 7×7 spatial map into a single number per channel. Shape: [1,1280,7,7] → [1,1280,1,1].",
  98: "Flatten — reshape [1,1280,1,1] into [1,1280] so the linear layer can take it.",
  99: "Classifier — one big matrix multiply. 1280 features → 1000 ImageNet class scores. That's the final prediction.",
};

// ── State ─────────────────────────────────────────────
const state = {
  tensors:      {},
  nodes:        [],
  selectedNode: null,
  selectedIdx:  -1,
  screen:       'parser',
  graphData:    null,      // raw graph JSON for engine
  weights:      null,      // loaded weight tensors
  labels:       null,      // ImageNet labels
};

// ── Inference stages (computed from tensor shapes) ─────
function stageName(res, idx) {
  if (res === 'flat') return 'Head';
  if (idx === 0) return 'Stem';
  return `Stage ${idx}`;
}

function buildStages(nodes, tensors) {
  const stages = [];
  let current = null;
  nodes.forEach((node, i) => {
    const t = tensors[node.outputs[0]];
    const res = (t && t.shape.length === 4)
      ? `${t.shape[2]}x${t.shape[3]}`
      : 'flat';
    if (!current || current.res !== res) {
      current = { res, label: stageName(res, stages.length), items: [] };
      stages.push(current);
    }
    current.items.push({ node, idx: i });
  });
  return stages;
}

const blogState = {
  open:        true,
  phase:       'parser',
  initialized: false,
};

// ── Init ──────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch('../assets/graph.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const graph = await res.json();
    state.tensors   = graph.tensors;
    state.nodes     = graph.nodes;
    state.graphData = graph;

    renderLegend();
    updateStats(graph.nodes.length, Object.keys(graph.tensors).length);
    renderGraph(graph.nodes);
    setupNav();
    setupPanel();
    setupScrollProgress();
    setupKeyboardNav();
    setupBlog();
    setupInference(graph);
  } catch (err) {
    document.getElementById('graph-track').innerHTML =
      `<p style="padding:40px;color:var(--text-3);font-size:11px;line-height:1.8">
        Couldn't load <code>../assets/graph.json</code>.<br>
        Run <code>make json</code> first, then <code>make serve</code>.
      </p>`;
    console.error('Failed to load graph.json:', err);
  }
}

// ── Legend ─────────────────────────────────────────────
function renderLegend() {
  const container = document.getElementById('legend-items');
  LEGEND_ITEMS.forEach(item => {
    const el = div('legend-item');
    const dot = div('legend-dot');
    dot.style.setProperty('--c', OP_COLOR[item.op]);
    const text = document.createElement('span');
    text.className = 'legend-text';
    text.textContent = item.label;
    const desc = document.createElement('span');
    desc.className = 'legend-desc';
    desc.textContent = item.desc;
    el.append(dot, text, desc);
    container.appendChild(el);
  });
}

// ── Render: nodes flat horizontal ─────────────────────
function renderGraph(nodes) {
  const track = document.getElementById('graph-track');
  track.innerHTML = '';

  nodes.forEach((node, i) => {
    const wrapper = div('node-wrapper');
    wrapper.appendChild(makeNodeCard(node, i));
    track.appendChild(wrapper);

    if (i < nodes.length - 1) {
      const nextIsAdd = nodes[i + 1].op === 'Add';
      track.appendChild(makeConnector(nextIsAdd));
    }
  });

  // Stagger entry animation
  const wrappers = track.querySelectorAll('.node-wrapper');
  const conns    = track.querySelectorAll('.conn');
  wrappers.forEach((el, i) => setTimeout(() => el.classList.add('in'), 30 + i * 20));
  conns.forEach((el, i)    => setTimeout(() => el.classList.add('in'), 60 + i * 20));

  // Dismiss scroll hint after first scroll
  const hint = document.getElementById('scroll-hint');
  if (hint) {
    track.addEventListener('scroll', () => hint.classList.add('hidden'), { once: true });
  }
}

// ── Annotation zone ───────────────────────────────────
function makeAnnotationZone(nodeId) {
  const zone = div('ann-zone');

  if (ANNOTATIONS[nodeId] != null) {
    const bubble = div('ann-bubble');
    bubble.textContent = ANNOTATIONS[nodeId];
    zone.appendChild(bubble);
  }

  return zone;
}

// ── Node card ─────────────────────────────────────────
function makeNodeCard(node, idx) {
  const card = div('node-card');
  card.dataset.id = node.id;
  card.dataset.idx = idx;
  card.setAttribute('tabindex', '0');
  card.setAttribute('role', 'button');
  card.setAttribute('aria-label', `${OP_LABELS[node.op] || node.op} node #${node.id}`);

  const color = opColor(node.op);

  // Colored top stripe
  const stripe = div('node-stripe');
  stripe.style.setProperty('--c', color);

  const inner = div('node-inner');
  inner.style.setProperty('--c', OP_BG[node.op] || 'var(--op-other-bg)');

  // Op name
  const opLabel = div('node-op');
  opLabel.style.setProperty('--c', color);
  opLabel.textContent = node.op;

  // Human-readable subtitle
  const humanLabel = div('node-human');
  humanLabel.textContent = OP_LABELS[node.op] || node.op;

  // Output shape
  const outLabel = div('node-out');
  const outTensor = state.tensors[node.outputs[0]];
  if (outTensor && outTensor.shape.length > 0) {
    outLabel.textContent = outTensor.shape.join('×');
  } else {
    outLabel.textContent = node.outputs[0] || '—';
  }

  // Node ID
  const idLabel = div('node-id');
  idLabel.textContent = `#${node.id}`;

  inner.append(opLabel, humanLabel, outLabel);
  card.append(stripe, inner, idLabel);

  // Annotation marker — small dot to indicate this node has a note
  if (ANNOTATIONS[node.id] != null) {
    const marker = div('node-ann-marker');
    marker.title = 'Has annotation — click to read';
    card.appendChild(marker);
  }

  card.addEventListener('click', () => onNodeClick(node, idx));
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onNodeClick(node, idx);
    }
  });

  return card;
}

// ── Connector arrow ───────────────────────────────────
function makeConnector(isAdd) {
  const c = div('conn');
  if (isAdd) c.classList.add('is-add');
  c.append(div('conn-line'), div('conn-arrow'));
  return c;
}

// ── Node click: show detail panel ─────────────────────
function onNodeClick(node, idx) {
  document.querySelectorAll('.node-card.sel').forEach(c => c.classList.remove('sel'));

  const card = document.querySelector(`.node-card[data-id="${node.id}"]`);
  if (card) {
    card.classList.add('sel');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
  }

  state.selectedNode = node;
  state.selectedIdx = idx;
  openPanel(node);
}

// ── Detail panel ──────────────────────────────────────
function openPanel(node) {
  const panel = document.getElementById('panel');
  const body  = document.getElementById('panel-body');
  const color = opColor(node.op);
  const barHint = document.getElementById('panel-bar-hint');

  if (barHint) barHint.textContent = `${OP_LABELS[node.op] || node.op} #${node.id}  ·  click to close`;
  panel.style.setProperty('--panel-accent', color);

  // Build input rows
  const inputRows = node.inputs.map(name => {
    const t = state.tensors[name];
    const shape = t ? '[' + t.shape.join(', ') + ']' : '—';
    const weightDot = (t && t.has_data) ? '<span class="has-data" title="Weight data loaded in memory"></span>' : '';
    return `<div class="panel-row"><b>${esc(trunc(name, 24))}</b> ${esc(shape)}${weightDot}</div>`;
  }).join('');

  // Output rows
  const outputRows = node.outputs.map(name => {
    const t = state.tensors[name];
    const shape = t ? '[' + t.shape.join(', ') + ']' : '—';
    return `<div class="panel-row"><b>${esc(trunc(name, 24))}</b> ${esc(shape)}</div>`;
  }).join('');

  // Attributes
  const attrRows = Object.keys(node.attrs).length > 0
    ? Object.entries(node.attrs).map(([k, v]) =>
        `<div class="panel-row"><b>${esc(k)}</b> [${esc(v.join(', '))}]</div>`
      ).join('')
    : `<div class="panel-row" style="color:var(--text-3)">none</div>`;

  // Parameter count
  const wInput = node.inputs.find(i => state.tensors[i]?.has_data);
  let paramSection = '';
  if (wInput) {
    const shape = state.tensors[wInput].shape;
    if (shape.length > 0) {
      const count = shape.reduce((a, b) => a * b, 1);
      const kb    = (count * 4 / 1024).toFixed(1);
      paramSection = `
        <div class="panel-section">
          <div class="panel-section-label">Parameters</div>
          <div class="panel-row"><b>${count.toLocaleString()}</b> weights</div>
          <div class="panel-row"><b>${kb}</b> KB on disk</div>
        </div>`;
    }
  }

  // Annotation section — show the human note if this node has one
  let annSection = '';
  if (ANNOTATIONS[node.id] != null) {
    annSection = `
      <div class="panel-section panel-annotation">
        <div class="panel-section-label">Note</div>
        <div class="panel-ann-text">${esc(ANNOTATIONS[node.id])}</div>
      </div>`;
  }

  // Plain-English tooltip
  let tooltipSection = '';
  if (OP_TOOLTIP[node.op]) {
    tooltipSection = `
      <div class="panel-section panel-tooltip-section">
        <div class="panel-section-label">What does this do?</div>
        <div class="panel-tooltip-text">${esc(OP_TOOLTIP[node.op])}</div>
      </div>`;
  }

  body.innerHTML = `
    <div class="panel-head">
      <div class="panel-head-left">
        <div class="panel-op" style="color:${color}">${esc(node.op)}</div>
        <div class="panel-human">${esc(OP_LABELS[node.op] || node.op)}</div>
      </div>
      <div class="panel-head-right">
        <div class="panel-id">node #${esc(String(node.id))}</div>
        <button class="panel-close" id="panel-close-btn" aria-label="Close panel">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>
    ${tooltipSection}
    ${annSection}
    <div class="panel-grid">
      <div class="panel-section">
        <div class="panel-section-label">Inputs</div>
        ${inputRows}
      </div>
      <div class="panel-section">
        <div class="panel-section-label">Outputs</div>
        ${outputRows}
      </div>
      <div class="panel-section">
        <div class="panel-section-label">Attributes</div>
        ${attrRows}
      </div>
      ${paramSection}
    </div>`;

  // Wire up close button
  const closeBtn = document.getElementById('panel-close-btn');
  if (closeBtn) closeBtn.addEventListener('click', closePanel);

  // Stagger section animations
  body.querySelectorAll('.panel-head, .panel-tooltip-section, .panel-annotation, .panel-grid, .panel-grid > .panel-section').forEach((el, i) => {
    el.style.animationDelay = `${0.04 + i * 0.05}s`;
  });

  panel.classList.add('open');
}

function closePanel() {
  document.getElementById('panel').classList.remove('open');
  document.querySelectorAll('.node-card.sel').forEach(c => c.classList.remove('sel'));
  state.selectedNode = null;
  state.selectedIdx = -1;
  const barHint = document.getElementById('panel-bar-hint');
  if (barHint) barHint.textContent = 'Click a node to inspect';
}

function setupPanel() {
  document.getElementById('panel-bar').addEventListener('click', closePanel);
}

// ── Scroll progress bar ──────────────────────────────
function setupScrollProgress() {
  const track = document.getElementById('graph-track');
  const bar   = document.getElementById('scroll-bar');
  if (!track || !bar) return;

  track.addEventListener('scroll', () => {
    const max = track.scrollWidth - track.clientWidth;
    if (max > 0) {
      bar.style.width = ((track.scrollLeft / max) * 100) + '%';
    }
  });
}

// ── Keyboard navigation ──────────────────────────────
function setupKeyboardNav() {
  document.addEventListener('keydown', (e) => {
    if (state.screen !== 'parser') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      const next = state.selectedIdx + dir;
      if (next >= 0 && next < state.nodes.length) {
        onNodeClick(state.nodes[next], next);
      }
    }

    if (e.key === 'Escape') {
      closePanel();
    }
  });
}

// ── Screen navigation ─────────────────────────────────
function setupNav() {
  showScreen('parser');

  document.querySelectorAll('.phase-btn').forEach(btn => {
    btn.addEventListener('click', () => showScreen(btn.dataset.target));
  });
}

function showScreen(name) {
  const vp = document.getElementById('viewport');
  vp.classList.toggle('show-inference', name === 'inference');

  document.querySelectorAll('.phase-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.target === name);
  });

  state.screen = name;
  closePanel();

  if (name === 'parser' || name === 'inference') {
    blogState.phase = name;
    if (blogState.initialized) {
      loadBlogForPhase(name);
    }
  }
}

// ── Blog markdown loader ───────────────────────────────
function renderMarkdownToHtml(md) {
  const lines = md.split('\n');
  const parts = [];
  let i = 0;

  function inline(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code class="blog-md-ic">$1</code>');
  }

  while (i < lines.length) {
    const line = lines[i].replace(/\r$/, '');
    const trimmed = line.trim();

    if (!trimmed) { i++; continue; }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(trimmed)) {
      parts.push('<hr class="blog-md-divider">');
      i++; continue;
    }

    // Fenced code block
    if (trimmed.startsWith('```')) {
      i++;
      const codeLines = [];
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++;
      const code = codeLines.join('\n').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      parts.push(`<pre class="blog-md-pre"><code class="blog-md-code">${code}</code></pre>`);
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      const qLines = [];
      while (i < lines.length && lines[i].trim().startsWith('> ')) {
        qLines.push(lines[i].trim().slice(2));
        i++;
      }
      parts.push(`<blockquote class="blog-md-blockquote"><p>${inline(qLines.join(' '))}</p></blockquote>`);
      continue;
    }

    // Table
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      const tLines = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        tLines.push(lines[i].trim());
        i++;
      }
      if (tLines.length >= 2) {
        const headers = tLines[0].slice(1,-1).split('|').map(c => c.trim());
        const rows = tLines.slice(2);
        let t = '<table class="blog-md-table"><thead><tr>';
        for (const h of headers) t += `<th>${inline(h)}</th>`;
        t += '</tr></thead><tbody>';
        for (const row of rows) {
          const cells = row.slice(1,-1).split('|').map(c => c.trim());
          t += '<tr>';
          for (const c of cells) t += `<td>${inline(c)}</td>`;
          t += '</tr>';
        }
        t += '</tbody></table>';
        parts.push(t);
      }
      continue;
    }

    // Unordered list
    if (trimmed.startsWith('- ')) {
      let html = '<ul class="blog-md-ul">';
      while (i < lines.length && lines[i].trim().startsWith('- ')) {
        html += `<li class="blog-md-li">${inline(lines[i].trim().slice(2))}</li>`;
        i++;
      }
      html += '</ul>';
      parts.push(html);
      continue;
    }

    // Ordered list
    if (/^\d+\.\s/.test(trimmed)) {
      let html = '<ol class="blog-md-ol">';
      while (i < lines.length && /^\d+\.\s/.test(lines[i].trim())) {
        html += `<li class="blog-md-li">${inline(lines[i].trim().replace(/^\d+\.\s/, ''))}</li>`;
        i++;
      }
      html += '</ol>';
      parts.push(html);
      continue;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      parts.push(`<h3 class="blog-md-h3">${inline(trimmed.slice(4))}</h3>`);
      i++; continue;
    }
    if (trimmed.startsWith('## ')) {
      parts.push(`<h2 class="blog-md-h2">${inline(trimmed.slice(3))}</h2>`);
      i++; continue;
    }
    if (trimmed.startsWith('# ')) {
      parts.push(`<h1 class="blog-md-h1">${inline(trimmed.slice(2))}</h1>`);
      i++; continue;
    }

    // Standalone image
    if (trimmed.startsWith('![') && trimmed.includes('](') && trimmed.endsWith(')')) {
      const match = trimmed.match(/^!\[(.*)]\((.*)\)$/);
      if (match) {
        const alt = match[1] || '';
        const src = match[2] || '';
        parts.push(
          `<figure class="blog-md-figure">` +
            `<div class="blog-md-image-wrap">` +
              `<img class="blog-md-image" src="${esc(src)}" alt="${esc(alt)}">` +
            `</div>` +
            (alt ? `<figcaption class="blog-md-caption">${esc(alt)}</figcaption>` : '') +
          `</figure>`
        );
        i++; continue;
      }
    }

    // Paragraph — collect until blank line or block-level element
    const paraLines = [];
    while (i < lines.length) {
      const t = lines[i].trim();
      if (!t) break;
      if (t.startsWith('#') || t.startsWith('> ') || t.startsWith('- ') ||
          t.startsWith('|') || t.startsWith('```') || /^[-*_]{3,}$/.test(t) ||
          /^\d+\.\s/.test(t)) break;
      paraLines.push(t);
      i++;
    }
    if (paraLines.length) {
      parts.push(`<p class="blog-md-paragraph">${inline(paraLines.join(' '))}</p>`);
    }
  }
  return parts.join('');
}

function loadBlogForPhase(phase) {
  const cfg = BLOG_PHASES[phase];
  if (!cfg) return;

  const container = document.getElementById('blog-content');
  const titleEl   = document.getElementById('blog-title');
  const eyebrowEl = document.getElementById('blog-eyebrow');
  if (!container || !titleEl || !eyebrowEl) return;

  eyebrowEl.textContent = cfg.eyebrow;
  titleEl.textContent   = cfg.title;

  const tmpl = document.getElementById(cfg.templateId);
  if (tmpl) {
    container.innerHTML = renderMarkdownToHtml(tmpl.textContent);
  } else {
    container.innerHTML = '<p class="blog-error">Content not found.</p>';
  }
}

// ── Blog text panel ─────────────────────────────────────
function setupBlog() {
  const toggle = document.getElementById('blog-toggle');
  const panel  = document.getElementById('blog-panel');
  const close  = document.getElementById('blog-close-btn');
  if (!toggle || !panel || !close) return;

  function openBlog() {
    blogState.open = true;
    panel.classList.add('open');
    toggle.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
    panel.setAttribute('aria-hidden', 'false');
  }

  function closeBlog() {
    blogState.open = false;
    panel.classList.remove('open');
    toggle.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
  }

  toggle.addEventListener('click', () => {
    if (blogState.open) closeBlog();
    else openBlog();
  });
  if (close) close.addEventListener('click', closeBlog);

  document.addEventListener('keydown', (e) => {
    if (!blogState.open) return;
    if (e.key === 'Escape') {
      closeBlog();
    }
  });

  blogState.initialized = true;
  blogState.phase = state.screen;
  loadBlogForPhase(state.screen);

  if (window.innerWidth <= 600) {
    closeBlog();
  } else {
    openBlog();
  }
}

// ── Stats bar ─────────────────────────────────────────
function updateStats(nodeCount, tensorCount) {
  document.getElementById('stat-nodes').textContent   = `${nodeCount} nodes`;
  document.getElementById('stat-tensors').textContent = `${tensorCount} tensors`;

  let totalParams = 0;
  for (const t of Object.values(state.tensors)) {
    if (t.has_data && t.shape.length > 0) {
      totalParams += t.shape.reduce((a, b) => a * b, 1);
    }
  }
  const mb = (totalParams * 4 / 1024 / 1024).toFixed(1);
  document.getElementById('stat-weights').textContent = `${mb} MB weights`;
}

// ── Helpers ───────────────────────────────────────────
function div(cls) {
  const el = document.createElement('div');
  if (cls) el.className = cls;
  return el;
}

function trunc(s, n) {
  return s.length > n ? '…' + s.slice(-(n - 1)) : s;
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Theme toggle ──────────────────────────────────────
function setupTheme() {
  const html  = document.documentElement;
  const btn   = document.getElementById('theme-btn');

  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const saved = localStorage.getItem('tf-theme');
  let current = saved || (systemDark ? 'dark' : 'light');

  function apply(theme) {
    current = theme;
    html.setAttribute('data-theme', theme);
    localStorage.setItem('tf-theme', theme);
    btn.classList.toggle('is-dark', theme === 'dark');
  }

  apply(current);

  btn.addEventListener('click', () => {
    apply(current === 'dark' ? 'light' : 'dark');
  });
}

// ── Inference UI — handled by inference-ui.js ─────────

// ── Go ────────────────────────────────────────────────
setupTheme();
init();
