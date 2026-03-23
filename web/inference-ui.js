/* ═══════════════════════════════════════════════════════
   TinyForge — inference-ui.js

   3-state inference experience:
     HERO → WALKTHROUGH → RESULTS → HERO

   Owns the state machine, manual node stepping,
   playback controls, and wires renderers to real data.
   ═══════════════════════════════════════════════════════ */

// ── Op accent colors (hex) ────────────────────────────

const OP_ACCENT = {
  Conv:       '#60A5FA',
  Clip:       '#34D399',
  Add:        '#F87171',
  Gemm:       '#C084FC',
  ReduceMean: '#FBD34D',
  Reshape:    '#94A3B8',
};

const OP_DESCRIPTIONS = {
  Conv:       'Convolution',
  Clip:       'Activation (ReLU6)',
  Add:        'Residual Add',
  ReduceMean: 'Average Pool',
  Reshape:    'Reshape',
  Gemm:       'Matrix Multiply',
};

// ── InferenceController ───────────────────────────────
// Replicates the engine.js graph-walking loop, but
// one node at a time for animated playback.

class InferenceController {
  constructor(graphData, weights, inputTensor) {
    this.nodes = graphData.nodes;
    this.activations = {};

    // Seed with input tensor
    this.activations['input'] = inputTensor;

    // Seed with all weights
    for (const [name, tensor] of Object.entries(weights)) {
      this.activations[name] = tensor;
    }

    this.currentIndex = 0;
  }

  get done() {
    return this.currentIndex >= this.nodes.length;
  }

  stepOne() {
    if (this.done) return null;

    const node = this.nodes[this.currentIndex];
    const nodeStart = performance.now();

    // Collect inputs
    const inputs = node.inputs.map(name => this.activations[name] || null);

    let result;
    switch (node.op) {
      case 'Conv': {
        const input  = inputs[0];
        const weight = inputs[1];
        const bias   = inputs[2] || null;
        result = opConv(input, weight, bias, node.attrs);
        break;
      }
      case 'Clip': {
        const input  = inputs[0];
        const minT   = inputs[1];
        const maxT   = inputs[2];
        const minVal = minT ? minT.data[0] : 0;
        const maxVal = maxT ? maxT.data[0] : 6;
        result = opClip(input, minVal, maxVal);
        break;
      }
      case 'Add': {
        result = opAdd(inputs[0], inputs[1]);
        break;
      }
      case 'ReduceMean': {
        const keepdims = node.attrs.keepdims ? node.attrs.keepdims[0] : 1;
        result = opReduceMean(inputs[0], keepdims);
        break;
      }
      case 'Reshape': {
        const shapeT = inputs[1];
        const target = Array.from(shapeT.data).map(v => Math.round(v));
        result = opReshape(inputs[0], target);
        break;
      }
      case 'Gemm': {
        result = opGemm(inputs[0], inputs[1], inputs[2] || null, node.attrs);
        break;
      }
      default:
        throw new Error(`Unknown op: ${node.op}`);
    }

    this.activations[node.outputs[0]] = result;
    const nodeMs = performance.now() - nodeStart;

    const idx = this.currentIndex;
    this.currentIndex++;
    return { node, result, inputs, index: idx, nodeMs };
  }

  finalize() {
    const logits = this.activations['output'];
    const probs  = opSoftmax(logits);
    return {
      logits,
      probs,
      top5: topK(probs, 5),
    };
  }
}

// ── UI state ──────────────────────────────────────────

const infState = {
  phase:      'hero',      // 'hero' | 'walkthrough' | 'results'
  image:      null,        // Image element from upload
  controller: null,        // InferenceController
  playing:    true,
  speed:      1.0,
  currentRenderer: null,
  loopRunning: false,
  labels:     null,        // ImageNet labels
  graphData:  null,        // Cached graph data from app.js state
  weights:    null,        // Cached weights
  _stepOnce:  false,       // True when step-forward was pressed
};

// Pause signal
let _resumeSignal = null;

function waitWhilePaused() {
  if (infState.playing) return Promise.resolve();
  return new Promise(resolve => { _resumeSignal = resolve; });
}

function resume() {
  if (_resumeSignal) {
    const r = _resumeSignal;
    _resumeSignal = null;
    r();
  }
}

// ── DOM refs ──────────────────────────────────────────

let elHero, elWalkthrough, elResults;
let elDropZone, elFileInput, elPreview, elDropPrompt, elRunBtn;
let elProgressFill, elTint, elNodeCounter, elOpName, elOpDesc, elShapeInfo, elAnnotation;
let elCanvas, elControls, elPlayPause, elStep, elSpeedSlider, elSpeedVal;
let elBarChart, elTryAgain, elResultsTime;

// ── Init ──────────────────────────────────────────────

function setupInference(graphData) {
  infState.graphData = graphData;

  // Cache DOM refs
  elHero         = document.getElementById('inf-hero');
  elWalkthrough  = document.getElementById('inf-walkthrough');
  elResults      = document.getElementById('inf-results-screen');
  elDropZone     = document.getElementById('inf-drop-zone');
  elFileInput    = document.getElementById('inf-file-input');
  elPreview      = document.getElementById('inf-preview');
  elDropPrompt   = document.getElementById('inf-drop-prompt');
  elRunBtn       = document.getElementById('inf-run-btn');
  elProgressFill = document.getElementById('inf-progress-fill');
  elTint         = document.getElementById('inf-tint');
  elNodeCounter  = document.getElementById('inf-node-counter');
  elOpName       = document.getElementById('inf-op-name');
  elOpDesc       = document.getElementById('inf-op-desc');
  elShapeInfo    = document.getElementById('inf-shape-info');
  elAnnotation   = document.getElementById('inf-annotation');
  elCanvas       = document.getElementById('inf-main-canvas');
  elPlayPause    = document.getElementById('inf-play-pause');
  elStep         = document.getElementById('inf-step');
  elSpeedSlider  = document.getElementById('inf-speed');
  elSpeedVal     = document.getElementById('inf-speed-val');
  elBarChart     = document.getElementById('inf-bar-chart');
  elTryAgain     = document.getElementById('inf-try-again');
  elResultsTime  = document.getElementById('inf-results-time');

  if (!elHero) return; // inference screen not present

  // Set canvas resolution
  elCanvas.width  = 760;
  elCanvas.height = 400;

  // Drop zone click
  elDropZone.addEventListener('click', () => elFileInput.click());
  elFileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleImageFile(e.target.files[0]);
  });

  // Drag and drop
  elDropZone.addEventListener('dragover', e => {
    e.preventDefault();
    elDropZone.classList.add('drag-over');
  });
  elDropZone.addEventListener('dragleave', () => {
    elDropZone.classList.remove('drag-over');
  });
  elDropZone.addEventListener('drop', e => {
    e.preventDefault();
    elDropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleImageFile(f);
  });

  // Run button
  elRunBtn.addEventListener('click', startInference);

  // Controls
  elPlayPause.addEventListener('click', togglePlayPause);
  elStep.addEventListener('click', stepOnce);
  elSpeedSlider.addEventListener('input', () => {
    infState.speed = parseFloat(elSpeedSlider.value);
    elSpeedVal.textContent = infState.speed + '×';
  });

  // Try Again
  elTryAgain.addEventListener('click', resetToHero);

  // Keyboard
  document.addEventListener('keydown', onKeydown);
}

// ── Image upload ──────────────────────────────────────

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      infState.image = img;
      elPreview.src = img.src;
      elPreview.hidden = false;
      elDropPrompt.classList.add('hidden');
      elDropZone.classList.add('has-image');
      elRunBtn.disabled = false;
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

// ── Start inference ───────────────────────────────────

async function startInference() {
  if (!infState.image || !infState.graphData) return;

  // Transition hero → walkthrough
  transitionTo('walkthrough');
  infState.playing = true;
  updatePlayPauseBtn();

  // Load ImageNet labels (lazy)
  if (!infState.labels) {
    try {
      const r = await fetch('../assets/imagenet_labels.txt');
      const txt = await r.text();
      infState.labels = txt.trim().split('\n');
    } catch (e) {
      infState.labels = [];
    }
  }

  // Init WASM
  await initWasm();

  // Load weights (lazy)
  if (!infState.weights) {
    updateChrome(null, 0, '—');
    setTint('#94A3B8');
    if (elNodeCounter) elNodeCounter.textContent = 'loading weights...';
    infState.weights = await loadWeights(infState.graphData);
  }

  // Preprocess image
  const inputTensor = preprocessImage(infState.image);

  // Create controller
  infState.controller = new InferenceController(infState.graphData, infState.weights, inputTensor);
  infState.loopRunning = true;

  const totalStart = performance.now();
  await playbackLoop();
  const totalMs = performance.now() - totalStart;

  // Finalize: softmax
  const result = infState.controller.finalize();
  result.totalMs = totalMs;

  // Run softmax animation
  if (elCanvas) {
    const softmaxRenderer = new SoftmaxRenderer(elCanvas, result, infState.labels, () => infState.speed);
    infState.currentRenderer = softmaxRenderer;
    updateChrome(null, 100, '—');
    setAccentColor('#FBD34D');
    if (elOpName) { elOpName.textContent = 'Softmax'; elOpName.style.color = '#FBD34D'; }
    if (elOpDesc) elOpDesc.textContent = 'Converting scores to probabilities';
    if (elShapeInfo) elShapeInfo.textContent = '[1, 1000]';
    if (elAnnotation) elAnnotation.classList.remove('visible');
    await softmaxRenderer.run();
    infState.currentRenderer = null;
  }

  infState.loopRunning = false;
  showResults(result);
}

// ── Playback loop ─────────────────────────────────────

async function playbackLoop() {
  const { controller } = infState;
  const total = controller.nodes.length;

  while (!controller.done) {
    await waitWhilePaused();

    const stepResult = controller.stepOne();
    if (!stepResult) break;

    const { node, result, inputs, index } = stepResult;

    // Update chrome (progress, op name, colors)
    updateChrome(node, index, formatShape(result.shape));

    // Pick renderer
    const RendererClass = RENDERER_MAP[node.op];
    if (RendererClass && elCanvas) {
      const renderer = new RendererClass(
        elCanvas,
        node,
        inputs,
        result,
        () => infState.speed
      );
      infState.currentRenderer = renderer;
      await renderer.run();
      infState.currentRenderer = null;
    }

    // After one step in step mode, re-pause
    if (infState._stepOnce) {
      infState._stepOnce = false;
      infState.playing = false;
      updatePlayPauseBtn();
    }

    // Yield to browser
    await new Promise(r => setTimeout(r, 0));
  }
}

// ── Chrome updates ─────────────────────────────────────

function updateChrome(node, index, shapeStr) {
  const total = infState.graphData ? infState.graphData.nodes.length : 100;
  const pct = (index / total) * 100;

  if (elProgressFill) {
    elProgressFill.style.width = pct + '%';
  }

  if (elNodeCounter) {
    elNodeCounter.textContent = node ? `${index + 1} / ${total}` : `${index} / ${total}`;
  }

  if (node) {
    const accent = OP_ACCENT[node.op] || '#94A3B8';
    setAccentColor(accent);

    if (elOpName) {
      elOpName.textContent = node.op;
      elOpName.style.color = accent;
    }
    if (elOpDesc) {
      elOpDesc.textContent = OP_DESCRIPTIONS[node.op] || node.op;
    }
    if (elShapeInfo) {
      elShapeInfo.textContent = shapeStr || '';
    }

    // Annotation
    const ANNOTATIONS = window.ANNOTATIONS || {};
    if (elAnnotation) {
      const ann = ANNOTATIONS[index];
      if (ann) {
        elAnnotation.textContent = ann;
        elAnnotation.classList.add('visible');
      } else {
        elAnnotation.classList.remove('visible');
      }
    }
  }
}

function setAccentColor(hex) {
  const el = document.getElementById('screen-inference');
  if (el) el.style.setProperty('--inf-accent', hex);
  if (elProgressFill) elProgressFill.style.background = hex;
  if (elTint) {
    elTint.style.background = `radial-gradient(ellipse 80% 60% at 50% 40%, ${hex}, transparent 70%)`;
  }
}

function setTint(hex) {
  if (elTint) {
    elTint.style.background = `radial-gradient(ellipse 80% 60% at 50% 40%, ${hex}, transparent 70%)`;
  }
}

function formatShape(shape) {
  if (!shape) return '';
  return '[' + shape.join(', ') + ']';
}

// ── Controls ──────────────────────────────────────────

function togglePlayPause() {
  infState.playing = !infState.playing;
  updatePlayPauseBtn();
  if (infState.playing) {
    // Unpause: resolve the waitWhilePaused promise
    infState._stepOnce = false;
    resume();
  }
}

function updatePlayPauseBtn() {
  if (!elPlayPause) return;
  const playing = infState.playing;
  elPlayPause.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
  elPlayPause.classList.toggle('playing', playing);
  elPlayPause.title = playing ? 'Pause (Space)' : 'Play (Space)';
}

async function stepOnce() {
  // Signal: pause again after exactly one node animation
  infState._stepOnce = true;
  infState.playing = true;
  updatePlayPauseBtn();
  resume(); // unblock waitWhilePaused

  // If mid-animation, skip to end immediately
  if (infState.currentRenderer) {
    infState.currentRenderer.cancel();
  }
}

// ── Results ───────────────────────────────────────────

function showResults(result) {
  const { top5, totalMs } = result;

  // Build bar chart
  if (elBarChart) {
    elBarChart.innerHTML = '';
    const max = top5[0].probability;

    top5.slice(0, 5).forEach((item, i) => {
      const label = (infState.labels && infState.labels[item.index])
        ? infState.labels[item.index].split(',')[0].trim()
        : `class ${item.index}`;
      const pct = (item.probability * 100).toFixed(1);
      const barW = (item.probability / max) * 100;

      const row = document.createElement('div');
      row.className = 'inf-bar-row';
      row.innerHTML = `
        <span class="inf-bar-label${i === 0 ? ' top-1' : ''}">${esc(label)}</span>
        <div class="inf-bar-track">
          <div class="inf-bar-fill${i === 0 ? ' top-1' : ''}" data-w="${barW}"></div>
        </div>
        <span class="inf-bar-pct${i === 0 ? ' top-1' : ''}">${pct}%</span>
      `;
      elBarChart.appendChild(row);
    });

    // Total time
    if (elResultsTime) {
      elResultsTime.textContent = `${(totalMs / 1000).toFixed(1)}s inference time`;
    }
  }

  // Transition walkthrough → results
  transitionTo('results');

  // Trigger bar animations after transition
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.inf-bar-fill').forEach(el => {
        el.style.width = el.dataset.w + '%';
      });
    });
  });
}

// ── Reset ─────────────────────────────────────────────

function resetToHero() {
  // Cancel any running animation
  if (infState.currentRenderer) {
    infState.currentRenderer.cancel();
    infState.currentRenderer = null;
  }
  infState.playing = true;
  infState.controller = null;
  infState.loopRunning = false;
  infState.image = null;
  _resumeSignal = null;

  // Reset upload UI
  if (elPreview) { elPreview.hidden = true; elPreview.src = ''; }
  if (elDropPrompt) elDropPrompt.classList.remove('hidden');
  if (elDropZone) elDropZone.classList.remove('has-image');
  if (elRunBtn) elRunBtn.disabled = true;
  if (elFileInput) elFileInput.value = '';

  // Reset progress
  if (elProgressFill) elProgressFill.style.width = '0%';

  // Reset canvas
  if (elCanvas) {
    const ctx = elCanvas.getContext('2d');
    ctx.clearRect(0, 0, elCanvas.width, elCanvas.height);
    ctx.fillStyle = '#060610';
    ctx.fillRect(0, 0, elCanvas.width, elCanvas.height);
  }

  transitionTo('hero');
}

// ── State transitions ─────────────────────────────────

function transitionTo(phase) {
  const screens = { hero: elHero, walkthrough: elWalkthrough, results: elResults };
  const current = screens[infState.phase];
  const next = screens[phase];

  if (current && current !== next) {
    current.classList.add('inf-state-exit');
    setTimeout(() => {
      current.hidden = true;
      current.classList.remove('inf-state-exit');
    }, 420);
  }

  if (next) {
    next.hidden = false;
    next.classList.add('inf-state-enter');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        next.classList.remove('inf-state-enter');
      });
    });
  }

  infState.phase = phase;
}

// ── Keyboard ──────────────────────────────────────────

function onKeydown(e) {
  if (infState.phase !== 'walkthrough') return;
  // Don't intercept if focus is on an input
  if (e.target.tagName === 'INPUT') return;

  if (e.code === 'Space') {
    e.preventDefault();
    togglePlayPause();
  }
  if (e.code === 'ArrowRight') {
    e.preventDefault();
    stepOnce();
  }
}

// ── Icons (SVG strings) ───────────────────────────────

const ICON_PLAY  = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>`;
const ICON_PAUSE = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
const ICON_STEP  = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/><rect x="19" y="3" width="2" height="18"/></svg>`;

// ── Escape helper ─────────────────────────────────────

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
