/* ═══════════════════════════════════════════════════════
   TinyForge — engine.js

   Graph runner for MobileNetV2 inference.
   Handles: image preprocessing, weight loading,
   graph walking, and result extraction.
   Calls into wasm-bridge.js for the actual math.
   ═══════════════════════════════════════════════════════ */

// ── Image preprocessing ──────────────────────────────
// Resize to 224x224, normalize with ImageNet stats, convert to NCHW.

const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

function preprocessImage(imageElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, 224, 224);
  const pixels = ctx.getImageData(0, 0, 224, 224).data;

  const out = new Float32Array(3 * 224 * 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const px = (y * 224 + x) * 4;
      for (let c = 0; c < 3; c++) {
        const val = pixels[px + c] / 255.0;
        out[c * 224 * 224 + y * 224 + x] = (val - MEAN[c]) / STD[c];
      }
    }
  }
  return { data: out, shape: [1, 3, 224, 224] };
}

// ── Weight loading ───────────────────────────────────
// Fetch the binary weights file and slice it into named tensors.

async function loadWeights(graphData) {
  const res = await fetch('../assets/weights.bin');
  if (!res.ok) throw new Error(`Failed to load weights: HTTP ${res.status}`);
  const buffer = await res.arrayBuffer();

  const weights = {};
  for (const [name, info] of Object.entries(graphData.tensors)) {
    if (info.has_data) {
      const arr = new Float32Array(buffer, info.offset, info.byte_length / 4);
      weights[name] = { data: arr, shape: info.shape };
    }
  }
  return weights;
}

// ── Top-K extraction ─────────────────────────────────

function topK(probs, k) {
  const indices = Array.from({ length: probs.data.length }, (_, i) => i);
  indices.sort((a, b) => probs.data[b] - probs.data[a]);
  return indices.slice(0, k).map(i => ({
    index: i,
    probability: probs.data[i],
  }));
}

// ── Graph runner ─────────────────────────────────────
// Walk every node in order, dispatch to the WASM bridge,
// store activations, yield to the browser between nodes.

async function runInference(graphData, weights, inputTensor, onProgress) {
  await initWasm();

  const activations = {};
  activations['input'] = inputTensor;

  for (const [name, tensor] of Object.entries(weights)) {
    activations[name] = tensor;
  }

  const nodes = graphData.nodes;
  const totalStart = performance.now();

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const nodeStart = performance.now();
    let result;

    switch (node.op) {
      case 'Conv': {
        const input  = activations[node.inputs[0]];
        const weight = activations[node.inputs[1]];
        const bias   = node.inputs[2] ? activations[node.inputs[2]] : null;
        result = opConv(input, weight, bias, node.attrs);
        break;
      }

      case 'Clip': {
        const input  = activations[node.inputs[0]];
        const minT   = activations[node.inputs[1]];
        const maxT   = activations[node.inputs[2]];
        const minVal = minT ? minT.data[0] : 0;
        const maxVal = maxT ? maxT.data[0] : 6;
        result = opClip(input, minVal, maxVal);
        break;
      }

      case 'Add': {
        const a = activations[node.inputs[0]];
        const b = activations[node.inputs[1]];
        result = opAdd(a, b);
        break;
      }

      case 'ReduceMean': {
        const input = activations[node.inputs[0]];
        const keepdims = node.attrs.keepdims ? node.attrs.keepdims[0] : 1;
        result = opReduceMean(input, keepdims);
        break;
      }

      case 'Reshape': {
        const input  = activations[node.inputs[0]];
        const shapeT = activations[node.inputs[1]];
        const target = Array.from(shapeT.data).map(v => Math.round(v));
        result = opReshape(input, target);
        break;
      }

      case 'Gemm': {
        const A    = activations[node.inputs[0]];
        const B    = activations[node.inputs[1]];
        const bias = node.inputs[2] ? activations[node.inputs[2]] : null;
        result = opGemm(A, B, bias, node.attrs);
        break;
      }

      default:
        throw new Error(`Unknown op: ${node.op}`);
    }

    activations[node.outputs[0]] = result;

    const nodeMs = performance.now() - nodeStart;
    const totalMs = performance.now() - totalStart;

    if (onProgress) {
      onProgress(i, nodes.length, node, result.shape, nodeMs, totalMs, result, activations);
    }

    // Yield to browser so the UI can update
    await new Promise(r => setTimeout(r, 0));
  }

  const logits = activations['output'];
  const probs  = opSoftmax(logits);
  const top5   = topK(probs, 5);

  return {
    logits,
    probs,
    top5,
    totalMs: performance.now() - totalStart,
  };
}
