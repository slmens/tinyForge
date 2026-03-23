/* ═══════════════════════════════════════════════════════
   TinyForge — wasm-bridge.js

   Thin bridge between JS and the WASM inference engine.
   Handles: WASM init, heap memory copies, op dispatch.
   ═══════════════════════════════════════════════════════ */

let wasm = null;

async function initWasm() {
  if (wasm) return;
  wasm = await TinyForgeEngine();
}

function wasmReady() {
  return wasm !== null;
}

// ── Heap helpers ─────────────────────────────────────

function toWasm(arr) {
  const bytes = arr.length * 4;
  const ptr = wasm._malloc(bytes);
  wasm.HEAPF32.set(arr, ptr >> 2);
  return ptr;
}

function fromWasm(ptr, count) {
  const out = new Float32Array(count);
  out.set(wasm.HEAPF32.subarray(ptr >> 2, (ptr >> 2) + count));
  return out;
}

// ── Op wrappers ──────────────────────────────────────
// Each function: copy tensors in → call WASM → copy result out → free.

function opConv(input, weight, bias, attrs) {
  const IC = input.shape[1], IH = input.shape[2], IW = input.shape[3];
  const OC = weight.shape[0], ICg = weight.shape[1];
  const KH = weight.shape[2], KW = weight.shape[3];
  const SH = attrs.strides[0], SW = attrs.strides[1];
  const pt = attrs.pads[0], pl = attrs.pads[1];
  const pb = attrs.pads[2], pr = attrs.pads[3];
  const DH = attrs.dilations[0], DW = attrs.dilations[1];
  const G  = attrs.group[0];

  const OH = Math.floor((IH + pt + pb - (DH * (KH - 1) + 1)) / SH) + 1;
  const OW = Math.floor((IW + pl + pr - (DW * (KW - 1) + 1)) / SW) + 1;
  const outCount = OC * OH * OW;

  const pInput  = toWasm(input.data);
  const pWeight = toWasm(weight.data);
  const pBias   = bias ? toWasm(bias.data) : 0;
  const pOutput = wasm._malloc(outCount * 4);

  wasm._tf_conv(
    pInput, IC, IH, IW,
    pWeight, OC, ICg, KH, KW,
    pBias,
    SH, SW, pt, pl, pb, pr, DH, DW, G,
    pOutput
  );

  const result = fromWasm(pOutput, outCount);
  wasm._free(pInput);
  wasm._free(pWeight);
  if (pBias) wasm._free(pBias);
  wasm._free(pOutput);

  return { data: result, shape: [1, OC, OH, OW] };
}

function opClip(input, minVal, maxVal) {
  const count = input.data.length;
  const pIn  = toWasm(input.data);
  const pOut = wasm._malloc(count * 4);

  wasm._tf_clip(pIn, pOut, count, minVal, maxVal);

  const result = fromWasm(pOut, count);
  wasm._free(pIn);
  wasm._free(pOut);
  return { data: result, shape: [...input.shape] };
}

function opAdd(a, b) {
  const count = a.data.length;
  const pA   = toWasm(a.data);
  const pB   = toWasm(b.data);
  const pOut = wasm._malloc(count * 4);

  wasm._tf_add(pA, pB, pOut, count);

  const result = fromWasm(pOut, count);
  wasm._free(pA);
  wasm._free(pB);
  wasm._free(pOut);
  return { data: result, shape: [...a.shape] };
}

function opReduceMean(input, keepdims) {
  const C = input.shape[1], H = input.shape[2], W = input.shape[3];
  const pIn  = toWasm(input.data);
  const pOut = wasm._malloc(C * 4);

  wasm._tf_reduce_mean(pIn, pOut, C, H, W);

  const result = fromWasm(pOut, C);
  wasm._free(pIn);
  wasm._free(pOut);
  return { data: result, shape: keepdims ? [1, C, 1, 1] : [1, C] };
}

function opReshape(input, targetShape) {
  const resolved = [...targetShape];
  let unknownIdx = -1;
  let known = 1;
  const totalSize = input.data.length;

  for (let i = 0; i < resolved.length; i++) {
    if (resolved[i] === 0) resolved[i] = input.shape[i];
    if (resolved[i] === -1) unknownIdx = i;
    else known *= resolved[i];
  }
  if (unknownIdx >= 0) resolved[unknownIdx] = totalSize / known;

  return { data: input.data, shape: resolved };
}

function opGemm(A, B, bias, attrs) {
  const transB = attrs.transB ? attrs.transB[0] : 0;
  const K = A.shape[1];
  const N = transB ? B.shape[0] : B.shape[1];

  const pA    = toWasm(A.data);
  const pB    = toWasm(B.data);
  const pBias = bias ? toWasm(bias.data) : 0;
  const pOut  = wasm._malloc(N * 4);

  wasm._tf_gemm(pA, pB, pBias, pOut, N, K);

  const result = fromWasm(pOut, N);
  wasm._free(pA);
  wasm._free(pB);
  if (pBias) wasm._free(pBias);
  wasm._free(pOut);
  return { data: result, shape: [1, N] };
}

function opSoftmax(input) {
  const N = input.data.length;
  const pIn  = toWasm(input.data);
  const pOut = wasm._malloc(N * 4);

  wasm._tf_softmax(pIn, pOut, N);

  const result = fromWasm(pOut, N);
  wasm._free(pIn);
  wasm._free(pOut);
  return { data: result, shape: [...input.shape] };
}
