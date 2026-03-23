/* ═══════════════════════════════════════════════════════
   TinyForge — inference.cpp

   Naive inference engine compiled to WebAssembly.
   No optimizations. Every operation is the simplest
   possible nested-loop implementation.

   Compiled with: emcc inference.cpp -o inference.js
   ═══════════════════════════════════════════════════════ */

#include <emscripten/emscripten.h>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ── Memory management exposed to JS ──────────────────

extern "C" {

EMSCRIPTEN_KEEPALIVE
float* tf_alloc(int count) {
    return (float*)malloc(count * sizeof(float));
}

EMSCRIPTEN_KEEPALIVE
void tf_free(float* ptr) {
    free(ptr);
}

// ── Conv ─────────────────────────────────────────────
// input:  [1, IC, IH, IW]
// weight: [OC, IC/G, KH, KW]
// bias:   [OC] (can be nullptr)
// output: caller-allocated [1, OC, OH, OW]
// Returns: OH * 10000 + OW (packed output dims)

EMSCRIPTEN_KEEPALIVE
int tf_conv(
    const float* input, int IC, int IH, int IW,
    const float* weight, int OC, int ICg, int KH, int KW,
    const float* bias,
    int SH, int SW, int pt, int pl, int pb, int pr,
    int DH, int DW, int G,
    float* output
) {
    int OH = (IH + pt + pb - (DH * (KH - 1) + 1)) / SH + 1;
    int OW = (IW + pl + pr - (DW * (KW - 1) + 1)) / SW + 1;
    int ocPerGroup = OC / G;

    for (int oc = 0; oc < OC; oc++) {
        int g = oc / ocPerGroup;
        float biasVal = bias ? bias[oc] : 0.0f;

        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float sum = biasVal;

                for (int ic = 0; ic < ICg; ic++) {
                    int inputC = g * ICg + ic;
                    int inputBase = inputC * IH * IW;
                    int weightBase = oc * ICg * KH * KW + ic * KH * KW;

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - pt + kh * DH;
                        if (ih < 0 || ih >= IH) continue;
                        int inputRow = inputBase + ih * IW;
                        int weightRow = weightBase + kh * KW;

                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - pl + kw * DW;
                            if (iw < 0 || iw >= IW) continue;
                            sum += input[inputRow + iw] * weight[weightRow + kw];
                        }
                    }
                }

                output[oc * OH * OW + oh * OW + ow] = sum;
            }
        }
    }

    return OH * 10000 + OW;
}

// ── Clip (ReLU6) ─────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void tf_clip(const float* input, float* output, int count, float minVal, float maxVal) {
    for (int i = 0; i < count; i++) {
        float v = input[i];
        if (v < minVal) v = minVal;
        if (v > maxVal) v = maxVal;
        output[i] = v;
    }
}

// ── Add (element-wise) ──────────────────────────────

EMSCRIPTEN_KEEPALIVE
void tf_add(const float* a, const float* b, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = a[i] + b[i];
    }
}

// ── ReduceMean over spatial dims ────────────────────
// input: [1, C, H, W] → output: [1, C, 1, 1]

EMSCRIPTEN_KEEPALIVE
void tf_reduce_mean(const float* input, float* output, int C, int H, int W) {
    int spatial = H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        const float* base = input + c * spatial;
        for (int i = 0; i < spatial; i++) {
            sum += base[i];
        }
        output[c] = sum / (float)spatial;
    }
}

// ── Gemm: C = A @ B^T + bias ───────────────────────
// A: [1, K], B: [N, K] (transposed), bias: [N]
// output: [1, N]

EMSCRIPTEN_KEEPALIVE
void tf_gemm(const float* A, const float* B, const float* bias,
             float* output, int N, int K) {
    for (int n = 0; n < N; n++) {
        float sum = bias ? bias[n] : 0.0f;
        const float* bRow = B + n * K;
        for (int k = 0; k < K; k++) {
            sum += A[k] * bRow[k];
        }
        output[n] = sum;
    }
}

// ── Softmax ─────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
void tf_softmax(const float* input, float* output, int N) {
    float maxVal = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > maxVal) maxVal = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - maxVal);
        sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

} // extern "C"
