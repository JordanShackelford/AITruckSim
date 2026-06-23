'use strict';
/* Single-query cross-attention scene encoder: the ego vector attends over a variable set of agent
   vectors (permutation-invariant, mask-aware), producing a context vector. [ego ; context] then
   feeds an MLP head. This is the scene-transformer representation (VectorNet/Wayformer family),
   replacing the flattened nearest-K MLP. Hand-derived backprop, gradient-checked.

   Exposes the same interface PPO expects (forward/backward/zeroGrad/scaleGrad/adamStep). */
const { MLP, xavier } = require('./nn.js');

function adamUpd(W, g, m, v, t, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
  for (let i = 0; i < W.length; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    W[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
  }
}

class AttnNet {
  // obs layout: [ego(egoDim), then K agents of agentDim each]; last feature of each agent = present(0/1)
  constructor(cfg) {
    const rng = cfg.rng || Math.random;
    this.egoDim = cfg.egoDim; this.K = cfg.K; this.agentDim = cfg.agentDim;
    this.dh = cfg.dh || 16; this.d = cfg.d || 16; this.dv = cfg.dv || 16;
    this.Wa = xavier(this.agentDim, this.dh, rng); this.ba = new Float64Array(this.dh);   // agent embed (tanh)
    this.Wq = xavier(this.egoDim, this.d, rng);                                            // query from ego
    this.Wk = xavier(this.dh, this.d, rng);                                                // key from agent embed
    this.Wv = xavier(this.dh, this.dv, rng);                                               // value from agent embed
    this.head = new MLP([this.egoDim + this.dv, ...(cfg.head || [64, 64]), cfg.outDim], rng, cfg.outScale || 1.0);
    this.encDim = this.egoDim + this.dv;
    // adam state
    this._g(); this._adam(); this.t = 0;
  }
  _g() { this.gWa = new Float64Array(this.Wa.length); this.gba = new Float64Array(this.ba.length);
    this.gWq = new Float64Array(this.Wq.length); this.gWk = new Float64Array(this.Wk.length); this.gWv = new Float64Array(this.Wv.length); }
  _adam() { const z = a => new Float64Array(a.length);
    this.mWa = z(this.Wa); this.vWa = z(this.Wa); this.mba = z(this.ba); this.vba = z(this.ba);
    this.mWq = z(this.Wq); this.vWq = z(this.Wq); this.mWk = z(this.Wk); this.vWk = z(this.Wk); this.mWv = z(this.Wv); this.vWv = z(this.Wv); }
  zeroGrad() { this.gWa.fill(0); this.gba.fill(0); this.gWq.fill(0); this.gWk.fill(0); this.gWv.fill(0); this.head.zeroGrad(); }
  scaleGrad(inv) { for (const g of [this.gWa, this.gba, this.gWq, this.gWk, this.gWv]) for (let i = 0; i < g.length; i++) g[i] *= inv; this.head.scaleGrad(inv); }

  forward(obs) {
    const { egoDim, K, agentDim, dh, d, dv } = this;
    const ego = obs.subarray ? obs.subarray(0, egoDim) : obs.slice(0, egoDim);
    // embed agents (tanh), build keys/values; record mask
    const H = [], Kk = [], V = [], agents = [], mask = [];
    for (let i = 0; i < K; i++) {
      const a = obs.subarray ? obs.subarray(egoDim + i * agentDim, egoDim + (i + 1) * agentDim) : obs.slice(egoDim + i * agentDim, egoDim + (i + 1) * agentDim);
      agents.push(a);
      mask.push(a[agentDim - 1] > 0.5 ? 1 : 0);
      const pe = new Float64Array(dh), h = new Float64Array(dh);
      for (let j = 0; j < dh; j++) { let s = this.ba[j], base = j * agentDim; for (let l = 0; l < agentDim; l++) s += this.Wa[base + l] * a[l]; pe[j] = s; h[j] = Math.tanh(s); }
      H.push(h);
      const k = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0, base = j * dh; for (let l = 0; l < dh; l++) s += this.Wk[base + l] * h[l]; k[j] = s; }
      const vv = new Float64Array(dv); for (let j = 0; j < dv; j++) { let s = 0, base = j * dh; for (let l = 0; l < dh; l++) s += this.Wv[base + l] * h[l]; vv[j] = s; }
      Kk.push(k); V.push(vv);
    }
    // query
    const q = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0, base = j * egoDim; for (let l = 0; l < egoDim; l++) s += this.Wq[base + l] * ego[l]; q[j] = s; }
    // scaled dot-product scores + masked softmax
    const scale = 1 / Math.sqrt(d), score = new Float64Array(K);
    let anyPresent = 0;
    for (let i = 0; i < K; i++) { if (!mask[i]) { score[i] = -1e30; continue; } anyPresent = 1; let s = 0; for (let j = 0; j < d; j++) s += q[j] * Kk[i][j]; score[i] = s * scale; }
    const alpha = new Float64Array(K);
    if (anyPresent) {
      let mx = -Infinity; for (let i = 0; i < K; i++) if (score[i] > mx) mx = score[i];
      let sum = 0; for (let i = 0; i < K; i++) { alpha[i] = Math.exp(score[i] - mx); sum += alpha[i]; }
      for (let i = 0; i < K; i++) alpha[i] /= sum;
    } // else all-zero context (no agents)
    // context
    const c = new Float64Array(dv); for (let i = 0; i < K; i++) { const ai = alpha[i]; if (ai === 0) continue; for (let j = 0; j < dv; j++) c[j] += ai * V[i][j]; }
    // encoding = [ego ; context]
    const enc = new Float64Array(egoDim + dv);
    for (let i = 0; i < egoDim; i++) enc[i] = ego[i]; for (let j = 0; j < dv; j++) enc[egoDim + j] = c[j];
    // cache for backward
    this._c = { ego, agents, H, Kk, V, mask, q, alpha, scale, anyPresent };
    return this.head.forward(enc);
  }

  backward(gOut) {
    const { egoDim, K, agentDim, dh, d, dv } = this;
    const C = this._c;
    const gEnc = this.head.backward(gOut);          // dL/d[ego;context]
    const gc = new Float64Array(dv); for (let j = 0; j < dv; j++) gc[j] = gEnc[egoDim + j];
    // dL/dalpha_i = gc . V_i ; dL/dV_i = alpha_i * gc
    const gAlpha = new Float64Array(K), gV = [];
    for (let i = 0; i < K; i++) { let s = 0; for (let j = 0; j < dv; j++) s += gc[j] * C.V[i][j]; gAlpha[i] = s; const gv = new Float64Array(dv); for (let j = 0; j < dv; j++) gv[j] = C.alpha[i] * gc[j]; gV.push(gv); }
    // softmax backward (present-only; absent alpha=0 so masked term contributes nothing)
    let dot = 0; for (let i = 0; i < K; i++) dot += gAlpha[i] * C.alpha[i];
    const gScore = new Float64Array(K);
    if (C.anyPresent) for (let i = 0; i < K; i++) gScore[i] = C.mask[i] ? C.alpha[i] * (gAlpha[i] - dot) : 0;
    // scores = scale*(q.k_i):  gq += scale*gScore_i*k_i ; gk_i = scale*gScore_i*q
    const gq = new Float64Array(d), gH = [];
    for (let i = 0; i < K; i++) gH.push(new Float64Array(dh));
    for (let i = 0; i < K; i++) {
      const gs = gScore[i] * C.scale;
      if (gs !== 0) for (let j = 0; j < d; j++) gq[j] += gs * C.Kk[i][j];
      const gk = new Float64Array(d); for (let j = 0; j < d; j++) gk[j] = gs * C.q[j];
      // key path: k_i = Wk h_i
      for (let j = 0; j < d; j++) { const gkj = gk[j], base = j * dh; for (let l = 0; l < dh; l++) { this.gWk[base + l] += gkj * C.H[i][l]; gH[i][l] += this.Wk[base + l] * gkj; } }
      // value path: v_i = Wv h_i  (gV_i = dL/dv_i)
      for (let j = 0; j < dv; j++) { const gvj = gV[i][j], base = j * dh; for (let l = 0; l < dh; l++) { this.gWv[base + l] += gvj * C.H[i][l]; gH[i][l] += this.Wv[base + l] * gvj; } }
    }
    // query: q = Wq ego
    for (let j = 0; j < d; j++) { const gqj = gq[j], base = j * egoDim; for (let l = 0; l < egoDim; l++) this.gWq[base + l] += gqj * C.ego[l]; }
    // embed: h_i = tanh(Wa a_i + ba)
    for (let i = 0; i < K; i++) {
      const a = C.agents[i];
      for (let j = 0; j < dh; j++) {
        const gpre = gH[i][j] * (1 - C.H[i][j] * C.H[i][j]);
        if (gpre === 0) continue;
        this.gba[j] += gpre; const base = j * agentDim;
        for (let l = 0; l < agentDim; l++) this.gWa[base + l] += gpre * a[l];
      }
    }
    return null; // ego is an input; no upstream net
  }

  adamStep(lr = 3e-4) {
    this.t++;
    adamUpd(this.Wa, this.gWa, this.mWa, this.vWa, this.t, lr); adamUpd(this.ba, this.gba, this.mba, this.vba, this.t, lr);
    adamUpd(this.Wq, this.gWq, this.mWq, this.vWq, this.t, lr); adamUpd(this.Wk, this.gWk, this.mWk, this.vWk, this.t, lr); adamUpd(this.Wv, this.gWv, this.mWv, this.vWv, this.t, lr);
    this.head.adamStep(lr);
  }
}
module.exports = { AttnNet };
