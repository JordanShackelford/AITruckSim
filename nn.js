'use strict';
/* Minimal MLP with explicit forward/backward (tanh hidden, linear output) + Adam.
   Pure JS so it can be verified headlessly and gradient-checked. The browser build
   swaps this layer for tfjs/WebGPU; the env + PPO logic stay identical. */

function xavier(nIn, nOut, rng) {
  const s = Math.sqrt(6 / (nIn + nOut));
  const w = new Float64Array(nIn * nOut);
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * s;
  return w;
}

class MLP {
  // sizes: [in, h1, h2, ..., out]; hidden act = tanh, output = linear
  constructor(sizes, rng = Math.random, outScale = 1.0) {
    this.sizes = sizes.slice();
    this.L = sizes.length - 1;
    this.W = []; this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const w = xavier(nIn, nOut, rng);
      if (l === this.L - 1) for (let i = 0; i < w.length; i++) w[i] *= outScale; // small final layer -> calm init policy
      this.W.push(w);
      this.b.push(new Float64Array(nOut));
    }
    this._initAdam();
  }
  _initAdam() {
    this.mW = this.W.map(w => new Float64Array(w.length));
    this.vW = this.W.map(w => new Float64Array(w.length));
    this.mB = this.b.map(b => new Float64Array(b.length));
    this.vB = this.b.map(b => new Float64Array(b.length));
    this.gW = this.W.map(w => new Float64Array(w.length));
    this.gB = this.b.map(b => new Float64Array(b.length));
    this.t = 0;
  }
  zeroGrad() {
    for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gB[l].fill(0); }
  }
  scaleGrad(inv) {
    for (let l = 0; l < this.L; l++) { const gW=this.gW[l],gB=this.gB[l]; for (let i=0;i<gW.length;i++)gW[i]*=inv; for (let i=0;i<gB.length;i++)gB[i]*=inv; }
  }
  // forward, caching pre-activations(z) and activations(a) for backward
  forward(x) {
    this._a = [x];
    let a = x;
    for (let l = 0; l < this.L; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], b = this.b[l];
      const z = new Float64Array(nOut);
      for (let j = 0; j < nOut; j++) {
        let s = b[j]; const base = j * nIn;
        for (let i = 0; i < nIn; i++) s += W[base + i] * a[i];
        z[j] = s;
      }
      let out;
      if (l < this.L - 1) { out = new Float64Array(nOut); for (let j = 0; j < nOut; j++) out[j] = Math.tanh(z[j]); }
      else out = z; // linear output
      this._a.push(out);
      a = out;
    }
    return a;
  }
  // backward: gOut = dLoss/dOutput (length out). Accumulates into gW/gB. Returns dLoss/dInput.
  backward(gOut) {
    let g = gOut;
    for (let l = this.L - 1; l >= 0; l--) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const aIn = this._a[l], aOut = this._a[l + 1];
      // through activation: hidden tanh -> dz = g * (1-a^2); output linear -> dz = g
      let dz = g;
      if (l < this.L - 1) { dz = new Float64Array(nOut); for (let j = 0; j < nOut; j++) dz[j] = g[j] * (1 - aOut[j] * aOut[j]); }
      const gW = this.gW[l], gB = this.gB[l], W = this.W[l];
      for (let j = 0; j < nOut; j++) {
        const dzj = dz[j]; gB[j] += dzj; const base = j * nIn;
        for (let i = 0; i < nIn; i++) gW[base + i] += dzj * aIn[i];
      }
      if (l >= 0) {
        const gIn = new Float64Array(nIn);
        for (let j = 0; j < nOut; j++) { const dzj = dz[j], base = j * nIn; for (let i = 0; i < nIn; i++) gIn[i] += W[base + i] * dzj; }
        g = gIn;
      }
    }
    return g; // dLoss/dInput
  }
  adamStep(lr = 3e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, clip = 0.5) {
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    // global grad-norm clip
    if (clip > 0) {
      let n = 0;
      for (let l = 0; l < this.L; l++) { const gW = this.gW[l], gB = this.gB[l]; for (let i = 0; i < gW.length; i++) n += gW[i] * gW[i]; for (let i = 0; i < gB.length; i++) n += gB[i] * gB[i]; }
      n = Math.sqrt(n);
      if (n > clip) { const s = clip / n; for (let l = 0; l < this.L; l++) { const gW = this.gW[l], gB = this.gB[l]; for (let i = 0; i < gW.length; i++) gW[i] *= s; for (let i = 0; i < gB.length; i++) gB[i] *= s; } }
    }
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l], b = this.b[l], gW = this.gW[l], gB = this.gB[l];
      const mW = this.mW[l], vW = this.vW[l], mB = this.mB[l], vB = this.vB[l];
      for (let i = 0; i < W.length; i++) {
        const g = gW[i]; mW[i] = b1 * mW[i] + (1 - b1) * g; vW[i] = b2 * vW[i] + (1 - b2) * g * g;
        W[i] -= lr * (mW[i] / bc1) / (Math.sqrt(vW[i] / bc2) + eps);
      }
      for (let j = 0; j < b.length; j++) {
        const g = gB[j]; mB[j] = b1 * mB[j] + (1 - b1) * g; vB[j] = b2 * vB[j] + (1 - b2) * g * g;
        b[j] -= lr * (mB[j] / bc1) / (Math.sqrt(vB[j] / bc2) + eps);
      }
    }
  }
}

module.exports = { MLP, xavier };
