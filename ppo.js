'use strict';
const { MLP } = require('./nn.js');

function gauss(rng) { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const LOG2PI = Math.log(2 * Math.PI);

class PPO {
  constructor(obsDim, actDim, cfg = {}) {
    this.obsDim = obsDim; this.actDim = actDim;
    const rng = cfg.rng || Math.random;
    const hid = cfg.hidden || [64, 64];
    this.pi = cfg.makePi ? cfg.makePi(obsDim, actDim, rng) : new MLP([obsDim, ...hid, actDim], rng, 0.01);
    this.vf = cfg.makeVf ? cfg.makeVf(obsDim, 1, rng) : new MLP([obsDim, ...hid, 1], rng, 1.0);
    this.logStd = new Float64Array(actDim).fill(Math.log(cfg.initStd || 0.6));
    this.mLS = new Float64Array(actDim); this.vLS = new Float64Array(actDim); this.tLS = 0;
    this.gamma = cfg.gamma ?? 0.99; this.lam = cfg.lam ?? 0.95;
    this.clip = cfg.clip ?? 0.2; this.entCoef = cfg.entCoef ?? 0.0;
    this.vfCoef = cfg.vfCoef ?? 0.5; this.lr = cfg.lr ?? 3e-4;
    this.epochs = cfg.epochs ?? 10; this.mb = cfg.mb ?? 64; this.targetKL = cfg.targetKL ?? 0.02;
    this.rng = rng;
  }
  // sample (or deterministic) action; returns logProb + value for the rollout buffer
  act(obs, deterministic = false) {
    const mean = this.pi.forward(obs);
    const action = new Float64Array(this.actDim), m = new Float64Array(this.actDim);
    let logProb = 0;
    for (let i = 0; i < this.actDim; i++) {
      const std = Math.exp(this.logStd[i]);
      const a = deterministic ? mean[i] : mean[i] + std * gauss(this.rng);
      action[i] = a; m[i] = mean[i];
      const d = (a - mean[i]) / std;
      logProb += -0.5 * d * d - this.logStd[i] - 0.5 * LOG2PI;
    }
    const value = this.vf.forward(obs)[0];
    return { action, logProb, value, mean: m };
  }
  // batch: arrays obs[], act[], logp[], rew[], val[], done[] (per step, single env, sequential), plus lastVal for bootstrap
  computeGAE(rew, val, done, lastVal) {
    const T = rew.length, adv = new Float64Array(T), ret = new Float64Array(T);
    let gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nextV = t === T - 1 ? lastVal : val[t + 1];
      const nonterm = 1 - done[t];
      const delta = rew[t] + this.gamma * nextV * nonterm - val[t];
      gae = delta + this.gamma * this.lam * nonterm * gae;
      adv[t] = gae; ret[t] = gae + val[t];
    }
    return { adv, ret };
  }
  update(buf) {
    const { obs, act, logp, adv, ret } = buf;
    const N = obs.length;
    // normalize advantages
    let mean = 0; for (let i = 0; i < N; i++) mean += adv[i]; mean /= N;
    let v = 0; for (let i = 0; i < N; i++) v += (adv[i] - mean) ** 2; v /= N;
    const std = Math.sqrt(v) + 1e-8;
    const advN = new Float64Array(N); for (let i = 0; i < N; i++) advN[i] = (adv[i] - mean) / std;

    const idx = Array.from({ length: N }, (_, i) => i);
    let lastPiLoss = 0, lastVLoss = 0, lastKL = 0, stoppedAt = this.epochs;
    for (let ep = 0; ep < this.epochs; ep++) {
      let epKLsum = 0, epKLn = 0;
      // shuffle
      for (let i = N - 1; i > 0; i--) { const j = (this.rng() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
      for (let s = 0; s < N; s += this.mb) {
        const end = Math.min(s + this.mb, N), m = end - s;
        this.pi.zeroGrad(); this.vf.zeroGrad();
        const gLS = new Float64Array(this.actDim);
        let piLoss = 0, vLoss = 0, kl = 0;
        for (let k = s; k < end; k++) {
          const t = idx[k];
          const o = obs[t], a = act[t], oldlp = logp[t], A = advN[t], R = ret[t];
          // policy forward
          const mu = this.pi.forward(o);
          let newlp = 0;
          for (let i = 0; i < this.actDim; i++) { const sd = Math.exp(this.logStd[i]); const d = (a[i] - mu[i]) / sd; newlp += -0.5 * d * d - this.logStd[i] - 0.5 * LOG2PI; }
          const ratio = Math.exp(newlp - oldlp);
          kl += oldlp - newlp;
          // clipped surrogate: L = -min(ratio*A, clip(ratio)*A). dL/dnewlp:
          const unclipped = ratio * A;
          const clipped = Math.max(1 - this.clip, Math.min(1 + this.clip, ratio)) * A;
          let dL_dlp = 0;
          if (unclipped <= clipped) { dL_dlp = -A * ratio; piLoss += -unclipped; } // min is unclipped term -> grad flows
          else { dL_dlp = 0; piLoss += -clipped; }                                  // clipped branch -> no grad
          // backprop policy: dlp/dmu_i = (a-mu)/sd^2 ; dlp/dlogStd_i = (a-mu)^2/sd^2 - 1
          const gMu = new Float64Array(this.actDim);
          for (let i = 0; i < this.actDim; i++) {
            const sd = Math.exp(this.logStd[i]); const diff = a[i] - mu[i];
            gMu[i] = dL_dlp * (diff / (sd * sd));
            gLS[i] += dL_dlp * ((diff * diff) / (sd * sd) - 1);
            gLS[i] += this.entCoef * (-1); // entropy = sum(logStd + const); maximize -> subtract from loss grad
          }
          this.pi.backward(gMu);
          // value
          const Vp = this.vf.forward(o)[0];
          const dV = (Vp - R);
          vLoss += 0.5 * dV * dV;
          const gV = new Float64Array(1); gV[0] = this.vfCoef * dV;
          this.vf.backward(gV);
        }
        // average grads over minibatch
        const inv = 1 / m;
        this.pi.scaleGrad(inv); this.vf.scaleGrad(inv);
        this.pi.adamStep(this.lr); this.vf.adamStep(this.lr);
        // logStd Adam (averaged)
        this.tLS++;
        const bc1 = 1 - Math.pow(0.9, this.tLS), bc2 = 1 - Math.pow(0.999, this.tLS);
        for (let i = 0; i < this.actDim; i++) {
          let g = gLS[i] * inv; g = Math.max(-1, Math.min(1, g));
          this.mLS[i] = 0.9 * this.mLS[i] + 0.1 * g; this.vLS[i] = 0.999 * this.vLS[i] + 0.001 * g * g;
          this.logStd[i] -= this.lr * (this.mLS[i] / bc1) / (Math.sqrt(this.vLS[i] / bc2) + 1e-8);
          this.logStd[i] = Math.max(-2.5, Math.min(0.7, this.logStd[i])); // keep exploration sane
        }
        lastPiLoss = piLoss * inv; lastVLoss = vLoss * inv; lastKL = kl * inv;
        epKLsum += lastKL; epKLn++;
      }
      if (epKLn && (epKLsum / epKLn) > 1.5 * this.targetKL) { stoppedAt = ep + 1; break; } // KL early-stop -> stable steps
    }
    return { piLoss: lastPiLoss, vLoss: lastVLoss, kl: lastKL, std: Math.exp(this.logStd[0]), epochs: stoppedAt };
  }
}
module.exports = { PPO, gauss };
