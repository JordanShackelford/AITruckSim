'use strict';
/* Articulated-truck driving env with a VECTORIZED-SCENE observation (ego state + nearest-K agents
   as relative pos/vel in the ego frame). Kinematics ported verbatim from the NEAT sim.
   Dense potential-based progress reward + difficulty knobs (nObs, goalX) for curriculum. */

const L1 = 38, L2 = 78, MAX_STEER = 0.55, JACKKNIFE = 1.61;
function normA(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; }

const LANE_HW = 70, V_MAX = 2.2, RANGE = 320, K = 5;
const OBS_DIM = 6 + K * 5, ACT_DIM = 2;

class TruckEnv {
  constructor(rng = Math.random, opts = {}) { this.rng = rng; this.OBS_DIM = OBS_DIM; this.ACT_DIM = ACT_DIM; this.goalX = opts.goalX || 1100; this.nObs = opts.nObs ?? 0; this.maxT = opts.maxT || 700; }
  reset() {
    this.x = 120; this.y = (this.rng() * 2 - 1) * 15; this.h1 = 0; this.h2 = 0; this.v = 1.0;
    this.t = 0;
    this.prevDist = this._goalDist();
    this.obs = [];
    for (let i = 0; i < this.nObs; i++) {
      const ox = 360 + this.rng() * (this.goalX - 480);
      const oy = (this.rng() * 2 - 1) * (LANE_HW - 18);
      const ovy = (this.rng() < 0.4) ? (this.rng() * 2 - 1) * 0.5 : 0;
      this.obs.push({ x: ox, y: oy, vx: 0, vy: ovy });
    }
    return this._obs();
  }
  _goalDist() { return Math.hypot(this.goalX - this.x, 0 - this.y); }
  _obs() {
    const o = new Float64Array(OBS_DIM);
    const goalAng = Math.atan2(0 - this.y, this.goalX - this.x);
    const hErr = normA(goalAng - this.h1);
    o[0] = Math.max(-1, Math.min(1, this.v / V_MAX));
    o[1] = normA(this.h1 - this.h2) / JACKKNIFE;
    o[2] = Math.sin(hErr); o[3] = Math.cos(hErr);
    o[4] = Math.max(-1.5, Math.min(1.5, this.y / LANE_HW));
    o[5] = Math.max(0, Math.min(1, (this.goalX - this.x) / this.goalX));
    const c = Math.cos(-this.h1), s = Math.sin(-this.h1);
    const rel = [];
    for (const ob of this.obs) {
      const dx = ob.x - this.x, dy = ob.y - this.y;
      const ex = dx * c - dy * s, ey = dx * s + dy * c;
      const d = Math.hypot(ex, ey);
      if (d < RANGE * 1.4) rel.push({ ex, ey, evx: (ob.vx * c - ob.vy * s), evy: (ob.vx * s + ob.vy * c), d });
    }
    rel.sort((a, b) => a.d - b.d);
    for (let i = 0; i < K; i++) {
      const base = 6 + i * 5;
      if (i < rel.length) {
        const r = rel[i];
        o[base] = Math.max(-1, Math.min(1, r.ex / RANGE));
        o[base + 1] = Math.max(-1, Math.min(1, r.ey / RANGE));
        o[base + 2] = Math.max(-1, Math.min(1, r.evx));
        o[base + 3] = Math.max(-1, Math.min(1, r.evy));
        o[base + 4] = 1;
      }
    }
    return o;
  }
  step(a) {
    const steer = Math.max(-1, Math.min(1, a[0])) * MAX_STEER;
    const thr = Math.max(-1, Math.min(1, a[1]));
    this.v = Math.max(-0.4, Math.min(V_MAX, this.v + thr * 0.08));
    this.x += this.v * Math.cos(this.h1); this.y += this.v * Math.sin(this.h1);
    this.h1 = normA(this.h1 + (this.v / L1) * Math.tan(steer));
    this.h2 = normA(this.h2 + (this.v / L2) * Math.sin(normA(this.h1 - this.h2)));
    for (const ob of this.obs) { ob.y += ob.vy; if (Math.abs(ob.y) > LANE_HW - 10) ob.vy *= -1; }
    this.t++;

    const dist = this._goalDist();
    let r = (this.prevDist - dist) * 0.03;        // dense potential-based progress
    r -= Math.abs(this.y) / LANE_HW * 0.003;       // mild centering
    r -= 0.003;                                    // time cost
    this.prevDist = dist;
    // dense proximity shaping: discourage approaching obstacles BEFORE the crash (teaches dodging)
    let nd = 1e9; for (const ob of this.obs) { const d = Math.hypot(ob.x - this.x, ob.y - this.y); if (d < nd) nd = d; }
    if (nd < 75) r -= (75 - nd) / 75 * 0.03;
    let done = false, info = {};

    for (const ob of this.obs) { if (Math.hypot(ob.x - this.x, ob.y - this.y) < 34) { r -= 8; done = true; info.end = 'crash'; break; } }
    if (!done) {
      if (Math.abs(normA(this.h1 - this.h2)) > JACKKNIFE) { r -= 8; done = true; info.end = 'jackknife'; }
      else if (Math.abs(this.y) > LANE_HW + 14) { r -= 6; done = true; info.end = 'offlane'; }
      else if (this.x >= this.goalX) { r += 12; done = true; info.end = 'goal'; }
      else if (this.t >= this.maxT) { done = true; info.end = 'timeout'; }
    }
    return { obs: this._obs(), reward: r, done, info };
  }
}
module.exports = { TruckEnv, OBS_DIM, ACT_DIM };
