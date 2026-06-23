'use strict';
/* Privileged analytic expert for the truck task. It reads ground-truth env state (full obstacle
   positions, exact pose) — something the learned policy never sees — and produces a normalized
   [steer, throttle] action in [-1,1], the SAME action space the policy uses. Used to generate
   imitation data: roll this out, record (policy-observation -> expert-action) pairs.

   Method: pure-pursuit toward a look-ahead point on a target lateral line; the target line is the
   lane center unless the nearest threatening obstacle ahead forces a lateral detour (pick the side
   that costs least steering while staying in-lane). A jackknife-damping term reduces steer when the
   trailer is lagging, and throttle eases off when a close obstacle isn't yet laterally cleared. */

const LANE_HW = 70, V_MAX = 2.2, JACKKNIFE = 1.61;

function normA(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; }
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const L1 = 38, L2 = 78, MAX_STEER = 0.55;

// Forward-simulate a constant (steer, targetSpeed) candidate through the EXACT kinematics and score
// it by progress while penalizing crash / jackknife / off-lane / proximity. Privileged: uses true
// obstacle state. This is a small MPC — the right tool since we have the model.
function rolloutScore(env, steer, targetV, p) {
  let x = env.x, y = env.y, h1 = env.h1, h2 = env.h2, v = env.v;
  const obs = env.obs.map(o => ({ x: o.x, y: o.y, vy: o.vy }));
  let score = 0, alive = p.H, maxAbsY = Math.abs(y);
  for (let t = 0; t < p.H; t++) {
    const st = clamp(steer, -1, 1) * MAX_STEER;
    const thr = clamp((targetV - v) * 2, -1, 1);
    v = clamp(v + thr * 0.08, -0.4, V_MAX);
    x += v * Math.cos(h1); y += v * Math.sin(h1);
    h1 = normA(h1 + (v / L1) * Math.tan(st));
    h2 = normA(h2 + (v / L2) * Math.sin(normA(h1 - h2)));
    if (Math.abs(y) > maxAbsY) maxAbsY = Math.abs(y);
    let mind = Infinity;
    for (const ob of obs) { ob.y += ob.vy; if (Math.abs(ob.y) > LANE_HW - 10) ob.vy *= -1; const d = Math.hypot(ob.x - x, ob.y - y); if (d < mind) mind = d; }
    if (mind < 34) { score -= 1000 * (p.H - t) / p.H; alive = t; break; }            // crash, weight by earliness
    if (Math.abs(normA(h1 - h2)) > JACKKNIFE) { score -= 1000 * (p.H - t) / p.H; alive = t; break; } // jackknife
    if (Math.abs(y) > LANE_HW + 14) { score -= 800 * (p.H - t) / p.H; alive = t; break; }            // off-lane
    if (mind < 58) score -= (58 - mind) * 0.45;                                       // keep-clear margin
    if (Math.abs(y) > 50) score -= (Math.abs(y) - 50) * 0.6;                          // stay off the lane edge
  }
  score += (x - env.x);                       // progress
  score -= 0.18 * Math.abs(y);                // centering (end pose)
  score -= 0.3 * Math.max(0, maxAbsY - 50);   // discourage edge-skimming paths
  score -= 0.15 * (p.H - alive);              // prefer staying alive longer
  return score;
}

function expertAction(env, p = {}) {
  p = { H: 18, ...p };
  const steers = [-1, -0.65, -0.35, -0.15, 0, 0.15, 0.35, 0.65, 1];
  const speeds = [V_MAX, 1.3, 0.7];
  let bestS = 0, bestV = V_MAX, best = -Infinity;
  for (const s of steers) for (const tv of speeds) {
    const sc = rolloutScore(env, s, tv, p);
    if (sc > best) { best = sc; bestS = s; bestV = tv; }
  }
  const thr = clamp((bestV - env.v) * 2.0, -1, 1);
  return [bestS, thr];
}

module.exports = { expertAction };

// ---- self-eval when run directly ----
if (require.main === module) {
  const { TruckEnv } = require('./env.js');
  function mkrng(s) { return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; }
  const counts = [0, 1, 2, 3, 4, 5, 6];
  console.log('expert self-eval (privileged controller), 100 eps each:');
  console.log(' nObs  goal%  crash%  jack%  offlane%  timeout%  avgSteps');
  for (const n of counts) {
    const env = new TruckEnv(mkrng(4242 + n), { nObs: n, goalX: 1100, maxT: 700 });
    const tally = { goal: 0, crash: 0, jackknife: 0, offlane: 0, timeout: 0 };
    let steps = 0, eps = 100;
    for (let e = 0; e < eps; e++) {
      env.reset(); let done = false;
      while (!done) { const r = env.step(expertAction(env)); steps++; if (r.done) { tally[r.info.end]++; done = true; } }
    }
    const pct = k => (tally[k] / eps * 100).toFixed(0).padStart(3);
    console.log('  ' + n + '    ' + pct('goal') + '    ' + pct('crash') + '   ' + pct('jackknife') +
      '     ' + pct('offlane') + '      ' + pct('timeout') + '     ' + (steps / eps).toFixed(0));
  }
}
