'use strict';
/* Imitation pretrain -> RL fine-tune.
   (1) Roll out the privileged MPC expert across a density curriculum, recording (policy-observation
       -> expert-action) pairs. The student records env._obs() (nearest-K vectorized scene) while the
       expert acts on full ground truth.
   (2) Behavior-clone: supervised regression of the policy MLP onto expert actions (MSE on the mean).
   (3) Inject the BC weights into a PPO policy and fine-tune. A from-scratch PPO run with identical
       settings is trained alongside as the honest baseline.

   Everything here is measured headlessly; no numbers are asserted that aren't printed by a run. */

const { MLP } = require('./nn.js');
const { PPO } = require('./ppo.js');
const { TruckEnv, OBS_DIM, ACT_DIM } = require('./env.js');
const { expertAction } = require('./expert.js');
const { AttnNet } = require('./attn.js');

function mkrng(s) { return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; }
const clampVec = a => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = Math.max(-1, Math.min(1, a[i])); return o; };

const N = +(process.env.N || 40000);          // imitation samples to collect
const EPOCHS = +(process.env.EPOCHS || 14);   // BC epochs
const ITERS = +(process.env.ITERS || 30);     // PPO fine-tune iterations
const HID = [64, 64];
const NET = process.env.NET || 'mlp';         // policy architecture: mlp | attn
const TRAIN_COUNTS = (process.env.COUNTS || '1,2,3,4').split(',').map(Number); // curriculum densities

// build a policy/value net of the selected architecture (same interface either way)
function makeNet(rng, outDim, outScale) {
  if (NET === 'attn') return new AttnNet({ rng, egoDim: 6, K: 5, agentDim: 5, dh: 16, d: 16, dv: 16, head: HID, outDim, outScale });
  return new MLP([OBS_DIM, ...HID, outDim], rng, outScale);
}
function resetAdam(net) { if (net.head && net._adam) { net._adam(); net.t = 0; net.head._initAdam(); } else { net._initAdam(); } }

// ---------- deterministic evaluation: act(obs)->action, fixed seeds per density ----------
function evalPolicy(act, counts = [1, 2, 3, 4, 5, 6], eps = 50) {
  const out = {};
  for (const n of counts) {
    const env = new TruckEnv(mkrng(9000 + n), { nObs: n, goalX: 1100, maxT: 700 });
    let goals = 0;
    for (let e = 0; e < eps; e++) {
      let o = env.reset(), done = false;
      while (!done) { const r = env.step(act(o)); o = r.obs; if (r.done) { if (r.info.end === 'goal') goals++; done = true; } }
    }
    out[n] = goals / eps * 100;
  }
  return out;
}
const fmt = ev => Object.keys(ev).map(k => 'n' + k + '=' + ev[k].toFixed(0) + '%').join('  ');

// ---------- (1) collect expert data ----------
function collect(nSamples) {
  const rng = mkrng(123);
  const obsBuf = [], actBuf = [];
  let epEnds = { goal: 0, crash: 0, jackknife: 0, offlane: 0, timeout: 0 }, eps = 0;
  while (obsBuf.length < nSamples) {
    const n = TRAIN_COUNTS[(rng() * TRAIN_COUNTS.length) | 0];
    const env = new TruckEnv(rng, { nObs: n, goalX: 1100, maxT: 700 });
    let o = env.reset(), done = false;
    const epObs = [], epAct = [];
    while (!done) {
      const a = expertAction(env);
      epObs.push(o); epAct.push(new Float64Array(a));
      const r = env.step(a); o = r.obs;
      if (r.done) { done = true; epEnds[r.info.end]++; eps++;
        // keep only trajectories that reached the goal -> clean expert demonstrations
        if (r.info.end === 'goal') { for (let i = 0; i < epObs.length; i++) { obsBuf.push(epObs[i]); actBuf.push(epAct[i]); } }
      }
    }
  }
  const keptGoalRate = (epEnds.goal / eps * 100).toFixed(0);
  return { obsBuf, actBuf, epEnds, eps, keptGoalRate };
}

// ---------- (2) behavior cloning (reusable trainer over an aggregated buffer) ----------
function trainOn(net, obsBuf, actBuf, rng, epochs, lr = 1e-3, mb = 256) {
  const M = obsBuf.length, idx = Array.from({ length: M }, (_, i) => i);
  let lastLoss = 0;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = M - 1; i > 0; i--) { const j = (rng() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
    let loss = 0;
    for (let s = 0; s < M; s += mb) {
      const end = Math.min(s + mb, M), m = end - s;
      net.zeroGrad();
      for (let k = s; k < end; k++) {
        const t = idx[k], o = obsBuf[t], a = actBuf[t];
        const out = net.forward(o);
        const g = new Float64Array(ACT_DIM);
        for (let d = 0; d < ACT_DIM; d++) { const e = out[d] - a[d]; g[d] = e; loss += 0.5 * e * e; }
        net.backward(g);
      }
      net.scaleGrad(1 / m); net.adamStep(lr);
    }
    lastLoss = loss / M;
  }
  return lastLoss;
}

// roll out the CURRENT student, relabel every visited state with the expert (DAgger aggregation)
function aggregateDagger(net, obsBuf, actBuf, rng, nAdd) {
  let added = 0;
  while (added < nAdd) {
    const n = TRAIN_COUNTS[(rng() * TRAIN_COUNTS.length) | 0];
    const env = new TruckEnv(rng, { nObs: n, goalX: 1100, maxT: 700 });
    let o = env.reset(), done = false;
    while (!done) {
      obsBuf.push(o); actBuf.push(new Float64Array(expertAction(env)));  // expert label at student-visited state
      added++;
      const a = clampVec(net.forward(o));                                // but DRIVE with the student
      const r = env.step(a); o = r.obs; done = r.done;
    }
  }
  return added;
}

// ---------- (3) PPO, with and without BC warm start ----------
function makePPO(rng) {
  const cfg = { rng, hidden: HID, lr: +(process.env.LR || 2.5e-4), initStd: 0.6, entCoef: 0.004, gamma: 0.99, lam: 0.95, epochs: 8, mb: 128, targetKL: +(process.env.KL || 0.05) };
  if (NET === 'attn') {
    cfg.makePi = (o, a, r) => makeNet(r, a, 0.01);
    cfg.makeVf = (o, a, r) => makeNet(r, 1, 1.0);
  }
  return new PPO(OBS_DIM, ACT_DIM, cfg);
}
function injectWeights(agent, bcNet) {
  agent.pi = bcNet;                              // adopt the imitation-trained policy directly
  resetAdam(agent.pi);                           // fresh optimizer state for the RL objective
  agent.logStd.fill(Math.log(+(process.env.WARMSTD || 0.4))); // exploration around the competent policy
}
function rollout(agent, env, T = 2048) {
  const obs = [], act = [], logp = [], rew = [], val = [], done = [];
  let o = env.reset(), ends = {}, rets = [], epR = 0;
  for (let t = 0; t < T; t++) {
    const { action, logProb, value } = agent.act(o);
    const r = env.step(action);
    obs.push(o); act.push(action); logp.push(logProb); val.push(value); rew.push(r.reward); done.push(r.done ? 1 : 0);
    epR += r.reward; o = r.obs;
    if (r.done) { rets.push(epR); epR = 0; ends[r.info.end] = (ends[r.info.end] || 0) + 1; o = env.reset(); }
  }
  const lastVal = agent.act(o).value;
  const { adv, ret } = agent.computeGAE(rew, val, done, lastVal);
  return { buf: { obs, act, logp, adv, ret }, ends };
}
function trainPPO(agent, label, rng) {
  const env = new TruckEnv(rng, { nObs: TRAIN_COUNTS[0], goalX: 1100, maxT: 700 });
  const pick = () => { env.nObs = TRAIN_COUNTS[(rng() * TRAIN_COUNTS.length) | 0]; };
  console.log('  ' + label + ' @iter0  ' + fmt(evalPolicy(o => agent.act(o, true).action, [1, 2, 3, 4], 30)));
  for (let it = 1; it <= ITERS; it++) {
    pick();
    const { buf } = rollout(agent, env);
    agent.update(buf);
    if (it % 10 === 0 || it === ITERS) console.log('  ' + label + ' @iter' + String(it).padStart(2) + '  ' + fmt(evalPolicy(o => agent.act(o, true).action, [1, 2, 3, 4], 30)));
  }
  return evalPolicy(o => agent.act(o, true).action, [1, 2, 3, 4, 5, 6], 50);
}

// ===================== run =====================
console.log('=== Imitation pretrain -> PPO fine-tune  [net=' + NET + ']  (curriculum densities ' + TRAIN_COUNTS.join(',') + ') ===');
const t0 = Date.now();

console.log('\n[1] collecting expert data...');
const { obsBuf, actBuf, epEnds, eps, keptGoalRate } = collect(N);
console.log('    expert ran ' + eps + ' episodes (goal rate ' + keptGoalRate + '%); kept ' + obsBuf.length + ' samples from goal-reaching episodes');
console.log('    expert episode ends: ' + JSON.stringify(epEnds));

console.log('\n[2] behavior cloning + DAgger...');
const DAGGER_ROUNDS = +(process.env.DAGGER || 5);
const ADD = +(process.env.ADD || 6000);
const trng = mkrng(777);
const bcNet = makeNet(mkrng(99), ACT_DIM, 1.0);
let loss = trainOn(bcNet, obsBuf, actBuf, trng, EPOCHS);
let ev = evalPolicy(o => clampVec(bcNet.forward(o)), [1, 2, 3, 4], 30);
console.log('  BC(0)  ' + obsBuf.length + ' samples  mse ' + loss.toFixed(4) + '   eval ' + fmt(ev));
for (let r = 1; r <= DAGGER_ROUNDS; r++) {
  aggregateDagger(bcNet, obsBuf, actBuf, trng, ADD);          // add student-visited states, expert-labeled
  loss = trainOn(bcNet, obsBuf, actBuf, trng, 4);            // refit on aggregated buffer
  ev = evalPolicy(o => clampVec(bcNet.forward(o)), [1, 2, 3, 4], 30);
  console.log('  DAgger(' + r + ') ' + obsBuf.length + ' samples  mse ' + loss.toFixed(4) + '   eval ' + fmt(ev));
}
const bcEval = evalPolicy(o => clampVec(bcNet.forward(o)), [1, 2, 3, 4, 5, 6], 50);
console.log('    imitation policy (no RL): ' + fmt(bcEval));

console.log('\n[3a] PPO fine-tune FROM BC warm start...');
const warm = makePPO(mkrng(2024)); injectWeights(warm, bcNet);
const warmFinal = trainPPO(warm, 'BC+PPO', mkrng(55));

console.log('\n[3b] PPO from scratch (baseline, identical settings)...');
const scratch = makePPO(mkrng(2024));
const scratchFinal = trainPPO(scratch, 'scratch', mkrng(55));

console.log('\n=== SUMMARY [net=' + NET + '] (deterministic goal%, 50 eps/density) ===');
console.log('  imitation only      : ' + fmt(bcEval));
console.log('  imit + PPO (' + ITERS + ' it) : ' + fmt(warmFinal));
console.log('  scratch PPO (' + ITERS + ' it): ' + fmt(scratchFinal));
console.log('  RESULT ' + NET + ' imit ' + [1,2,3,4,5,6].map(k => (bcEval[k]||0).toFixed(0)).join(' ') + ' | warm ' + [1,2,3,4,5,6].map(k => (warmFinal[k]||0).toFixed(0)).join(' ') + ' | scratch ' + [1,2,3,4,5,6].map(k => (scratchFinal[k]||0).toFixed(0)).join(' '));
console.log('  elapsed ' + ((Date.now() - t0) / 1000).toFixed(0) + 's');
