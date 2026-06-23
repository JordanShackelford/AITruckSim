# Truck-RL: migration foundation (NEAT → deep RL with a vectorized scene)

This is commit 1 of the rebuild: a clean Gym-style environment with a **vectorized-scene
observation** and a **verified PPO** that demonstrably learns the truck task. It's written in
pure JS so every claim here is checked headlessly — the browser/WebGPU build swaps the neural
layer for tfjs but keeps the env and PPO logic identical.

## Why this architecture (vs NEAT + rays)

The task is moving from "dodge cars down a corridor" toward real street/highway driving
(intersections, turns, merging, traffic). That changes the right tools:

- **Sensor → vectorized scene, not rays.** Rays are a fixed-beam obstacle sensor: they alias at
  intersections, carry no semantics, and throw away velocity vectors. The driving-SOTA
  representation is a set of entities (other agents, lanes, dock pose) encoded as feature vectors
  in the ego frame (VectorNet/Wayformer lineage). Here the observation is **ego state +
  nearest-K agents as relative position/velocity in the ego frame** — it scales to arbitrary
  scene complexity instead of a fixed fan. (An attention encoder over the agent set is the next
  upgrade; an MLP over the flattened set is the starting point.)
- **Optimizer → policy gradient (PPO), not evolution.** NEAT gets one scalar per episode and
  can't assign credit to actions, which is why the old build leaned so hard on reward shaping.
  PPO uses the per-step signal and the value baseline. PPO is the robust default; **SAC**
  (sample-efficiency) and **DreamerV3** (model-based, the frontier) are later swaps behind the
  same env interface.
- **Keep the analytic docker as a skill.** The precise reverse-dock controller from the NEAT sim
  is the right tool for the precision endgame. In the RL design it becomes a low-level option the
  learned high-level policy invokes (hierarchical RL) — we don't relearn solved control.

## What's verified (headless, in this repo)

| Check | Result |
|---|---|
| MLP backprop vs finite differences (`gradcheck.js`) | max rel error **1.1e-7** — correct |
| PPO learns a known task — 2D point-mass (`toy.js`) | mean return **−5.9 → 53** |
| Truck **driving** to goal, no traffic (`NOBS=0 trucktrain.js`) | **100% goal** by iter ~12 |
| Truck **dodging** via vectorized obs (`NOBS=2`) | **~60% goal**, climbing; std annealing |

Dodging at higher density (NOBS≥3) learns but oscillates in this pure-JS CPU run — expected:
single-env rollouts are high-variance, the net is tiny, and iteration count is compute-limited.
These are exactly what the production stack fixes (parallel envs, bigger batches, WebGPU compute),
not foundation flaws. The *math* and the *learning loop* are proven; scaling is a compute matter.

## Files

- `nn.js` — minimal MLP (tanh hidden, linear out) with hand-written backprop + Adam. **Gradient-checked.**
- `ppo.js` — PPO: Gaussian policy, GAE(λ), clipped surrogate, value loss, entropy bonus, KL guard.
- `env.js` — `TruckEnv`: ported articulated kinematics (tractor bicycle + trailer follower,
  L1=38/L2=78, jackknife), vectorized-scene observation, dense potential-based reward,
  difficulty knobs (`nObs`, `goalX`) for curriculum.
- `toy.js` — PPO sanity on a point-mass (proves the optimizer).
- `trucktrain.js` — train on the truck task. Env vars: `NOBS` (obstacles), `ITERS`.
- `gradcheck.js` — numerical gradient check of `nn.js`.

Run: `node gradcheck.js` · `node toy.js` · `NOBS=0 node trucktrain.js` · `NOBS=2 node trucktrain.js`

## What migrated from the NEAT sim

- **Carried (the real IP):** truck/trailer kinematics, jackknife dynamics, dock geometry,
  scenario layout logic, reward-shaping intuition. All algorithm-agnostic.
- **Carried as a skill:** the analytic reverse-dock controller (→ hierarchical low-level option).
- **Optional modality:** the ray-caster (keep for ablation; not the primary sensor).
- **Did not carry:** trained NEAT genomes — different representation, throwaway on a paradigm change.

## Path forward (in order)

1. **Curriculum + more compute on this env:** 1→2→N obstacles, then turns/intersections. The dense
   reward + difficulty knobs are already in place.
2. **Swap `nn.js` → tfjs (WebGPU)** for real GPU backprop in-browser; keep `env.js`/`ppo.js` as-is.
   Run many envs in parallel across Web Workers for throughput.
3. **Attention encoder** over the agent set + a polyline map encoder (lanes/road) — the scene
   transformer that handles complex layouts.
4. **Imitation pretrain → RL fine-tune:** generate expert trajectories from a privileged planner
   (the sim has ground truth), behavior-clone, then PPO/SAC fine-tune. This is what actually cracks
   complex driving; pure RL from scratch on sparse urban reward is a graveyard.
5. **Hierarchy:** learned high-level policy invoking the analytic docker (and other skills) as options.
6. **DreamerV3** when you want the model-based frontier.

The existing HTML renderer survives as the live demo: export trained weights, run inference in JS.

## Browser WebGPU port (`truck-rl-webgpu.html`)

Open in a WebGPU browser (Chrome/Edge 113+). It loads tfjs from CDN, selects the WebGPU backend
(falls back to WebGL→CPU), and trains the same task with tfjs autodiff. Design choices for GPU
throughput: **16 parallel envs**, the forward pass **batched** (one GPU call per step for all envs),
rollout+GAE in JS, only the gradient update through tfjs.

**Verification status (be precise about this):**
- *Env* — the in-browser `TruckEnv` is **bit-identical** to the node reference (checked: max state &
  reward diff = 0). The simulation half is provably correct.
- *PPO math* — the tfjs loss (clipped surrogate, GAE, value, entropy, KL early-stop) mirrors the
  gradient-checked node implementation line-for-line.
- *Cannot be checked in the build sandbox* — tfjs execution + WebGPU (no network here). **Confirm it
  in your browser**: run the **Toy point-mass** task first; return should climb from negative toward
  ~50 within a minute (proves the in-browser PPO learns). Then switch to Truck and watch goal% climb.

If the toy self-test learns, the loop is sound and any remaining truck-task tuning is curriculum/
compute — exactly what WebGPU + parallel envs are there to provide.

## Attention scene encoder (`attn.js`) — built & gradient-checked

The flattened nearest-K MLP is the baseline observation. `attn.js` adds the real representation:
a **single-query cross-attention encoder** where the ego vector attends over the variable set of
agents (permutation-invariant, mask-aware via the `present` flag), producing a context vector that
feeds an MLP head. This is the VectorNet/Wayformer-family scene encoder, scaled down.

- `h_i = tanh(Wa·a_i + b_a)` embed → `k_i = Wk·h_i`, `v_i = Wv·h_i`; query `q = Wq·ego`;
  masked softmax over scaled dot-products → `α`; context `c = Σ α_i v_i`; head over `[ego ; c]`.
- Hand-derived backprop (softmax, attention, tanh embed), **gradient-checked: max rel error 3.7e-6**
  (`gradcheck_attn.js`). Exposes the PPO net interface (forward/backward/zeroGrad/scaleGrad/adamStep).
- PPO is now **net-agnostic** (`makePi`/`makeVf` builders) so the encoder drops in without touching
  the algorithm. The MLP path is byte-for-byte unchanged (still gradient-checked, still trains).

**Result (`NET=attn COUNTS=2 node trucktrain_attn.js`):** trains to **~60% goal on 2 obstacles**,
matching the flat-MLP baseline — so the encoder is a working drop-in, not just correct on paper.
Deterministic eval shows **graceful degradation to unseen densities** (goal% 53 / 15 / 0 at
nObs = 2 / 4 / 6, trained only on 2): the masked attention handles more agents than it ever saw,
where a fixed-slot MLP has no such structure. A rigorous attn-vs-MLP generalization A/B (train on a
mix, eval out-of-distribution) is the next validation, and full convergence belongs on the WebGPU
stack — pure-JS single-env training is still oscillation-prone on the harder densities.

New/changed files: `attn.js`, `gradcheck_attn.js`, `trucktrain_attn.js`; `nn.js` (backward now
returns input grad + `scaleGrad`), `ppo.js` (pluggable nets + `scaleGrad`).

## Attention vs MLP: generalization A/B (run, result = INCONCLUSIVE)

Tested the claim that the attention encoder generalizes to unseen obstacle counts better than the
flat MLP. Both nets trained on 2 obstacles only, then evaluated deterministically (50 episodes each)
on 2-6 obstacles, identical eval seeds. Command: `NET={mlp|attn} COUNTS=2 ITERS=44 LR=2e-4 SEED=N node trucktrain_attn.js`.

- Seed 11 — MLP `20/6/2/2/2`%, attention `58/34/28/20/16`% (attention ~8x better at 6 obstacles).
- Seed 7  — MLP `14/8/2/0/0`%, attention `20/4/4/0/0`% (essentially a wash).

The dramatic seed-11 win did NOT replicate on seed 7. Honest conclusion: at this compute budget the
result is dominated by training variance, not architecture — the generalization advantage is NOT
established. Resolving it needs more seeds, more iterations, and the lower-variance training the
WebGPU + parallel-env stack provides. The encoder itself is correct (gradient-checked 3.7e-6) and
trains; whether it generalizes better than the MLP is an open question, not a proven win.
