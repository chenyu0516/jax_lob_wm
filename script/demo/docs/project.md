# Discrete Market Making with JAX — Project Plan

> Output of the SDD planning workflow (Steps 2–6). This document is the
> source-of-truth specification for the implementation skill that
> follows. Background and citations live in [./background.md](./background.md).

---

## References

- [Background research](./background.md) — domain context, prior art, sources.
- [../coding_workflow.md](../coding_workflow.md) — the SDD+TDD baseline this project follows.

---

## Brief goal

Build a discrete single-level market-making MDP as a `gymnax`-conformant
JAX environment, and demonstrate via a 4-tier benchmark (naive Python
loop → `lax.scan` → `vmap` over parallel envs → end-to-end PureJaxRL
with `vmap` over seeds and `λ`) that PPO training wall-clock time
decreases measurably at each tier while producing a policy that learns
inventory-controlled P&L behaviour consistent with the textbook MM
objective.

---

## Using scenario

### Platform

NVIDIA CUDA GPU laptop running Jupyter (Linux or Windows-with-WSL).
macOS is the development environment; the demo target has CUDA. The
notebook auto-detects `jax.default_backend()` at startup so it remains
runnable (with reduced sweep size) on CPU.

### Performance envelope

- Each tier's training cell completes within roughly one minute of
  live-class time.
- Headline Tier 3 / Tier 0 ratio ≥ 100× on the demo GPU.
- The `(seed × λ)` sweep packs at least 4 × 4 = 16 concurrent PPO runs
  into one `vmap` call.
- Per-step env latency (Tier 1, jitted) under 50 µs on the demo GPU.

### Software/hardware limits

- Single artefact: `notebooks/demo.ipynb`. All real code lives in
  notebook cells; there is no companion Python package.
- Pinned stack: `jax[cuda12]`, `flax`, `optax`, `gymnax`, `matplotlib`.
  No external test suite — standards of achievement are enforced by
  in-notebook `assert`-based sanity-check cells.
- No real market data. The mid-price is synthetic (random walk on the
  tick grid).

### User interaction model

Live class demo. The instructor runs `notebooks/demo.ipynb` cell by
cell while teaching. Students watch; they do not fork or rerun the
notebook themselves.

---

## Language & hardware decisions

- **Language: Python 3.11+.** Forced by the JAX stack, by `gymnax`
  (Python-only), and by the notebook deliverable. No alternative is on
  the table.
- **Hardware: NVIDIA CUDA GPU laptop.** The 4-tier benchmark's headline
  result (Tier 3 / Tier 0 ≥ 100×) depends on SIMT execution on a real
  GPU; a CPU-only run still demonstrates each pattern but with a
  smaller ratio. Notebook defaults shrink to a CPU-friendly grid if no
  GPU is detected.
- **Libraries pinned**: `jax`, `flax` (NN modules), `optax` (Adam),
  `gymnax` (env API), `matplotlib`. Explicitly *not* using
  `equinox`, `haiku`, `stable-baselines`, or `flashbax`.

---

## Documented decisions and overrides

During Steps 1–5 the planner pushed back on three choices the user
adopted anyway. Recorded here so the rationale survives.

1. **Self-contained notebook vs. importable package**
   - User chose self-contained notebook with inline `assert` cells.
   - Planner's pushback: kills external `pytest`, makes refactoring
     cells across tiers awkward, and the PPO update alone is a ~80-line
     cell. The user accepted these costs in exchange for a single-file
     class demo that students can follow top-to-bottom.

2. **Drop bitwise / statistical equivalence clause from the brief goal**
   - User chose to omit any cross-tier equivalence requirement.
   - Planner's pushback: without equivalence the "same policy at each
     tier" claim is informal — a Tier-1 bug that changes the math could
     pass unnoticed. User accepted because the demo focus is wall-clock,
     not numerical correctness across tiers.

3. **Observation includes raw mid `S` (resolved to centered form)**
   - User initially asked for raw `S` in the observation; planner
     pushed back on unbounded drift; user adopted the centered/normalised
     form `[(S − S₀)/(σ√T), q/q_max, c/c_scale, t/T]` instead. Resolved
     in the planner's favour; recorded so the choice is traceable.

---

## Functions

Six high-level capabilities. Each is defined in Given-When-Then form
with a standard of achievement. The implementation breakdown of these
capabilities into notebook cells lives in the project tree below.

### Function 1 — Simulate one MM episode

- **Given** a `gymnax`-conformant env, an initial PRNG key, an episode
  length `T`, and a (possibly trivial) policy mapping `obs → (δ_b, δ_a) ∈ {1..K}²`,
- **When** the env is reset and stepped for `T` timesteps,
- **Then** the env produces a length-`T` trajectory of
  `(obs, action, reward, next_obs, done)` consistent with:
  (a) mid follows `S_{t+1} = S_t + tick · round(σZ)`,
  (b) bid/ask fills are independent Bernoulli with `P = exp(−κδ)`,
  (c) `r_t = Δ(S·q + c) − λ·q²`,
  (d) at `t = T` inventory is force-flattened at mid.
- **Standard of achievement**: reset + `T` steps is a pure jittable
  function of `(key, params)`; rerunning with the same `key` yields the
  same trajectory; the identity
  `Σ r_t = (S_T·q_T + c_T) − (S_0·q_0 + c_0) − λ·Σ q_t²` holds within
  `1e-5` float tolerance (asserted in Subsection 1.7).

### Function 2 — Train a PPO agent on the MM env

- **Given** training hyperparameters (network shape, PPO clip, learning
  rate, `n_envs`, `n_updates`, `T`, `λ`) and a seed,
- **When** the training loop runs to completion,
- **Then** it returns trained `(policy_params, value_params)` plus a
  learning curve (mean episode return per update).
- **Standard of achievement**: trained policy attains mean episode
  return strictly higher than a fixed-symmetric-spread heuristic over
  32 held-out validation seeds; mean absolute inventory `E[|q_t|]` is
  measurably below the heuristic's (concrete thresholds set during
  implementation against observed heuristic numbers). Asserted in
  Subsection 4.5.

### Function 3 — Parallel-env rollouts via `vmap`

- **Given** a fixed policy, a batch of `N` PRNG keys, and `T`,
- **When** `vmap(rollout_single)` is invoked on the keys,
- **Then** it returns a batched trajectory of shape `(N, T, …)`.
- **Standard of achievement**: for `N ≤ 8`, per-env outputs match
  running `rollout_single` `N` times sequentially within `1e-5`
  tolerance; wall-clock for batched-`N` is sub-linear in `N` on the
  target device. Asserted in Subsection 2.5.

### Function 4 — Hyperparameter sweep via `vmap` over `(seed, λ)`

- **Given** a 2-D grid `seeds × λs` of shape `(M, L)` and a fixed network
  spec,
- **When** the jitted `train` is `vmap`'d over the grid,
- **Then** it returns learning curves of shape `(M, L, n_updates)` from
  a single device call.
- **Standard of achievement**: produces a non-degenerate `(M, L)` panel
  where mean trained inventory variance is monotonically decreasing in
  `λ` — the qualitative MM result. Asserted/visualised in Subsection
  5.3.

### Function 5 — 4-tier benchmark

- **Given** a fixed env, policy network, batch size and `T`,
- **When** the benchmark harness times each of Tiers 0–3,
- **Then** it produces a throughput table (env-steps / sec; updates /
  sec; wall-clock per `n_updates`).
- **Standard of achievement**: throughput strictly increases across
  tiers on the demo GPU; the headline Tier 3 / Tier 0 ratio is ≥ 100×.
  Numbers reproducible across two runs within 10%. Visualised in
  Subsection 5.1.

### Function 6 — Reporting (notebook narrative + figures)

- **Given** all outputs from Functions 1–5,
- **When** the notebook is run top-to-bottom,
- **Then** it produces: throughput table, overlaid learning curves per
  tier, a `(seed × λ)` inventory-variance panel, and a sample-episode
  trace for the best `(seed, λ)`.
- **Standard of achievement**: every figure renders inline at legible
  size; the notebook completes end-to-end without errors on the demo
  hardware; closing markdown narrative summarises the speedup story.
  Visual check only (no `assert`).

---

## Project tree

5 sections, 27 subsections. Each subsection has an SDD **specification**
(what it is responsible for; inputs/outputs/dependencies — no
implementation detail) and **paraphrased tests** (assertions enforced
either in dedicated sanity-check cells or as visual checks). All cells
live in the single artefact `notebooks/demo.ipynb`.

---

### Section 1 — Environment

The `gymnax`-conformant MM MDP. Defines the dynamics, the reward, and
the observation. Reused by every later section.

#### Subsection 1.1 — `EnvParams` and `EnvState` data types

##### Specification
Two frozen pytrees holding (a) the immutable hyperparameters of the
environment (`T`, `K`, `tick`, `σ`, `κ`, `λ`, `S_0`, normalisation
scales `q_max`, `c_scale`) and (b) the carried state per timestep
(`S_t`, `q_t`, `c_t`, `t`, `S_0`). `EnvParams` is constructed once and
passed to every `step` call; `EnvState` is the scan-carry. Dependencies:
none — these types are the contract every other env subsection
implements against.

##### Tests
- Construction with default values returns a valid pytree (no `None`s,
  all leaves jax-array or python int).
- `EnvParams` is hashable (so it can sit on the static side of
  `jax.jit`).
- `EnvState` round-trips through `jax.tree_util.tree_flatten /
  tree_unflatten` unchanged.

#### Subsection 1.2 — Mid-price random walk step

##### Specification
A pure function that, given the current `S_t` and a PRNG key, returns
the next mid `S_{t+1} = S_t + tick · round(σ · Z)` with `Z ~ N(0, 1)`.
Output dtype: integer-valued float (rounded). No side effects. Depends
on `EnvParams` for `tick` and `σ`.

##### Tests
- Over `n = 100_000` samples with `σ = 1, tick = 1`, the empirical
  variance of the increment is within 5 % of the theoretical
  `Var[round(Z)] ≈ 0.74` (after-rounding variance).
- Increments are zero-mean within 3 standard errors.
- Two identical keys produce bitwise-identical outputs.

#### Subsection 1.3 — Fill kernel

##### Specification
A pure function that, given quote distances `(δ_b, δ_a) ∈ {1..K}²` and
a PRNG key, returns `(filled_b, filled_a) ∈ {0, 1}²` as two independent
Bernoulli draws with marginal probabilities `exp(−κ·δ_b)` and
`exp(−κ·δ_a)`. Depends on `EnvParams` for `κ`.

##### Tests
- Over `n = 100_000` samples at `δ = 1`, the empirical fill rate is
  within 1 % of `exp(−κ)`.
- Fills on bid and ask are statistically independent (sample correlation
  within 3 standard errors of zero).
- `P(fill | δ = K)` is positive but ≤ `P(fill | δ = 1)` — monotone
  decreasing in `δ`.

#### Subsection 1.4 — `env.reset`

##### Specification
`reset(key, params) → (obs, state)`. Initialises `S_0`, `q_0 = 0`,
`c_0 = 0`, `t = 0`, and returns the initial observation built per
Subsection 1.6. Pure function; PRNG-keyed only to leave room for future
random initial-state extensions. Depends on subsections 1.1 and 1.6.

##### Tests
- `state.q == 0` and `state.c == 0` after reset.
- `state.S == state.S_0` (the initial mid is captured for centering).
- `obs[0]` (the centered `S`) is exactly `0.0` at reset.
- `obs[3]` (the `t/T` channel) is exactly `0.0` at reset.

#### Subsection 1.5 — `env.step`

##### Specification
`step(key, state, action, params) → (obs, next_state, reward, done, info)`.
Composes Subsections 1.2 (mid RW) and 1.3 (fills) to update
`(S, q, c, t)`; computes reward `r_t = Δ(S·q + c) − λ·q²` against the
*post-action* state; at `t == T − 1` forces a terminal liquidation
(closes any remaining inventory at the new mid, sets `q = 0`, adds the
liquidation cash, marks `done = True`). Pure jittable function.
Depends on Subsections 1.1, 1.2, 1.3, 1.6.

##### Tests
- After a non-terminal step with both quotes filled at `δ = 1`,
  inventory changes by zero and cash changes by `+2·tick`
  (captured spread).
- After a terminal step, `next_state.q == 0` and `done == True`.
- Stepping with a fixed `key` produces deterministic output bit-for-bit
  across calls (purity).
- `info` carries the fill flags so downstream code can audit
  trajectories.

#### Subsection 1.6 — `env.get_obs`

##### Specification
Given `state` and `params`, returns the four-channel observation
`[(S − S_0)/(σ·√T), q/q_max, c/c_scale, t/T]`. Pure function. Depends
on Subsection 1.1.

##### Tests
- All four channels are finite for any reachable state.
- At reset: channel 0 = 0, channel 3 = 0.
- After `T − 1` steps the time channel equals `(T − 1)/T`.
- Inputs at the extremes (`|q| = q_max`, `c = c_scale`) produce
  channels within `[−1, 1]`.

#### Subsection 1.7 — Sanity cell: reward identity and fill rate

##### Specification
A standalone notebook cell that runs `M = 16` random rollouts of length
`T` with a fixed dummy policy and asserts the standard of achievement
of Function 1. No new functions defined — this cell composes the env
and a trivial policy and audits the trajectory.

##### Tests
- For each rollout: `Σ r_t` equals
  `(S_T·q_T + c_T) − (S_0·q_0 + c_0) − λ·Σ q_t²` within `1e-5`.
- Empirical bid-fill rate at the dummy policy's chosen `δ` matches
  `exp(−κ·δ)` within 1 %.
- Terminal inventory is exactly zero in every rollout.

---

### Section 2 — Rollout speedups (Tiers 0 → 1 → 2)

Three reference rollout implementations sharing one timing utility.
This section delivers the first half of the headline speedup story.

#### Subsection 2.1 — Tier 0: pure-Python rollout

##### Specification
A function `rollout_python(env, params, policy, key, T)` that runs a
Python `for t in range(T):` loop calling `env.step` and accumulating
the trajectory in a list. No JIT, no scan. Acts as the wall-clock
baseline for the benchmark. Depends on Section 1.

##### Tests
- Output trajectory has length `T`.
- Final state passes the reward identity from Subsection 1.7 (same
  math, different orchestration).
- Asserts no part of the loop is jitted (raises if `policy` is wrapped
  in `jit` to keep the comparison honest).

#### Subsection 2.2 — Timing utility

##### Specification
A helper `time_callable(fn, n_warmup, n_repeat, args)` that runs `fn`
once for warmup (with `block_until_ready`), then `n_repeat` more times,
and reports median wall-clock per call. Used by every later timing
cell. Pure utility — no env dependency.

##### Tests
- For a no-op function the reported time is positive and below 10 ms.
- For a function that sleeps 100 ms the reported time is within 10 % of
  100 ms.
- Returns both median and inter-quartile range so the timing table can
  show variance.

#### Subsection 2.3 — Tier 1: `lax.scan` rollout, jitted

##### Specification
A function `rollout_scan(env, params, policy, key, T)` that places the
entire timestep loop inside `jax.lax.scan` and wraps the whole
function in `jax.jit`. Carries `(state, key)`; emits per-step
`(obs, action, reward, done)`. Depends on Section 1.

##### Tests
- Output trajectory shapes are `(T, …)` and match the Tier 0 shapes.
- The jitted function compiles successfully (first-call warmup
  succeeds without `ConcretizationTypeError` or `TracerArrayConversionError`).
- After warmup, median wall-clock is strictly lower than Tier 0 for
  the same `T`.

#### Subsection 2.4 — Tier 2: `vmap` over `N` parallel envs

##### Specification
A function `rollout_batch(env, params, policy, keys, T)` defined as
`jax.vmap(rollout_scan, in_axes=(None, None, None, 0, None))` over a
batch of `N` PRNG keys, returning a trajectory of shape `(N, T, …)`.
Depends on Subsection 2.3.

##### Tests
- Output shape is `(N, T, …)` for input keys of shape `(N,)`.
- For `N = 4` and identical sub-keys, every batch element is
  bitwise-identical (sanity check for vmap correctness).
- Throughput per env-step is strictly higher than Tier 1 on the demo
  hardware.

#### Subsection 2.5 — Sanity cell: monotone throughput and vmap match

##### Specification
A standalone cell that times Tiers 0, 1, 2 on identical `(env, T)` and
asserts the standards of achievement for Functions 3 and 5 (partial —
only Tiers 0–2; Tier 3 is asserted in Subsection 4.5).

##### Tests
- Median wall-clock per env-step strictly decreases across Tiers 0,
  1, 2.
- For `N = 8`, per-env trajectory output of `rollout_batch` matches
  running `rollout_scan` `N` times sequentially within `1e-5`.

---

### Section 3 — Policy & PPO building blocks

The neural network policy/value head, the action mapping, the GAE
advantage estimator, and the PPO clipped-objective update. Decoupled
from the env so the algorithm reads cleanly on its own.

#### Subsection 3.1 — Actor-critic network

##### Specification
A `flax.linen` module with shared trunk MLP (two hidden layers, `tanh`
activations), branching into (a) a categorical actor head over `K²`
logits and (b) a scalar critic head. Takes observation shape
`(4,)` from Subsection 1.6 and returns `(logits, value)`. Pure
parameterised function; weights live in a separate pytree. Depends on
Subsection 1.6 (observation contract).

##### Tests
- `init(key, dummy_obs)` returns a params pytree of finite leaves.
- Forward pass on a batch of `(B, 4)` observations returns logits of
  shape `(B, K²)` and values of shape `(B,)`.
- Logits are not NaN for any input within `[−1, 1]^4`.

#### Subsection 3.2 — Action mapping

##### Specification
A bijection between a single integer action `a ∈ {0, …, K² − 1}` and a
pair `(δ_b, δ_a) ∈ {1, …, K}²`. Used in two places: the env consumes
the pair; PPO samples / log-probs the integer. Pure utility, no
trainable state.

##### Tests
- The mapping is bijective: every integer in range maps to a unique
  pair and vice versa.
- Round-trip `pair → int → pair` returns the input exactly.
- Vectorises cleanly under `vmap` (returns `(B, 2)` for input shape `(B,)`).

#### Subsection 3.3 — GAE advantage computation

##### Specification
Given a trajectory of `(rewards, values, dones)` of shape `(T,)`, a
discount `γ` and a GAE smoothing `λ_GAE`, return advantages and
returns of shape `(T,)`. Pure JAX function, vectorisable under `vmap`
over the batch axis. Standard PureJaxRL formulation.

##### Tests
- For `γ = 0`, advantages equal `rewards − values`.
- For `γ = 1, λ_GAE = 1`, advantages equal Monte Carlo returns minus
  values.
- Output shape matches input.
- Across a batch of `(N, T)` inputs under `vmap`, results match calling
  the function `N` times sequentially.

#### Subsection 3.4 — PPO clipped loss + optax update

##### Specification
A pure function `ppo_update(params, opt_state, batch) → (new_params, new_opt_state, info)`
that computes the clipped surrogate loss, the value loss, and the
entropy bonus on a minibatch, then applies an Adam step via `optax`.
Depends on Subsection 3.1 and the GAE outputs from 3.3.

##### Tests
- Gradients of the loss with respect to `params` are finite for all
  leaves.
- After 100 update calls on a fixed synthetic batch with no env
  randomness, the loss strictly decreases on average (monotone
  enough; a 10-update window mean is monotone non-increasing).
- The function is jit-compatible: wrapping with `jit` succeeds.

#### Subsection 3.5 — Sanity cell: gradients and toy convergence

##### Specification
A standalone cell that instantiates a small actor-critic, runs 100
`ppo_update` calls on a fixed synthetic batch, and asserts the standard
of achievement for the PPO building blocks.

##### Tests
- All grads are finite at every step.
- The loss trace is monotone non-increasing under a 10-step rolling
  mean.
- Final logits differ from initial logits by more than `1e-3` (the
  update actually moved the policy).

---

### Section 4 — Tier 3: End-to-end PureJaxRL

Compose Section 1 (env), Section 2.4 (`vmap` rollouts), and Section 3
(PPO) into a single jitted training loop, then `vmap` it across
`(seed, λ)`.

#### Subsection 4.1 — Single train iteration

##### Specification
A function `train_step(carry, _) → (new_carry, info)` (the scan body)
that performs one PPO iteration: collect `n_envs × T` rollouts via
Subsection 2.4, compute GAE via 3.3, run `n_epochs × n_minibatches`
applications of `ppo_update` from 3.4, and return updated `(params,
opt_state, key)`. Pure jittable function. Depends on Sections 1–3.

##### Tests
- Calling `train_step` once advances `params` (i.e., they change).
- The returned `info` dict contains finite `loss`, `entropy`,
  `mean_return`.
- Carry shape is identical pre- and post-call (so it composes under
  `lax.scan`).

#### Subsection 4.2 — Full train loop via `lax.scan`

##### Specification
A function `train(hp, seed, λ) → (trained_params, learning_curve)` that
runs `lax.scan(train_step, init_carry, xs=None, length=n_updates)` and
returns final params plus a length-`n_updates` curve of mean episode
returns. The entire function is wrapped in `jax.jit`. Depends on 4.1.

##### Tests
- Output learning curve has shape `(n_updates,)`.
- Final params differ from initial params on every leaf.
- Re-running with the same `seed` produces identical curves
  (determinism).

#### Subsection 4.3 — `vmap` over `(seed, λ)` grid

##### Specification
The full sweep entry point: a function `sweep(hp, seeds, lambdas)` that
`vmap`s `train` over both axes, returning a learning-curve tensor of
shape `(M, L, n_updates)` and a params pytree of shape `(M, L, …)`.
Depends on 4.2.

##### Tests
- Output curve tensor shape is `(M, L, n_updates)`.
- For `M = L = 1`, the output matches a single `train` call.
- The sweep completes in less wall-clock time than running `M × L`
  `train` calls sequentially (the speedup point).

#### Subsection 4.4 — Headline timing: Tier 3 vs Tier 0

##### Specification
A timing cell that runs Tier 0 for a representative work unit (training
proxy: `n_updates × n_envs × T` env-steps under a random policy because
Tier 0 cannot actually train a PPO agent in reasonable time) and Tier 3
for the same total work. Reports both wall-clock times and the
ratio. Depends on Subsections 2.1 and 4.3.

##### Tests
- Tier 3 wall-clock < Tier 0 wall-clock.
- Reported ratio is ≥ 100× on the demo GPU (the headline). On CPU the
  bar drops to ≥ 30× and prints a `(CPU)` annotation.

#### Subsection 4.5 — Sanity cell: trained policy beats fixed-spread

##### Specification
A standalone cell that, using the trained params from Subsection 4.3 at
the best `(seed, λ)`, runs 32 validation rollouts and compares mean
episode return and mean `|q_t|` against a fixed-symmetric-spread
heuristic (`δ_b = δ_a = δ*` for a hand-picked `δ*`). Asserts Function
2's standard of achievement.

##### Tests
- Trained mean episode return > heuristic mean episode return.
- Trained `E[|q_t|]` < heuristic `E[|q_t|]`.
- The trained policy's action distribution is not degenerate (entropy
  above a small floor — rules out a collapsed policy).

---

### Section 5 — Analysis & figures

Pure presentation cells. No new mechanics. Every cell here consumes
outputs already produced by Sections 1–4.

#### Subsection 5.1 — Throughput summary table

##### Specification
A markdown / pandas-DataFrame table comparing the four tiers on:
median wall-clock per call, env-steps / sec, and speedup ratio vs.
Tier 0. Depends on the timing outputs from Subsections 2.5 and 4.4.

##### Tests (visual)
- The four rows appear in order Tier 0 → Tier 3.
- The "speedup vs Tier 0" column is monotone non-decreasing.

#### Subsection 5.2 — Overlaid learning curves

##### Specification
A matplotlib plot overlaying the mean (± std across seeds) learning
curves from Tier 3 for each `λ` in the sweep. Depends on Subsection
4.3 output.

##### Tests (visual)
- Each `λ` curve is monotonically increasing in mean over the first
  half of training (sanity for learning at all).
- Curves are labelled with their `λ` value.

#### Subsection 5.3 — `(seed × λ)` inventory-variance panel

##### Specification
A figure showing, for each `λ`, the distribution of `Var[q_t]` across
seeds (e.g., a violin or box plot). The headline visual for Function 4.
Depends on the trained params from 4.3 and a validation rollout pass.

##### Tests (visual)
- The median inventory variance is monotonically decreasing in `λ`
  (the qualitative MM result; if this fails we have a bug).
- Each violin has at least 4 seeds plotted.

#### Subsection 5.4 — Sample episode trace for best `(seed, λ)`

##### Specification
A 4-panel time-series plot for one trained policy on one validation
seed: top — mid-price `S_t` with bid/ask quotes overlaid; second —
inventory `q_t`; third — cash `c_t`; bottom — cumulative reward.
Depends on a single validation rollout.

##### Tests (visual)
- The plotted bid is always strictly below the mid; ask strictly
  above.
- Final inventory is zero (terminal liquidation visible).
- The cumulative reward ends positive on the chosen "best"
  `(seed, λ)`.

#### Subsection 5.5 — Closing narrative

##### Specification
A markdown-only cell summarising: (a) the speedup ratios actually
achieved, (b) the qualitative MM finding from the `λ` sweep, (c) the
limitations (single level, Bernoulli fills, no adverse selection, no
real data), and (d) suggested extensions for follow-on lectures
(stateful book, multi-level, adverse selection, real data, multi-agent).

##### Tests
- None. Markdown only — visual review by instructor.

---

## Implementation order (suggested)

The natural construction order respects the dependency arrows above:

1. **Section 1** end-to-end (env data types → RW → fills → reset → step
   → obs → sanity cell). Section 1 must pass Subsection 1.7 before
   anything else.
2. **Section 2.1, 2.2, 2.3** (Tier 0 + timing + Tier 1) — earliest
   point at which the speedup story has a first data point.
3. **Section 2.4, 2.5** (Tier 2 + sanity).
4. **Section 3** in order 3.1 → 3.2 → 3.3 → 3.4 → 3.5 (network → action
   map → GAE → PPO update → sanity).
5. **Section 4.1 → 4.2 → 4.5** (train step → train loop → policy beats
   heuristic). 4.5 gates the rest of Section 4 — if PPO isn't actually
   learning, the sweep and timing are meaningless.
6. **Section 4.3 → 4.4** (`vmap` sweep → headline timing).
7. **Section 5** in order (table → curves → panel → trace → narrative).

This ordering is a suggestion; the implementation skill should follow
it unless it has a concrete reason to deviate.
