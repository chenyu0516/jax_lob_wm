# Background Research — Discrete Market Making with JAX/RL

> Produced at the end of Step 1 of the SDD planning workflow. All external
> claims are tagged with their source. The user supplied the project spec
> directly (no anchor papers); external citations are used only to ground
> conventions and confirm modelling choices.

---

## 1. Domain context

**Market making** is the activity of providing two-sided liquidity in a
limit order book (LOB) by simultaneously posting a buy (bid) and a sell
(ask) limit order. The market maker (MM) profits by capturing the spread
between bid and ask, while taking on **inventory risk** — adverse mid-price
moves while the MM holds a non-zero position can wipe out spread P&L.

The canonical mathematical treatment is the **Avellaneda–Stoikov (2008)**
continuous-time model: the mid follows a Brownian motion, MOs arrive as a
Poisson process whose intensity decays exponentially with the distance
from the mid, and the MM solves an HJB equation for optimal bid/ask quotes
under an exponential utility of terminal wealth. Subsequent work
(Cartea, Jaimungal & Penalva; Guéant) discretized the problem and
generalized the inventory penalty into a running quadratic term `λq²`.

For reinforcement learning, the problem becomes a finite-horizon MDP:
state = (inventory, cash, time, market features), action = the pair of
quote distances from mid, reward = change in mark-to-market wealth minus
an inventory penalty. The model is appealing pedagogically because the
action space is small and discrete, the dynamics are cheap to simulate,
and the optimal policy has a known analytical reference in the
continuous-time limit.

The **engineering focus of this project** is orthogonal to the financial
modelling: it is a JAX-optimisation showcase. The aim is to demonstrate
that the *same* MDP, when implemented as a `gymnax`-style functional
environment, can be rolled out and trained orders of magnitude faster
than a naive Python-loop implementation by exploiting:

- `jax.lax.scan` for the time-step loop;
- `jax.vmap` for parallel-environment rollouts;
- end-to-end `jit` of the PPO train step (the PureJaxRL pattern);
- `vmap` over seeds / hyperparameters (notably `λ`) for parallel sweeps.

---

## 2. Prior art

| Work | Relevance |
|------|-----------|
| Avellaneda & Stoikov (2008), *High-Frequency Trading in a Limit Order Book* | Canonical continuous-time MM model with exponential fill intensities. The functional form `exp(−κδ)` for fill probabilities used in this project originates here. |
| Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading* | Textbook discretization; introduces the running inventory penalty `λq²` (rather than only a terminal penalty) that this project's reward inherits. ([source](https://arxiv.org/pdf/2403.02572)) |
| Spooner et al. (2018), *Market Making via Reinforcement Learning* | Discrete-action RL MM on a simulated LOB; reward = ΔPnL − inventory-risk penalty. Closest spiritual predecessor of the baseline this project builds. |
| Falces Marin et al. (2022), *A reinforcement learning approach to improve the Avellaneda-Stoikov MM algorithm* (PLOS One) | Deep-RL on top of an A-S simulator; reward defined as change in agent value discounted by inventory-volatility penalty. ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC9767337/)) |
| Lange (2022), *gymnax* | JAX-native re-implementation of the Gym API; defines the `env.reset / env.step` signature that this project conforms to. ([source](https://github.com/RobertTLange/gymnax)) |
| Lu et al. (2023), *Discovered Policy Optimisation* / **PureJaxRL** | Demonstrates ~4000× speedups by keeping the entire PPO loop on-device via `jit`/`scan`/`vmap`. This is the engineering pattern this project copies. ([source](https://chrislu.page/blog/meta-disco/)) |

---

## 3. Key concepts and terminology

- **Mid-price `S_t`**: reference price of the asset at step `t`. Lives on
  the integer tick grid. Evolves as
  `S_{t+1} = S_t + tick · round(σ · Z_t)`, `Z_t ~ N(0,1)` i.i.d.
- **Tick `Δ`**: the smallest price increment. All prices and quote
  distances are integer multiples of `Δ`.
- **Quote distances `(δ_b, δ_a)`**: the MM posts a bid at `S_t − δ_b·Δ`
  and an ask at `S_t + δ_a·Δ`. Action space:
  `{1,…,K}² ⊂ ℕ²`, action size = `K²`.
- **Fill probability** (per side, per step, independent Bernoulli):
  `P(bid filled) = exp(−κ·δ_b)`, `P(ask filled) = exp(−κ·δ_a)`.
  Each fill is for one share. `κ > 0` controls how steeply
  fill probability decays with distance from mid.
- **Inventory `q_t`**: signed number of shares held. Updates as
  `q_{t+1} = q_t + 𝟙{bid_fill} − 𝟙{ask_fill}`.
- **Cash `c_t`**: monetary balance. Updates as
  `c_{t+1} = c_t − (S_t − δ_b·Δ)·𝟙{bid_fill} + (S_t + δ_a·Δ)·𝟙{ask_fill}`.
- **Environment state** (carried through `lax.scan`): the full Markov
  state `(S_t, q_t, c_t, t, S_0)`. `S_0` is captured at reset so that
  the centered mid `S_t − S_0` can be computed; `t` is the integer step
  counter; PRNG keys are threaded separately.
- **Agent observation** (input to the PPO policy network):
  `obs_t = [(S_t − S_0)/(σ√T), q_t/q_max, c_t/c_scale, t/T]`. The
  normalisations keep every component O(1) regardless of episode length
  so the policy network sees a stationary input distribution across
  episodes and across `λ` sweeps. `t/T` exposes time-to-go so the agent
  can learn to flatten before forced liquidation. (User spec named the
  state as `[q, c]`; this expanded form is the *trainable* input —
  `S − S_0` and `t/T` are added because raw `S` drifts unboundedly under
  the random walk and `t` is required for the agent to anticipate
  terminal liquidation. The conceptual state-of-record remains
  inventory + cash.)
- **Reward per step**:
  `r_t = (S_{t+1}·q_{t+1} + c_{t+1}) − (S_t·q_t + c_t) − λ·q_{t+1}²`.
  (i.e., change in mark-to-market wealth minus running inventory
  penalty.)
- **Terminal liquidation**: at step `T`, the agent is forced to flat its
  position at the mid (no quote distance), so the terminal cash
  adjustment is `+ S_T·q_T`, and `q_T` becomes 0 for accounting.
- **Risk aversion `λ`**: scalar that controls how harshly large
  inventories are penalised each step. Swept across runs via `vmap`.

---

## 4. Constraints inherited from the domain

- **Tick discreteness**: prices, quote offsets and inventory are integer
  quantities. The simulator must keep them in integer dtypes where
  possible; converting to float only when computing reward or fill
  probabilities. This affects `lax.scan` carry signatures.
- **Stationarity and memorylessness**: in the chosen Bernoulli fill model
  the LOB has no memory — there is no resting order queue to track. This
  is what makes the environment a *pure* function suitable for
  `jit`/`scan`/`vmap` without any Python-side state.
- **Reward shape**: the `λq²` running penalty is the only mechanism
  pushing the agent toward inventory neutrality between fills; the
  forced terminal liquidation supplies a one-shot end-of-episode
  flattening incentive. Both must be implemented for the reward signal
  to be well-posed.
- **Determinism modulo PRNG**: the environment is a deterministic
  function of `(state, action, key)`. PRNG handling must follow the
  JAX functional pattern — keys are split and threaded through the
  scan carry, never stored as global state.
- **`gymnax` API conformance**: `env.reset(key, params) → (obs, state)`
  and `env.step(key, state, action, params) → (obs, state, reward, done, info)`
  must be pure jittable functions. No Python control flow that depends
  on traced values. ([source](https://github.com/RobertTLange/gymnax))

---

## 5. Engineering target — what the JAX speedup story is supposed to show

The project produces a tutorial-style benchmark comparison across four
implementation tiers, training the *same* PPO agent on the *same* MDP:

1. **Tier 0 — Naive Python loop** (NumPy + Python `for t in range(T)`).
   Baseline wall-clock.
2. **Tier 1 — `lax.scan` rollout** (single env, jitted step, scan over
   `T`). Expect a large speedup from compilation + fused step kernel.
3. **Tier 2 — `vmap` over `N` parallel envs** on one device. Throughput
   scales with `N` until device memory/compute saturates.
4. **Tier 3 — End-to-end PureJaxRL**: PPO update loop is itself inside
   `jit`/`scan`, and `vmap` runs many seeds and many `λ` values
   concurrently. Headline speedup vs. Tier 0.

The artifact is a single self-contained Jupyter notebook
(`notebooks/demo.ipynb`) intended as a live class demo: markdown
narrative interleaved with code cells that build up the env, then run
each tier with throughput numbers printed inline, learning curves and a
final `(seed × λ)` panel rendered with matplotlib. All standards of
achievement from the function spec are enforced by dedicated
`assert`-based sanity-check cells inside the notebook (no external
pytest suite — chosen so the project remains a single file students can
follow top-to-bottom).

---

## 6. Sources

- Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a
  limit order book." (foundational A-S model — exponential fill
  intensities, terminal-wealth utility)
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and
  High-Frequency Trading.* (running λq² inventory penalty;
  exponential-decay fill probabilities reviewed at
  https://arxiv.org/pdf/2403.02572)
- Spooner, T. et al. (2018). "Market Making via Reinforcement
  Learning." (discrete-action RL MM, ΔPnL − inventory penalty reward)
- Falces Marin, J. et al. (2022). "A reinforcement learning approach to
  improve the performance of the Avellaneda-Stoikov market-making
  algorithm." PLOS One.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9767337/
- Lange, R. T. (2022). gymnax — RL environments in JAX.
  https://github.com/RobertTLange/gymnax
- Lu, C. et al. (2022). "Discovered Policy Optimisation" / PureJaxRL.
  https://chrislu.page/blog/meta-disco/

User-supplied spec ([../coding_workflow.md](../coding_workflow.md) and the
project prompt) is the source of record for: state `[q, c]`, action
`(δ_b, δ_a)`, reward `Δ(S·q + c) − λq²`, fill kernel
`exp(−κ·δ)`, mid-RW `S + tick·round(σ·Z)`, PPO algorithm, fixed-T with
terminal liquidation. Where the spec was silent, defaults were chosen
from the prior art cited above and confirmed with the user in Step 1.
