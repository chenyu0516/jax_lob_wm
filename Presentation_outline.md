## Presentation Outline

---

### Part 1 — The RL Framework

- **Environment**: The LOB simulator is the environment, following the standard `(state, action, reward, next_state)` loop
- **State**: `Φ(LOB) = (Imb, n)` — the projected book representation
- **Action**: Orders submitted by the strategy (type, side, size)
- **Reward**: Realized P&L, penalized by market impact `ϕₜ`
- **Episode**: One trading session, stepping through Algorithm 1
- **Why gymnax**: We need to run this loop over thousands of random seeds and parameter sweeps simultaneously — standard PyTorch loops over episodes sequentially

---

### Part 2 — Gymnax-Specific Functions & What They Accelerate

---

**`step_env` + `jax.jit`**
- What it replaces: Python re-entry at every event in the simulation loop
- What it accelerates: The entire Algorithm 1 — event sampling, book update, impact state update — is compiled once into a single XLA kernel and never leaves the accelerator between steps

---

**`reset_env` + `jax.jit`**
- What it replaces: Re-initializing the book in Python between episodes
- What it accelerates: Parameter sweeps and Monte Carlo runs — the 100,000+ metaorder paths from §4.1 are reset and re-run without touching Python

---

**`get_obs_space` / `get_action_space` + `jax.vmap`**
- What it replaces: A Python `for` loop over parallel environments or manual CUDA batching in PyTorch
- What it accelerates: Running thousands of independent LOB simulations simultaneously across random seeds — one `vmap` call over `step_env` replaces the entire outer loop

---

**`jax.lax.scan` inside `step_env`**
- What it replaces: The `while simulating` loop of Algorithm 1
- What it accelerates: The event-by-event trajectory — instead of unrolling in Python or breaking out of a compiled graph, `scan` keeps the entire episode as a single compiled computation with no Python overhead per step