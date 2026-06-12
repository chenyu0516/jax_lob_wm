# Mathematical Framework — `demo.ipynb`

This document specifies the mathematical objects implemented in
[`demo.ipynb`](../demo.ipynb): the market-making environment (MDP), the
actor–critic trading agent, and the PPO policy-optimization objective.

---

## 1. Environment — Finite-Horizon Market-Making MDP

The environment is a discrete-time, finite-horizon Markov decision
process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, T)$ implemented in
the `MMEnv` class.

### 1.1 State

The internal state at time $t \in \{0, 1, \dots, T\}$ is

$$
x_t = (S_t,\; q_t,\; c_t) \in \mathbb{R} \times \mathbb{Z} \times \mathbb{R} ,
$$

where
- $S_t$ is the **mid-price** (random walk on a tick grid),
- $q_t$ is the **inventory** (signed integer number of units held),
- $c_t$ is the **cash** account,

The episode is initialized at $x_0 = (S_0, 0, c_0)$ with $S_0 = 100$ and $c_0 = 1000$

### 1.2 Action and quote bijection

The agent chooses a discrete action $(a_t,b_t) \in \{0, 1, \dots, K\}^2$
* $\tau$ is a tick price, 0.01 here
* $a_t$ stands for placing an ask order at price $S_t + \delta_a = S_t + a_t\times \tau$ 
* $b_t$ stands for placing an ask order at price $S_t - \delta_b = S_t - b_t\times \tau$ 

### 1.3 Mid-price dynamics
#### Dynamic programming

The mid evolves as a discretized random walk on the tick grid:

$$
S_{t+1} = S_t + \tau \cdot \mathrm{round}(\frac{\sigma \, Z_t}{\tau}),
\qquad Z_t \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1),
$$

* volatility $\sigma = 1.0$. Rounding keeps prices on the tick lattice.
* choose initial price $S_0 \in \mathbb{Z}+\frac{1}{2}\tau$
  * This can guarantee the spread between the best ask and best bid

### 1.4 Fill kernel (order execution)

Each posted quote is filled independently in one step with probability
that **decays exponentially** in distance from the mid (a discrete-time
analogue of the Avellaneda–Stoikov exponential intensity):

$$
\mathbb{P}(F_t^b = 1 \mid \delta_t^b) = e^{-\kappa \, \delta_t^b}, \qquad
\mathbb{P}(F_t^a = 1 \mid \delta_t^a) = e^{-\kappa \, \delta_t^a},
$$

with $\kappa = 1.5$. The fill indicators $F_t^b, F_t^a \in \{0, 1\}$ are
conditionally independent given $(\delta_t^b, \delta_t^a)$.

### 1.5 Inventory and cash updates

Conditional on the fills, inventory and cash evolve linearly:

$$
\begin{aligned}
q_{t+1} &= q_t + F_t^b - F_t^a, \\
c_{t+1} &= c_t \;-\; F_t^b \cdot P_t^{\text{bid}} \;+\; F_t^a \cdot P_t^{\text{ask}}.
\end{aligned}
$$

A bid fill increases inventory by 1 unit and debits the bid price from
cash; an ask fill decreases inventory by 1 unit and credits the ask price.

### 1.6 Terminal liquidation

At the final step $t = T - 1$ (with $T = 100$) the agent **liquidates**
all remaining inventory at the new mid:

$$
q_T \leftarrow 0, \qquad c_T \leftarrow c_{T-1} + S_T \cdot q_{T-1}.
$$

This enforces a clean episode boundary and guarantees a flat terminal
book.

### 1.7 Reward

The per-step reward is the **change in mark-to-market wealth** minus a
**quadratic inventory penalty**:

$$
r_t = \underbrace{(S_{t+1} q_{t+1} + c_{t+1}) - (S_t q_t + c_t)}_{\Delta\,\text{PV}_t}
\;-\; \lambda \, q_{t+1}^2,
$$

where $\lambda \geq 0$ is the **risk aversion / inventory-penalty
coefficient**. The sweep in §4 of the notebook varies $\lambda \in \{0.01,
0.1, 0.5, 1.0\}$.

By telescoping, the undiscounted episode return satisfies the closed-form
identity

$$
\sum_{t=0}^{T-1} r_t = (S_T q_T + c_T) - (S_0 q_0 + c_0) - \lambda \sum_{t=1}^{T} q_t^2,
$$

which is verified numerically in the §1.7 sanity cell of the notebook.

### 1.8 Observation

The agent does **not** see the raw state directly. Instead it receives a
4-channel normalized observation vector

$$
o_t = \left(
\frac{S_t - S_0}{\sigma \sqrt{T}},\;
\frac{q_t}{q_{\max}},\;
\frac{c_t}{c_{\text{scale}}},\;
\frac{t}{T}
\right) \in \mathbb{R}^4,
$$

with $q_{\max} = 10$ and $c_{\text{scale}} = 500$. Each channel is scaled
to roughly unit magnitude so that the MLP sees inputs on a common scale.

---

## 2. Trading Agent — Actor–Critic Network

The policy and value function share a single neural network
`ActorCritic`, parameterised by $\theta$.

### 2.1 Architecture

A 2-hidden-layer MLP trunk with $\tanh$ activations and hidden width
$H = 64$:

$$
h_1 = \tanh(W_1 o_t + b_1), \qquad
h_2 = \tanh(W_2 h_1 + b_2).
$$

The trunk branches into two heads:

- **Actor head** producing logits $\ell_t \in \mathbb{R}^{K^2}$ over the
  25 discrete actions:
  $$\ell_t = W_\pi h_2 + b_\pi.$$
- **Critic head** producing a scalar value estimate:
  $$V_\theta(o_t) = w_v^\top h_2 + b_v.$$

### 2.2 Policy distribution

The policy is a categorical distribution over the action set:

$$
\pi_\theta(a \mid o_t) = \mathrm{softmax}(\ell_t)_a
= \frac{\exp(\ell_t^{(a)})}{\sum_{a'} \exp(\ell_t^{(a')})}.
$$

Actions are sampled during rollout via `jax.random.categorical(key,
logits)`; at evaluation time the greedy action $\arg\max_a \ell_t^{(a)}$
is used.

### 2.3 Value function

$V_\theta(o_t)$ estimates the **expected return-to-go** under the current
policy,

$$
V_\theta(o_t) \approx \mathbb{E}_\pi\!\left[\sum_{k=0}^{T-t-1} \gamma^k r_{t+k} \,\Big|\, o_t \right],
$$

and is used as a baseline in the advantage estimator.

---

## 3. Policy Optimisation — PPO with GAE

The agent is trained with **Proximal Policy Optimization (PPO, Schulman
et al. 2017)** using **Generalized Advantage Estimation (GAE, Schulman
et al. 2016)**.

### 3.1 Trajectory collection

For each PPO iteration, $N_{\text{env}} = 16$ environments are rolled out
in parallel for $T = 100$ steps under the **current** policy
$\pi_{\theta_{\text{old}}}$, producing a batch of
$N_{\text{env}} \cdot T = 1600$ transitions

$$
\{(o_t^i, a_t^i, r_t^i, d_t^i, \log \pi_{\theta_{\text{old}}}(a_t^i \mid o_t^i), V_{\theta_{\text{old}}}(o_t^i))\}_{t,i}.
$$

### 3.2 Generalized Advantage Estimation

For each trajectory $i$, the TD residual is

$$
\delta_t = r_t + \gamma \, V_{\theta_{\text{old}}}(o_{t+1}) (1 - d_t) - V_{\theta_{\text{old}}}(o_t),
$$

and the GAE advantage is the exponentially weighted sum of residuals,
computed by a backward `lax.scan`:

$$
\hat{A}_t = \sum_{k=0}^{T-t-1} (\gamma \lambda_{\text{GAE}})^k \, \delta_{t+k},
\qquad
\hat{A}_t = \delta_t + \gamma \lambda_{\text{GAE}} (1 - d_t) \, \hat{A}_{t+1}.
$$

With $\gamma = 0.99$ and $\lambda_{\text{GAE}} = 0.95$. Because the
episode terminates cleanly at $T$, the bootstrap value at $t = T$ is
$\hat{V}_T = 0$. The corresponding return target is

$$
\hat{R}_t = \hat{A}_t + V_{\theta_{\text{old}}}(o_t).
$$

### 3.3 PPO loss

Let $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$
denote the importance ratio between the current and behaviour policies.
The total loss minimized at each gradient step is

$$
\mathcal{L}(\theta) = \mathcal{L}^{\text{PG}}(\theta)
\;+\; c_v \, \mathcal{L}^{\text{VF}}(\theta)
\;+\; c_e \, \mathcal{L}^{\text{ENT}}(\theta),
$$

with the three components defined below.

**(a) Clipped policy-gradient loss.**

$$
\mathcal{L}^{\text{PG}}(\theta) = -\,\mathbb{\hat{E}}_t \left[
\min\!\Big( r_t(\theta) \hat{A}_t,\;
\mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \, \hat{A}_t \Big)
\right],
$$

with clip range $\epsilon = 0.2$. The pessimistic $\min$ prevents the new
policy from moving too far from $\pi_{\theta_{\text{old}}}$ in directions
that improve the surrogate.

**(b) Value loss (MSE, no clipping).**

$$
\mathcal{L}^{\text{VF}}(\theta) = \mathbb{\hat{E}}_t \left[
\big(V_\theta(o_t) - \hat{R}_t\big)^2
\right], \qquad c_v = 0.5.
$$

**(c) Entropy bonus (encourages exploration).**

$$
\mathcal{L}^{\text{ENT}}(\theta) = -\,\mathbb{\hat{E}}_t \big[\, \mathcal{H}\!\left(\pi_\theta(\cdot \mid o_t)\right) \,\big],
\qquad
\mathcal{H}(\pi) = -\sum_a \pi(a) \log \pi(a),
\qquad c_e = 0.01.
$$

The negative sign converts the maximisation of entropy into a
minimisation problem consistent with the rest of $\mathcal{L}$.

### 3.4 Optimisation schedule

Each PPO iteration performs $E = 4$ **epochs** over the collected batch.
Each epoch shuffles the 1600 transitions and partitions them into $M = 4$
minibatches of 400. For each minibatch, parameters are updated via Adam
($\eta = 3 \times 10^{-4}$):

$$
\theta \leftarrow \mathrm{Adam}\big(\theta,\; \nabla_\theta \mathcal{L}(\theta)\big).
$$

The total number of outer PPO iterations is $N_{\text{update}} = 50$
(GPU) or $10$ (CPU).

### 3.5 Hyperparameter sweep

The full training function is `vmap`-ed twice — once over PRNG **seeds**
and once over the **inventory penalty** $\lambda$ — yielding a 2-D sweep
$\{(\text{seed}_m, \lambda_\ell)\}_{m, \ell}$ trained in parallel inside a
single `jit`-compiled call. This is the source of the headline GPU
speedup reported in §4.4 of the notebook.

---

## 4. Summary of Symbols and Default Values

| Symbol | Meaning | Value |
| --- | --- | --- |
| $T$ | episode length | $100$ |
| $K$ | quote-distance levels per side | $5$ |
| $\tau$ | tick size | $1.0$ |
| $\sigma$ | mid-price volatility | $1.0$ |
| $\kappa$ | fill-intensity decay | $1.5$ |
| $S_0$ | initial mid price | $100$ |
| $\lambda$ | inventory penalty (swept) | $\{0.01, 0.1, 0.5, 1.0\}$ |
| $q_{\max}, c_{\text{scale}}$ | observation normalisers | $10,\; 500$ |
| $H$ | MLP hidden width | $64$ |
| $\gamma$ | discount factor | $0.99$ |
| $\lambda_{\text{GAE}}$ | GAE smoothing | $0.95$ |
| $\epsilon$ | PPO clip range | $0.2$ |
| $c_v, c_e$ | value / entropy loss weights | $0.5,\; 0.01$ |
| $N_{\text{env}}$ | parallel envs per update | $16$ |
| $E, M$ | epochs, minibatches per update | $4,\; 4$ |
| $\eta$ | Adam learning rate | $3 \times 10^{-4}$ |
| $N_{\text{update}}$ | PPO outer iterations | $50$ (GPU) |
