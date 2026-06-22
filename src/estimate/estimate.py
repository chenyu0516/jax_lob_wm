import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed when batch-saving figures
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture


def get_event_condition(df: pd.DataFrame):
    return (
        [
            (df['action'] == action) & (df['level'] == n) & (df['side'] == s)
            for action in ['A', 'C']
            for n in [1, 2]
            for s in ['ask', 'bid']
        ]
        +
        [
            (df['action'] == action) & (df['side'] == s)
            for action in ['T', 'E']
            for s in ['ask', 'bid']
        ]
    )


def index_to_spread_imb(i):
    spread = (i // 21) + 1      # 0..4 -> 1..5
    imb = (i % 21) - 10         # 0..20 -> -10..10
    return spread, imb


def estimate_prob(in_dir: str, out_dir: str, fig_dir: str = None):
    # if no fig_dir given, default to a sibling 'figures' folder next to out_dir
    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(out_dir.rstrip('/')), 'figures')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    states_df = pd.read_parquet(f'{in_dir}/states.parquet')
    event_df = pd.read_parquet(f'{in_dir}/event.parquet')

    state_conditions = [
        (states_df['spread'] == s) & (states_df['imb'] == imb)
        for s in [1, 2, 3, 4, 5]
        for imb in range(-10, 11)
    ]

    n_states = len(state_conditions)          # 5 * 21 = 105
    n_events = 12                             # 8 with level + 4 without
    event_p = np.zeros((n_states, n_events), dtype=np.int64)

    init_state = []
    models = {}  # nested dict: models[i] = {...} of all fitted params for state i

    for i, cond in enumerate(state_conditions):
        spread, imb = index_to_spread_imb(i)
        current_states = states_df[cond]

        if current_states.empty:
            continue

        # select the initials
        rng = np.random.default_rng(seed=42)
        size = min(100, len(current_states))
        init_idx = rng.choice(current_states.index, size=size, replace=False)
        init_state.append(states_df.loc[init_idx])

        ts_con = current_states['ts']
        current_state_events = event_df[event_df['ts_con'].isin(ts_con)]

        state_key = f'spread{spread}_imb{imb}'
        models[state_key] = {'spread': int(spread), 'imb': int(imb), 'events': {}}

        for j, e in enumerate(get_event_condition(current_state_events)):
            cse = current_state_events[e]
            event_p[i, j] = cse.shape[0]

            event_models = {}

            # ---- fit log_dt probability (BGMM) ----
            dt = cse['ts_hap'].diff()
            log_dt = np.log(dt[dt > 0])
            log_dt = log_dt[np.isfinite(log_dt) & (log_dt < np.log(59400 * 1e9))]
            t_params = find_distr_t(log_dt, i, j, fig_dir)
            if t_params is not None:
                event_models['log_dt'] = t_params

            # ---- fit log event volume probability ----
            vol_params = find_distr_vol(cse['size_delta'].values, i, j, fig_dir)
            if vol_params is not None:
                event_models['log_size_delta'] = vol_params

            models[state_key]['events'][j] = event_models

        # ---- fit log deep_vol prob (levels 3 and 4, both sides via +/-k) ----
        deep_parts = []
        for k in [3, 4]:
            for col in (k, -k):
                if col in current_states.columns:
                    deep_parts.append(current_states[col])
        if deep_parts:
            deep_vol = pd.concat(deep_parts)
            deep_params = find_distr_deep_vol(deep_vol.values, i, fig_dir)
            if deep_params is not None:
                models[state_key]['deep_vol'] = deep_params

    # save everything
    np.save(os.path.join(out_dir, 'event_counts.npy'), event_p)
    with open(os.path.join(out_dir, 'fit_models.json'), 'w') as f:
        json.dump(models, f, indent=2)

    if init_state:
        pd.concat(init_state, ignore_index=True).to_parquet(
            os.path.join(out_dir, 'init.parquet')
        )


def find_distr_t(log_dt, i, j, fig_dir):
    """BGMM on log inter-event time. Returns JSON-serializable params."""
    if len(log_dt) < 20:
        return None

    spread, imb = index_to_spread_imb(i)
    X = log_dt.values.reshape(-1, 1)

    bgmm = BayesianGaussianMixture(
        n_components=20,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-2,
        random_state=42, max_iter=1000,
    ).fit(X)

    mask = bgmm.weights_ > 1e-3
    means = bgmm.means_.ravel()
    sds = np.sqrt(bgmm.covariances_.ravel())
    weights = bgmm.weights_.ravel()

    # plot
    log_plot = np.linspace(log_dt.min(), log_dt.max(), 1000)
    pdf = np.exp(bgmm.score_samples(log_plot.reshape(-1, 1)))

    plt.figure(figsize=(8, 4))
    plt.hist(log_dt, bins=50, density=True, alpha=0.5, label='Data')
    plt.plot(log_plot, pdf, 'r-', lw=2, label=f'BGMM fit (n={mask.sum()})')
    for idx in np.where(mask)[0]:
        plt.plot(log_plot, weights[idx] * norm.pdf(log_plot, means[idx], sds[idx]),
                 '--', lw=1, alpha=0.7)
    plt.legend()
    plt.title(f'log_dt BGMM (spread={spread}, imb={imb}, event={j})')
    plt.savefig(os.path.join(fig_dir, f'log_dt_s{spread}_imb{imb}_e{j}.png'))
    plt.close()

    # keep only effective components; renormalize their weights
    w = weights[mask]
    w = w / w.sum()
    return {
        'model': 'bgmm',
        'space': 'log',
        'means': means[mask].tolist(),
        'sigmas': sds[mask].tolist(),
        'weights': w.tolist(),
    }


def find_distr_vol(size_delta, i, j, fig_dir):
    """Discrete spikes + continuous lognormal on log10(size_delta)."""
    spread, imb = index_to_spread_imb(i)
    x = np.log10(size_delta[np.isfinite(size_delta) & (size_delta > 0)])
    if len(x) < 20:
        return None

    # detect spikes: exact values holding > 1% of mass
    vals, counts = np.unique(np.round(x, 4), return_counts=True)
    freq = counts / counts.sum()
    spike_vals = vals[freq > 0.01]

    is_spike = np.isin(np.round(x, 4), spike_vals)
    x_cont = x[~is_spike]
    p_spike = float(is_spike.mean())

    sv, sc = np.unique(np.round(x[is_spike], 4), return_counts=True)
    spike_mass = {str(float(k)): float(v / len(x)) for k, v in zip(sv, sc)}

    # continuous part: robust lognormal (median + IQR-based sigma)
    cont = None
    if len(x_cont) >= 20:
        mu = float(np.median(x_cont))
        sigma = float((np.percentile(x_cont, 84.1) - np.percentile(x_cont, 15.9)) / 2)
        cont = {'model': 'lognormal_base10', 'space': 'log10',
                'mu': mu, 'sigma': sigma, 'weight': float(1 - p_spike)}

    # plot
    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=80, density=True, alpha=0.4, label='Data (log10)')
    if cont is not None:
        grid = np.linspace(x.min(), x.max(), 1000)
        plt.plot(grid, (1 - p_spike) * norm.pdf(grid, mu, sigma),
                 'r-', lw=2, label='Continuous (scaled)')
    if spike_mass:
        plt.stem(list(map(float, spike_mass.keys())), list(spike_mass.values()),
                 linefmt='g-', markerfmt='go', basefmt=' ', label='Spike mass')
    plt.legend()
    plt.title(f'size_delta (spread={spread}, imb={imb}, event={j})')
    plt.savefig(os.path.join(fig_dir, f'vol_s{spread}_imb{imb}_e{j}.png'))
    plt.close()

    return {'model': 'discrete+continuous', 'space': 'log10',
            'p_spike': p_spike, 'spike_mass': spike_mass, 'continuous': cont}


def find_distr_deep_vol(deep_vol, i, fig_dir):
    """Robust lognormal on log(deep_vol)."""
    spread, imb = index_to_spread_imb(i)
    x = deep_vol[np.isfinite(deep_vol) & (deep_vol > 0)]
    if len(x) < 20:
        return None

    lx = np.log(x)
    mu = float(np.median(lx))
    sigma = float((np.percentile(lx, 84.1) - np.percentile(lx, 15.9)) / 2)

    grid = np.linspace(lx.min(), lx.max(), 500)
    plt.figure(figsize=(8, 4))
    plt.hist(lx, bins=50, density=True, alpha=0.5, label='Data (log)')
    plt.plot(grid, norm.pdf(grid, mu, sigma), 'r-', lw=2, label='Lognormal (robust)')
    plt.legend()
    plt.title(f'deep_vol (spread={spread}, imb={imb})')
    plt.savefig(os.path.join(fig_dir, f'deepvol_s{spread}_imb{imb}.png'))
    plt.close()

    return {'model': 'lognormal', 'space': 'log', 'mu': mu, 'sigma': sigma}