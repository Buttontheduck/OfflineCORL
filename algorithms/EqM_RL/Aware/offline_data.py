"""D4RL data for the frozen-weights scripts (make_weights.py, fit_actor.py): the transitions in
exactly the order of d4rl.qlearning_dataset (as Aware/main.py and CORL's iql.py load them), the
same state normalization, and the episode of every transition, which qlearning_dataset drops."""

import d4rl  # noqa: F401  (registers the D4RL envs with gym)
import gym
import numpy as np


def load_d4rl(env_name: str):
    """-> env, dataset (qlearning_dataset), state mean, state std, episode id per transition."""
    env = gym.make(env_name)
    data = d4rl.qlearning_dataset(env)
    obs = data["observations"]
    mean, std = obs.mean(0), obs.std(0) + 1e-3  # compute_mean_std(eps=1e-3) in main.py and iql.py
    return env, data, mean, std, episode_ids(env, len(obs))


def episode_ids(env, n: int) -> np.ndarray:
    """Episode index of each qlearning_dataset transition. qlearning_dataset (terminate_on_end
    False, with timeouts) walks i = 0 .. N-2 and skips the steps flagged as timeouts; an episode
    ends after a timeout or a terminal step."""
    raw = env.get_dataset()
    timeouts, terminals = raw["timeouts"].astype(bool), raw["terminals"].astype(bool)
    N = len(timeouts)
    ep_raw = np.concatenate([[0], np.cumsum(timeouts | terminals)[:-1]])
    ep = ep_raw[: N - 1][~timeouts[: N - 1]]
    assert len(ep) == n, f"episode ids ({len(ep)}) don't match qlearning_dataset ({n})"
    return ep


def data_filter(name: str, ep: np.ndarray, rewards: np.ndarray):
    """Row mask for a named subset, plus a short report.
    'all': every transition. 'expert_half': the second half of the episodes, which in
    medium-expert-v2 is the expert policy's (the file holds 1,000 medium episodes, then 1,000
    expert episodes; the report shows both halves' returns to check that)."""
    n_ep = int(ep.max()) + 1
    returns = np.bincount(ep, weights=rewards, minlength=n_ep)
    if name == "all":
        return np.ones(len(ep), bool), dict(n_episodes=n_ep)
    if name == "expert_half":
        keep_ep = np.arange(n_ep) >= n_ep // 2
        rep = dict(n_episodes=n_ep, kept_episodes=int(keep_ep.sum()),
                   kept_return_min=float(returns[keep_ep].min()), kept_return_mean=float(returns[keep_ep].mean()),
                   dropped_return_max=float(returns[~keep_ep].max()),
                   dropped_return_mean=float(returns[~keep_ep].mean()))
        return keep_ep[ep], rep
    raise ValueError(f"data_filter must be 'all' or 'expert_half', got {name!r}")
