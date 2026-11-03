"""Sampler drift vs critic, for one trained Aware run (EqM_RL.md, Experiment 3, interpretation 8).

Re-evaluates the final checkpoint with many sampler settings; nothing is retrained, so the field
and the critic are the ones training produced with its own sampler (as trained: K = 20,
eta = 0.01, momentum 0.3).

A (rollouts): every (eta, K, momentum) of the grid, n_eps episodes each (seeds 69, 70, ..., so
   every setting starts from the same states), all settings side by side as one batch. Per
   episode: return, flip (body angle obs[1] beyond 2 rad), forward speed obs[8], the share of
   action dims the sampler puts outside the box, and the critic's Q(s_t, a_t) against the
   discounted return actually collected from s_t (first `q_window` steps). Every 100 steps the
   visited state is compared with the dataset: distance to the nearest dataset state, distance
   of the executed action to the actions of the k nearest dataset states (as a ratio to the same
   distance for dataset actions on dataset states), and, on medium-expert, the share of those
   neighbours that come from the expert half (which region of the data the policy is in).
B (dataset states): n_states random dataset states x n_samples samples, tracked over K for every
   (eta, momentum), from the same starting noise: field norm at the endpoint, movement per step at
   step K, per-state spread, share outside the box, distance to the nearest neighbour action (as
   above), and Q at the sampled action minus Q at the data action, by this run's critic and, with
   --awac, by a trained Gaussian AWAC critic (an independent judge). With --awac also the AWAC
   policy's mean action scored the same way.

Medium-expert: the first half of the transitions comes from the medium policy, the second half
from the expert (qlearning_dataset keeps the file's order: 1,000 medium episodes, then 1,000
expert episodes).

    python diagnostics/drift.py <Aware checkpoint dir> diagnostics/out/drift/<name>.json \
        [--awac <AWAC checkpoint dir>]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from common import EQM_RL, Policy, latest_checkpoint

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path, help="checkpoint folder of one Aware run (has config.yaml)")
ap.add_argument("out", type=Path, help="output JSON")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--awac", type=Path, help="checkpoint folder of a CORL AWAC run on the same dataset")
ap.add_argument("--etas", default="0.0025,0.005,0.01,0.02,0.04,0.08", help="step sizes eta")
ap.add_argument("--Ks_env", default="1,2,5,10,20,40,80,160,320,640,1280", help="K values in A")
ap.add_argument("--mus_env", default="0,0.3,0.6", help="momentum (look-ahead) values in A")
ap.add_argument("--n_eps", type=int, default=20, help="episodes per setting in A")
ap.add_argument("--q_window", type=int, default=500, help="steps of each episode used for Q vs return")
ap.add_argument("--visit_every", type=int, default=100, help="compare visited states with the data every n steps")
ap.add_argument("--flip_angle", type=float, default=2.0, help="|body angle| in rad that counts as a flip")
ap.add_argument("--Ks_act", default="1,2,5,10,20,40,80,160,320,640,1280,2560", help="K values in B")
ap.add_argument("--mus_act", default="0,0.3,0.6", help="momentum values in B")
ap.add_argument("--n_states", type=int, default=2048, help="dataset states in B")
ap.add_argument("--n_samples", type=int, default=32, help="samples per state in B")
ap.add_argument("--knn", type=int, default=32, help="nearest dataset states used as the local action support")
ap.add_argument("--window", type=int, default=20, help="time steps around a dataset state excluded from its neighbours")
ap.add_argument("--skip_env", action="store_true", help="skip A")
args = ap.parse_args()
t0 = time.time()
floats = lambda s: [float(v) for v in s.split(",")]  # noqa: E731
ints = lambda s: [int(v) for v in s.split(",")]  # noqa: E731
etas, mus_env, mus_act = floats(args.etas), floats(args.mus_env), floats(args.mus_act)
Ks_env, Ks_act = ints(args.Ks_env), ints(args.Ks_act)

pol = Policy(args.run_dir, args.checkpoint, load_critic=True)
dev, lo, hi, ad = pol.device, pol.low, pol.high, pol.action_dim
gamma = float(pol.cfg["gamma"])
res = pol.info()
res["args"] = {k: str(v) for k, v in vars(args).items()}

# ---------------------------------------------------------------- optional AWAC reference
awac_q = awac_mean = None
if args.awac:
    sys.path.insert(0, str(EQM_RL.parent / "offline"))
    import awac as corl_awac  # CORL's algorithms/offline/awac.py

    acfg = yaml.safe_load(open(args.awac / "config.yaml"))
    assert acfg["env_name"] == pol.cfg["env_name"], "AWAC run is on another dataset"
    sd = torch.load(latest_checkpoint(args.awac), map_location=dev)
    a_actor = corl_awac.Actor(pol.state_dim, ad, acfg["hidden_dim"]).to(dev).eval()
    a_actor.load_state_dict(sd["actor"])
    a_critics = []
    for key in ("critic_1", "critic_2"):
        c = corl_awac.Critic(pol.state_dim, ad, acfg["hidden_dim"]).to(dev).eval()
        c.load_state_dict(sd[key])
        a_critics.append(c)
    # awac.py normalizes states with the same formula on the same dataset, so pol.states() applies

    @torch.no_grad()
    def awac_q(s, a):
        return torch.min(a_critics[0](s, a), a_critics[1](s, a)).squeeze(-1)

    @torch.no_grad()
    def awac_mean(s):
        return a_actor._mlp(s)  # the Gaussian mean, which awac.py evaluates with

    res["awac_run"] = args.awac.name

# ---------------------------------------------------------------- dataset and neighbours
obs_all, act_all = pol.data["observations"], pol.data["actions"]
n = len(obs_all)
all_S, all_A = pol.states(obs_all), torch.tensor(act_all, dtype=torch.float32, device=dev)
has_src = "medium-expert" in pol.cfg["env_name"]
src_all = (torch.arange(n, device=dev) >= n // 2).float() if has_src else None


@torch.no_grad()
def knn(Sq, k, q_idx=None):
    """k nearest dataset states of the states Sq (normalized space): (distances, indices), each
    (len(Sq), k). With q_idx (dataset indices of Sq), the state's own +-window steps are skipped."""
    out_d, out_j = [], []
    for b in range(0, len(Sq), 4096):
        q = Sq[b:b + 4096]
        best_d = torch.full((len(q), k), float("inf"), device=dev)
        best_j = torch.zeros((len(q), k), dtype=torch.long, device=dev)
        for c in range(0, n, 25_000):
            d = torch.cdist(q, all_S[c:c + 25_000])
            if q_idx is not None:
                j = torch.arange(c, c + d.shape[1], device=dev)
                d[(q_idx[b:b + 4096, None] - j[None, :]).abs() <= args.window] = float("inf")
            dk, jk = d.topk(k, dim=1, largest=False)
            cd, cj = torch.cat([best_d, dk], 1), torch.cat([best_j, jk + c], 1)
            best_d, order = cd.topk(k, dim=1, largest=False)
            best_j = cj.gather(1, order)
        out_d.append(best_d)
        out_j.append(best_j)
    return torch.cat(out_d), torch.cat(out_j)


def nn_action_dist(a, nb_a):
    """distance from actions a (N, m, ad) to the nearest of the neighbour actions nb_a (N, k, ad):
    (min distance (N, m), argmin (N, m))"""
    return (a[:, :, None, :] - nb_a[:, None, :, :]).norm(dim=-1).min(-1)


# Reference: dataset states and their own actions
N, M, k = args.n_states, args.n_samples, args.knn
idx = np.sort(np.random.default_rng(0).choice(n, N, replace=False))
idx_t = torch.tensor(idx, device=dev)
S, A = all_S[idx_t], all_A[idx_t]
nb_d, nb_j = knn(S, k, idx_t)
NB_A = all_A[nb_j]
with torch.no_grad():
    d_data = nn_action_dist(A[:, None, :], NB_A)[0][:, 0]
    q_data = pol.q(S, A)
    f_data = pol.model(A, S).norm(dim=-1).mean().item()
res["data"] = dict(nn_state_dist=nb_d[:, 0].mean().item(), nn_state_dist_median=nb_d[:, 0].median().item(),
                   nn_dist=d_data.mean().item(), nn_dist_median=d_data.median().item(),
                   at_bounds=(A.abs() >= 0.99 * hi).float().mean().item(),
                   q=q_data.mean().item(), f_norm=f_data)
if awac_q is not None:
    aq_data = awac_q(S, A)
    res["data"]["awac_q"] = aq_data.mean().item()
if has_src:
    src_state, NB_src = src_all[idx_t], src_all[nb_j]
    res["data"]["state_expert_share"] = src_state.mean().item()
    res["data"]["nb_expert_share_by_state"] = dict(
        medium=NB_src[src_state == 0].mean().item(), expert=NB_src[src_state == 1].mean().item())

# ---------------------------------------------------------------- A: rollouts
def rollout(act_fn, qfun, E, n_eps):
    """E episodes side by side; episode i uses seed 69 + i % n_eps. act_fn(states) returns
    unclipped actions (E, ad), qfun(states, actions) the critic's value. Returns per-episode
    arrays (length E) for the fields of a res['rollouts'] entry."""
    envs = [pol.make_env() for _ in range(E)]
    obs = np.stack([pol.reset(e, 69 + i % n_eps) for i, e in enumerate(envs)]).astype(np.float64)
    done = np.zeros(E, bool)
    flip_step = np.full(E, -1)
    R, Q, OUT, ALIVE, SPEED = [], [], [], [], []
    VS, VA, VALIVE = [], [], []  # visited states / executed actions every visit_every steps
    t = 0
    while not done.all():
        S_t = pol.states(obs)
        with torch.no_grad():
            x = act_fn(S_t)
            a = x.clamp(lo, hi)
            Q.append(qfun(S_t, a).cpu().numpy())
            OUT.append(((x < lo) | (x > hi)).float().mean(1).cpu().numpy())
        ALIVE.append(~done)
        if t % args.visit_every == 0:
            VS.append(S_t)
            VA.append(a)
            VALIVE.append(torch.tensor(~done, device=dev))
        a_np, r = a.cpu().numpy(), np.zeros(E)
        for i in np.flatnonzero(~done):
            obs[i], r[i], done[i] = pol.step(envs[i], a_np[i])
        flipped = (np.abs(obs[:, 1]) > args.flip_angle) & (flip_step < 0) & ALIVE[-1]
        flip_step[flipped] = t
        SPEED.append(obs[:, 8].copy())
        R.append(r)
        t += 1
    del envs
    R, Q, OUT, ALIVE, SPEED = map(np.array, (R, Q, OUT, ALIVE, SPEED))  # (T, E)
    G = np.zeros_like(R)
    for t in reversed(range(len(R))):
        G[t] = R[t] + gamma * (G[t + 1] if t + 1 < len(R) else 0.0)
    W = min(args.q_window, len(R))
    aw = ALIVE[:W]
    out = dict(returns=R.sum(0), flip_step=flip_step,
               speed=(SPEED * ALIVE).sum(0) / ALIVE.sum(0), outside=(OUT * ALIVE).sum(0) / ALIVE.sum(0),
               q=(Q[:W] * aw).sum(0) / aw.sum(0), g=(G[:W] * aw).sum(0) / aw.sum(0))

    # Visited states vs the dataset (state distance and action distance as ratios to dataset states)
    T_v = len(VS)
    Sv, Av, alive_v = torch.cat(VS), torch.cat(VA), torch.cat(VALIVE).float()  # rows: (time point, env)
    vd, vj = knn(Sv, k)
    with torch.no_grad():
        v_act = nn_action_dist(Av[:, None, :], all_A[vj])[0][:, 0] / d_data.mean()

    def per_env(v):
        return ((v * alive_v).view(T_v, E).sum(0) / alive_v.view(T_v, E).sum(0)).cpu().numpy()

    out["state_dist"] = per_env(vd[:, 0] / nb_d[:, 0].mean())
    out["act_dist"] = per_env(v_act)
    if has_src:
        out["expert_region"] = per_env(src_all[vj].mean(1))
    return out


def entry_of(out, sl, **extra):
    return dict(extra, **{key: v[sl].tolist() for key, v in out.items()})


if not args.skip_env:
    settings = list(itertools.product(etas, Ks_env, mus_env))
    n_eps = args.n_eps
    rep = lambda vals, dt: torch.tensor([v for v in vals for _ in range(n_eps)], dtype=dt, device=dev)  # noqa: E731
    K_row = rep([s[1] for s in settings], torch.long)
    eta_row = rep([s[0] for s in settings], torch.float32)
    mu_row = rep([s[2] for s in settings], torch.float32)
    torch.manual_seed(0)
    out = rollout(lambda s: pol.sample_rows(s, K_row, eta_row, mu_row), pol.q, len(settings) * n_eps, n_eps)
    res["rollouts"] = [entry_of(out, slice(j * n_eps, (j + 1) * n_eps), eta=eta, K=K, mu=mu)
                  for j, (eta, K, mu) in enumerate(settings)]
    if awac_mean is not None:
        # The Gaussian AWAC policy (mean action) from the same start states, scored by its own critic
        out = rollout(awac_mean, awac_q, n_eps, n_eps)
        res["awac_rollouts"] = entry_of(out, slice(0, n_eps))
    res["seconds_env"] = time.time() - t0

# ---------------------------------------------------------------- B: dataset states
with torch.no_grad():

    def score(x, S_, rows_per_state):
        """metrics of actions x (N * rows_per_state, ad) on states S_ (same rows)"""
        xc = x.clamp(lo, hi)
        dmin, jmin = nn_action_dist(xc.view(N, rows_per_state, ad), NB_A)
        q = pol.q(S_, xc)
        out = dict(outside=((x < lo) | (x > hi)).float().mean().item(),
                   nn_dist=dmin.mean().item(), nn_ratio=(dmin.mean() / d_data.mean()).item(),
                   dq=(q - q_data.repeat_interleave(rows_per_state)).mean().item())
        if awac_q is not None:
            out["awac_dq"] = (awac_q(S_, xc) - aq_data.repeat_interleave(rows_per_state)).mean().item()
        if has_src:
            out["expert_share"] = NB_src.gather(1, jmin).mean().item()
        return out

    if awac_mean is not None:
        res["awac_policy"] = score(awac_mean(S), S, 1)

    S_rep = S.repeat_interleave(M, 0)
    torch.manual_seed(1)
    x0 = torch.randn(N * M, ad, device=dev)
    snaps = set(Ks_act) | {K - 1 for K in Ks_act if K > 1}
    res["act"] = []
    for eta, mu in itertools.product(etas, mus_act):
        _, sn = pol.sample(S_rep, max(Ks_act), snaps=snaps, eta=eta, mu=mu, x0=x0)
        for K in Ks_act:
            x, prev = sn[K], sn.get(K - 1, x0)
            entry = dict(eta=eta, mu=mu, K=K,
                         f_norm=pol.model(x, S_rep).norm(dim=-1).mean().item(),
                         move=(x - prev).norm(dim=-1).mean().item(),
                         spread=x.clamp(lo, hi).view(N, M, ad).std(1).mean().item())
            entry.update(score(x, S_rep, M))
            res["act"].append(entry)
        del sn

res["seconds"] = time.time() - t0
res["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res))
print("done", res["run"], f"{res['seconds']:.0f}s")
