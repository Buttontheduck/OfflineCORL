"""Two-group check (EqM_RL.md, "OGBench check: do the actor's samples reach both kinds of actions in the data ..."),
for OGBench and D4RL, for EqM-BC actors and, as a comparison, Gaussian-BC actors trained on the same data.
Evaluation only: no training, no episodes, no wandb.

Questions:
  1. At states where the dataset has two different kinds of actions, does the actor produce both kinds?
  2. EqM only: after K sampler steps the samples keep moving. In which direction?

Words (the same as in EqM_RL.md):
  similar states     the 128 dataset states closest to a chosen state (normalized state numbers), from other
                     episodes only, at most --per_ep per episode. 'plain' = Maze2D's rule (own episode allowed).
  data actions       the 128 actions taken in those similar states.
  group A / B        the data actions split in two with 2-means; centre = average action of the group.
  two-group state    smaller group >= 20 % of the 128 actions, and centre distance > --sep times the spread inside
                     the groups, measured along the line from centre A to centre B. (Maze2D compared with the spread
                     averaged over all action numbers; in 8-D and 21-D that rule calls most single normal
                     distributions 'two groups'. Simulation, 128 points: --sep 4 gives 0 % false 'two groups' for
                     single normal distributions in 8-D and 21-D, and finds 83-100 % of two normal distributions
                     4-6 std apart.)
  both groups reached  >= 10 % of the state's samples closer to centre A, and >= 10 % closer to centre B.
  random points      256 points from a round normal distribution with the centre and the average std of the
                     actor's samples. Ignores the data; shows what width alone reaches.
  wrong split        |share of samples in B - share of data actions in B|.
  in the middle      share of samples between 1/3 and 2/3 of the way from centre A to centre B.
  width              std of a state's samples, averaged over the action numbers.
  distance to data   sample -> closest data action (one half of the 128, alternate rank), / sqrt(action_dim).
                     The data's own value (half -> other half) is the reference.
  data missed        data action (same half) -> closest of 64 samples, / sqrt(action_dim).
  |f|                length of the EqM model output at the sample; 0 = the sampler no longer moves it.
  along the data     EqM only: share of the movement between two K values (squared length) that lies in the
                     state's main data directions (top principal directions of its 128 data actions holding 90 %
                     of their variance). Random directions give k / action_dim.

Actors: <prefix>_env_name<env>_actor<kind>_seed<seed>-<env>-*/<checkpoint>. Defaults: OGBench -> Exp14_BC, eqm;
D4RL -> Exp10_BC, eqm and gaussian. EqM: 256 samples per state from one sampler run (exactly Actor._implicit),
snapshots at every K. Gaussian: 256 draws from N(mean(s), std) (its state-independent std), plus its mean action
(the action it executes). Samples are clipped to the action box for all measures except |f| and the movement.

Output: diagnostics/out/two_groups/<env>/ results.json, summary.md, reference.npz, samples_<kind>_seed<s>.npz.
Run from algorithms/EqM_RL/:
  OGBench:  /workspace/miniconda3/envs/ogbench/bin/python diagnostics/ogbench_two_groups.py --env humanoidmaze-medium-navigate-singletask-v0
  D4RL:     LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin /workspace/miniconda3/envs/hawk/bin/python diagnostics/ogbench_two_groups.py --env antmaze-umaze-v2
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Aware"))
import ogbench_data  # noqa: E402
from gaussian_actor import GaussianActor  # noqa: E402
from utils.networks import ConditionalMLP  # noqa: E402

KS = [5, 10, 15, 20, 25, 30, 50, 100, 1000]
MOVES = [(5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 1000)]
CKPT_ROOT = "/workspace/Trained_Models/checkpoints"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="humanoidmaze-medium-navigate-singletask-v0")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--kinds", default=None, help="eqm,gaussian (default: eqm on OGBench, both on D4RL)")
    p.add_argument("--bc_prefix", default=None, help="default: Exp14_BC on OGBench, Exp10_BC on D4RL")
    p.add_argument("--checkpoint", default="checkpoint_500000.pt")
    p.add_argument("--n_states", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--n_nn", type=int, default=128)
    p.add_argument("--per_ep", type=int, default=4)
    p.add_argument("--sep", type=float, default=4.0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ogb = ogbench_data.is_ogbench(a.env)
    a.kinds = a.kinds or ("eqm" if ogb else "eqm,gaussian")
    a.bc_prefix = a.bc_prefix or ("Exp14_BC" if ogb else "Exp10_BC")
    return a


def find_actor(a, kind, seed):
    pat = os.path.join(CKPT_ROOT, f"{a.bc_prefix}_env_name{a.env}_actor{kind}_seed{seed}-{a.env}-*")
    dirs = [d for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, a.checkpoint))]
    assert len(dirs) == 1, f"expected one run with {a.checkpoint} for {pat}, found {dirs}"
    cfg = yaml.safe_load(open(os.path.join(dirs[0], "config.yaml")))
    assert cfg["env_name"] == a.env and cfg["seed"] == seed and cfg["actor"]["kind"] == kind, dirs[0]
    return dirs[0], cfg


@torch.no_grad()
def neighbours(S, ep, q_idx, n_nn, per_ep, pool=32768, chunk=200_000):
    """Nearest dataset states of each query in normalized state space -> (visits idx, plain idx, radii, short)."""
    Q = S[q_idx]
    best_d = best_i = None
    for a in range(0, len(S), chunk):
        d = torch.cdist(Q, S[a:a + chunk])
        dk, ik = d.topk(min(pool, d.shape[1]), largest=False)
        ik = ik + a
        if best_d is not None:
            dk, ik = torch.cat([best_d, dk], 1), torch.cat([best_i, ik], 1)
        o = dk.topk(min(pool, dk.shape[1]), largest=False)
        best_d, best_i = o.values, ik.gather(1, o.indices)
    best_d, best_i = best_d.cpu().numpy(), best_i.cpu().numpy()
    vis, plain, r_vis, r_plain, short = [], [], [], [], 0
    for q in range(len(q_idx)):
        plain.append(best_i[q, :n_nn]); r_plain.append(best_d[q, n_nn - 1])
        own, cnt, pick, rad = ep[q_idx[q]], {}, [], 0.0
        for d, i in zip(best_d[q], best_i[q]):
            e = ep[i]
            if e == own or cnt.get(e, 0) >= per_ep:
                continue
            cnt[e] = cnt.get(e, 0) + 1
            pick.append(i); rad = d
            if len(pick) == n_nn:
                break
        if len(pick) < n_nn:
            short += 1
            pick += [pick[-1]] * (n_nn - len(pick))
        vis.append(np.array(pick)); r_vis.append(rad)
    return np.stack(vis), np.stack(plain), np.array(r_vis), np.array(r_plain), short


def two_means(x, it=30):
    m = x.mean(0)
    c1 = x[np.argmax(((x - m) ** 2).sum(1))]
    c2 = x[np.argmax(((x - c1) ** 2).sum(1))]
    cen = np.stack([c1, c2])
    for _ in range(it):
        lab = ((x[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        cen = np.stack([x[lab == k].mean(0) if (lab == k).any() else cen[k] for k in (0, 1)])
    return cen, lab


def group_test(D, sep):
    """-> (is two-group, centres, share of group B, separation along the A-B line, data's in-the-middle share)."""
    cen, lab = two_means(D)
    share = lab.mean()
    gap = np.linalg.norm(cen[1] - cen[0])
    if gap < 1e-9 or share in (0.0, 1.0):
        return False, cen, share, 0.0, 0.0
    u = (cen[1] - cen[0]) / gap
    p = (D - cen[0]) @ u
    mu = np.where(lab == 1, p[lab == 1].mean(), p[lab == 0].mean())
    ratio = gap / max(np.sqrt(np.mean((p - mu) ** 2)), 1e-12)
    t = p / gap
    return (min(share, 1 - share) >= 0.2 and ratio > sep), cen, share, ratio, float(((t > 1 / 3) & (t < 2 / 3)).mean())


def position(X, cen):
    """Position of each row of X on the line from centre A (0) to centre B (1)."""
    gap = np.linalg.norm(cen[1] - cen[0])
    return (X - cen[0]) @ ((cen[1] - cen[0]) / gap) / gap


def in_b(X, cen):
    return (((X - cen[1]) ** 2).sum(-1) < ((X - cen[0]) ** 2).sum(-1))


def coverage(X, groups, rng, low, high):
    """Over the two-group states: both groups reached (actor, random points), wrong split, in the middle."""
    if not groups:
        return dict(covered=None, null=None, split_error=None, between=None, n_covered=0, n_null=0, n_states=0)
    cov, cov0, err, btw = [], [], [], []
    for i, cen, share in groups:
        x = X[i]
        p = in_b(x, cen).mean()
        g = np.clip(x.mean(0) + x.std(0).mean() * rng.standard_normal(x.shape), low, high)
        p0 = in_b(g, cen).mean()
        t = position(x, cen)
        cov.append(min(p, 1 - p) >= 0.1); cov0.append(min(p0, 1 - p0) >= 0.1)
        err.append(abs(p - share)); btw.append(((t > 1 / 3) & (t < 2 / 3)).mean())
    return dict(covered=100 * float(np.mean(cov)), null=100 * float(np.mean(cov0)), split_error=float(np.mean(err)),
                between=float(np.mean(btw)), n_covered=int(np.sum(cov)), n_null=int(np.sum(cov0)), n_states=len(groups))


def measure(Xc, half_b, groups_vis, groups_plain, rng, low, high):
    """All measures for one set of clipped samples Xc (n, m, d) on the GPU."""
    d = Xc.shape[-1]
    x64 = Xc[:, :64]
    Xn = Xc.cpu().numpy()
    return dict(spread=Xc.std(1).mean().item(),
                precision=torch.cdist(x64, half_b).min(-1).values.mean().item() / d ** 0.5,
                recall=torch.cdist(half_b, x64).min(-1).values.mean().item() / d ** 0.5,
                at_bound=100 * (Xc.abs() > 0.99).float().mean().item(),
                visits=coverage(Xn, groups_vis, rng, low, high), plain=coverage(Xn, groups_plain, rng, low, high))


@torch.no_grad()
def run_eqm(a, seed, s, half_b, groups_vis, groups_plain, subs, low, high, dev, out):
    run, cfg = find_actor(a, "eqm", seed)
    n, m, action_dim = len(s), a.n_samples, half_b.shape[-1]
    model = ConditionalMLP(s.shape[1] + action_dim, action_dim, cfg["hidden_dim"], cfg["num_hidden_layers"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, a.checkpoint), map_location=dev)["model"])
    model.eval().requires_grad_(False)
    act = cfg["actor"]
    assert act["opt_type"] == "nag", act
    eta, mu = float(act["step_size"]), float(act["moment"])
    print(f"eqm seed {seed}: {run}/{a.checkpoint} (trained K = {act['num_step']}, eta {eta}, momentum {mu})", flush=True)
    st = s.repeat_interleave(m, 0)
    torch.manual_seed(0)                                          # the same starting noise for every actor
    x = torch.randn(n * m, action_dim, device=dev)
    mom = torch.zeros_like(x)
    snaps, fnorm = {}, {}
    for k in range(1, max(KS) + 1):                               # exactly Actor._descend (nag)
        mom = model(x - eta * mom * mu, st)
        x = x - eta * mom
        if k in KS:
            snaps[k] = x.view(n, m, action_dim).clone()
            fnorm[k] = model(x, st).norm(dim=1).mean().item()
    rng = np.random.default_rng(seed)
    res = dict(kind="eqm", seed=seed, run=run, trained_K=act["num_step"], per_K={}, moves={})
    for K in KS:
        res["per_K"][K] = dict(measure(snaps[K].clamp(low, high), half_b, groups_vis, groups_plain, rng, low, high),
                               f_norm=fnorm[K])
    for ka, kb in MOVES:
        dx = (snaps[kb] - snaps[ka]).cpu().numpy()
        row = dict(rms=float(np.sqrt((dx ** 2).sum(-1).mean() / action_dim)))
        for name, sub in subs.items():
            share = [((dx[i] @ sub[i]) ** 2).sum() / max((dx[i] ** 2).sum(), 1e-12) for i in range(n)]
            row[name] = dict(share=float(np.mean(share)), random=float(np.mean([P.shape[1] / action_dim for P in sub])))
        res["moves"][f"{ka}->{kb}"] = row
    np.savez_compressed(out / f"samples_eqm_seed{seed}.npz", **{f"K{K}": snaps[K].cpu().numpy().astype(np.float16) for K in KS})
    return res


@torch.no_grad()
def run_gaussian(a, seed, s, half_b, groups_vis, groups_plain, low, high, dev, out):
    run, cfg = find_actor(a, "gaussian", seed)
    n, m, action_dim = len(s), a.n_samples, half_b.shape[-1]
    model = GaussianActor(s.shape[1], action_dim, cfg["hidden_dim"], cfg["num_hidden_layers"],
                          cfg["min_log_std"], cfg["max_log_std"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, a.checkpoint), map_location=dev)["actor"])
    model.eval().requires_grad_(False)
    std = model._log_std.clamp(cfg["min_log_std"], cfg["max_log_std"]).exp()
    mean_a = model._mlp(s)                                         # (n, d): the action it executes (clipped)
    torch.manual_seed(0)
    X = mean_a[:, None] + std * torch.randn(n, m, action_dim, device=dev)
    print(f"gaussian seed {seed}: {run}/{a.checkpoint} (std per action number {np.round(std.tolist(), 3).tolist()})",
          flush=True)
    rng = np.random.default_rng(seed)
    res = dict(kind="gaussian", seed=seed, run=run, std=std.tolist(),
               samples=measure(X.clamp(low, high), half_b, groups_vis, groups_plain, rng, low, high))
    mc = mean_a.clamp(low, high)
    mn = mc.cpu().numpy()
    t = [float(position(mn[i:i + 1], cen)[0]) for i, cen, _ in groups_vis]
    res["mean_action"] = dict(
        precision=torch.cdist(mc[:, None], half_b).min(-1).values.mean().item() / action_dim ** 0.5,
        middle=100 * float(np.mean([1 / 3 < x < 2 / 3 for x in t])) if t else None,
        n_middle=int(np.sum([1 / 3 < x < 2 / 3 for x in t])), n_states=len(t))
    np.savez_compressed(out / f"samples_gaussian_seed{seed}.npz", samples=X.cpu().numpy().astype(np.float16),
                        mean_action=mean_a.cpu().numpy())
    return res


def subspace(D, frac=0.9):
    """Orthonormal basis of the top principal directions of D holding frac of its variance."""
    Dc = D - D.mean(0)
    _, sv, vt = np.linalg.svd(Dc, full_matrices=False)
    var = sv ** 2 / max((sv ** 2).sum(), 1e-12)
    return vt[: int(np.searchsorted(np.cumsum(var), frac) + 1)].T


def fmt(vals, digits=1, pm=True):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    m, sd = float(np.mean(vals)), float(np.std(vals))
    return f"{m:.{digits}f} ± {sd:.{digits}f}" if (pm and len(vals) > 1) else f"{m:.{digits}f}"


def counts(vals):
    return ", ".join(str(v) for v in vals)


def summary(a, info, R):
    n_g = info["_n_two_group"]
    L = [f"# Two-group check: `{a.env}` ({a.bc_prefix} actors, seeds {a.seeds})", "",
         "Evaluation only. Script and word definitions: `diagnostics/ogbench_two_groups.py` and EqM_RL.md. "
         f"{a.n_states} dataset states (numpy seed 0), {a.n_samples} samples per state. Values: average over the seeds "
         "± standard deviation between them. Counts: per seed.", "", "## The data", ""]
    L += [f"- **{k}:** {v}" for k, v in info.items() if not k.startswith("_")]
    L += ["", f"## Both kinds of actions? ({n_g} two-group states; width and distances over all {a.n_states} states)", "",
          "| actor | both groups reached (%) | states (per seed) | random points (%) | states (per seed) | wrong split | "
          "in the middle | width | distance to data | data missed | at ±0.99 (%) | ‖f‖ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, ms, f=None):
        v = [x["visits"] for x in ms]
        return (f"| {name} | {fmt([x['covered'] for x in v], 0)} | {counts([x['n_covered'] for x in v])} | "
                f"{fmt([x['null'] for x in v], 0)} | {counts([x['n_null'] for x in v])} | "
                f"{fmt([x['split_error'] for x in v], 2)} | {fmt([x['between'] for x in v], 2)} | "
                f"{fmt([x['spread'] for x in ms], 3)} | {fmt([x['precision'] for x in ms], 3)} | "
                f"{fmt([x['recall'] for x in ms], 3)} | {fmt([x['at_bound'] for x in ms], 1)} | {f if f else '–'} |")
    if "eqm" in R:
        for K in KS:
            ms = [r["per_K"][K] for r in R["eqm"]]
            L.append(row(f"EqM, K = {K}", ms, fmt([x["f_norm"] for x in ms], 2)))
    if "gaussian" in R:
        L.append(row("Gaussian, samples", [r["samples"] for r in R["gaussian"]]))
        mm = [r["mean_action"] for r in R["gaussian"]]
        L.append(f"| Gaussian, mean action | – | – | – | – | – | {fmt([x['middle'] for x in mm], 0, False)} % of states "
                 f"({counts([x['n_middle'] for x in mm])}) | – | {fmt([x['precision'] for x in mm], 3)} | – | – | – |")
    L.append(f"| the data | – | – | – | – | – | {info['_data_between']:.2f} | {info['_data_width']:.3f} | "
             f"{info['_data_precision']:.3f} (half vs half) | – | – | – |")
    L += ["", "For the Gaussian mean action (one action per state), 'in the middle' is the share of two-group states where "
          "that action lies between 1/3 and 2/3 of the way from centre A to centre B.", "",
          "## Same, with Maze2D's way of choosing similar states (own episode allowed)", "",
          "| actor | both groups reached (%) | random points (%) | wrong split | in the middle |", "|---|---|---|---|---|"]

    def row2(name, ms):
        v = [x["plain"] for x in ms]
        return (f"| {name} | {fmt([x['covered'] for x in v], 0)} | {fmt([x['null'] for x in v], 0)} | "
                f"{fmt([x['split_error'] for x in v], 2)} | {fmt([x['between'] for x in v], 2)} |")
    if "eqm" in R:
        L += [row2(f"EqM, K = {K}", [r["per_K"][K] for r in R["eqm"]]) for K in (10, 20, 30, 100)]
    if "gaussian" in R:
        L.append(row2("Gaussian, samples", [r["samples"] for r in R["gaussian"]]))
    if "eqm" in R:
        L += ["", "## EqM: direction of movement (all states)", "",
              "| from K → to K | distance moved (per action number) | along the data (128 states) | if random | "
              "along the data (64 states) | if random (64) |", "|---|---|---|---|---|---|"]
        for ka, kb in MOVES:
            mv = [r["moves"][f"{ka}->{kb}"] for r in R["eqm"]]
            L.append(f"| {ka} → {kb} | {fmt([x['rms'] for x in mv], 3, False)} | {fmt([x['pca128']['share'] for x in mv], 2, False)} | "
                     f"{fmt([x['pca128']['random'] for x in mv], 2, False)} | {fmt([x['pca64']['share'] for x in mv], 2, False)} | "
                     f"{fmt([x['pca64']['random'] for x in mv], 2, False)} |")
    return "\n".join(L) + "\n"


def main():
    a = args()
    t0 = time.time()
    dev = a.device
    out = Path(a.out) if a.out else HERE / "out" / "two_groups" / a.env
    out.mkdir(parents=True, exist_ok=True)
    if ogbench_data.is_ogbench(a.env):
        env, data, mean, std, ep, dinfo = ogbench_data.load_ogbench(a.env)
        n_ep = dinfo["episodes"]
    else:
        from offline_data import load_d4rl
        env, data, mean, std, ep = load_d4rl(a.env)
        n_ep = int(ep.max()) + 1
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    S_np = ((data["observations"] - mean) / std).astype(np.float32)
    A = data["actions"].astype(np.float32)
    action_dim = A.shape[1]
    q_idx = np.sort(np.random.default_rng(0).choice(len(S_np), a.n_states, replace=False))
    S = torch.tensor(S_np, device=dev)
    nn_vis, nn_plain, r_vis, r_plain, short = neighbours(S, ep, q_idx, a.n_nn, a.per_ep)
    pr = np.random.default_rng(2).choice(len(S_np), (10000, 2))
    r_random = float(np.median((S[pr[:, 0]] - S[pr[:, 1]]).norm(dim=1).cpu().numpy()))
    s = S[q_idx].clone()
    del S
    B_vis, B_plain = A[nn_vis], A[nn_plain]
    print(f"{a.env}: {len(S_np)} transitions, {n_ep} episodes, state {S_np.shape[1]}-D, action {action_dim}-D; "
          f"neighbours in {time.time() - t0:.0f} s", flush=True)

    def groups_of(B):
        g, ratios, btw = [], [], []
        for i in range(len(B)):
            ok, cen, share, ratio, b = group_test(B[i], a.sep)
            ratios.append(ratio)
            if ok:
                g.append((i, cen, share)); btw.append(b)
        return g, np.array(ratios), btw

    groups_vis, ratio_vis, btw_vis = groups_of(B_vis)
    groups_plain, _, _ = groups_of(B_plain)
    rng = np.random.default_rng(1)
    sur = [group_test(rng.multivariate_normal(D.mean(0), np.cov(D.T) + 1e-9 * np.eye(action_dim), size=len(D)), a.sep)[0]
           for D in B_vis]
    subs = dict(pca128=[subspace(B_vis[i]) for i in range(len(B_vis))],
                pca64=[subspace(A[nn_vis[i, :64]]) for i in range(len(B_vis))])
    Bt = torch.tensor(B_vis, device=dev)
    half_a, half_b = Bt[:, 0::2], Bt[:, 1::2]
    data_precision = torch.cdist(half_a, half_b).min(-1).values.mean().item() / action_dim ** 0.5
    info = {
        "dataset": f"{len(S_np)} transitions, {n_ep} episodes; a state has {S_np.shape[1]} numbers, an action {action_dim}",
        "similar states": f"{a.n_nn} closest states from other episodes, at most {a.per_ep} per episode. Median distance of "
                          f"the {a.n_nn}th: {np.median(r_vis):.2f} (own episode allowed: {np.median(r_plain):.2f}); two random "
                          f"dataset states: {r_random:.2f}. {short} states had fewer than {a.n_nn} such states in the search.",
        "two-group states": f"{len(groups_vis)} of {a.n_states} (own episode allowed: {len(groups_plain)}). Smaller group: "
                            f"{np.mean([min(g[2], 1 - g[2]) for g in groups_vis]) if groups_vis else float('nan'):.2f} of the "
                            f"data actions on average",
        "check of the two-group rule": f"single normal distributions with each state's own average and covariance pass in "
                                       f"{int(np.sum(sur))} of {a.n_states} states",
        "main data directions": f"{np.mean([P.shape[1] for P in subs['pca128']]):.1f} of {action_dim} on average "
                                f"(64 states: {np.mean([P.shape[1] for P in subs['pca64']]):.1f})",
        "_n_two_group": len(groups_vis), "_data_between": float(np.mean(btw_vis)) if btw_vis else float("nan"),
        "_data_width": float(B_vis.std(1).mean()), "_data_precision": data_precision,
    }
    for k, v in info.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}", flush=True)

    R = {}
    for kind in a.kinds.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            if kind == "eqm":
                r = run_eqm(a, seed, s, half_b, groups_vis, groups_plain, subs, low, high, dev, out)
                for K in (10, 20, 30, 100):
                    p, v = r["per_K"][K], r["per_K"][K]["visits"]
                    print(f"  eqm seed {seed} K={K:4d}: both groups {v['n_covered']}/{v['n_states']} (random points "
                          f"{v['n_null']}) | width {p['spread']:.3f} distance to data {p['precision']:.3f}", flush=True)
            else:
                r = run_gaussian(a, seed, s, half_b, groups_vis, groups_plain, low, high, dev, out)
                p, v = r["samples"], r["samples"]["visits"]
                print(f"  gaussian seed {seed}: both groups {v['n_covered']}/{v['n_states']} (random points {v['n_null']}) "
                      f"| width {p['spread']:.3f} distance to data {p['precision']:.3f} | mean action in the middle "
                      f"{r['mean_action']['n_middle']}/{r['mean_action']['n_states']}", flush=True)
            R.setdefault(kind, []).append(r)
    json.dump(dict(args=vars(a), info=info, state_index=q_idx.tolist(), results=R),
              open(out / "results.json", "w"), indent=1, default=float)
    np.savez_compressed(out / "reference.npz", state_index=q_idx, visits_actions=B_vis.astype(np.float16),
                        plain_actions=B_plain.astype(np.float16))
    (out / "summary.md").write_text(summary(a, info, R))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
