"""OOD check, part 2: accumulate |f| (GeCO's moving-average / leaky-bucket idea) for state-level OOD detection.
Evaluation only: no training, no episodes, no wandb. Part 1 (single reading of |f|): diagnostics/ood_check.py.

Per state: run the actor's own sampler (Actor._implicit, NAG) from noise, 16 starts, and record the trace
r_k = |f(x_k | s)| for k = 1..1000 (average over the 16 starts). Scores built from the trace:
  |f| at K = 20       part 1's best single reading (GeCO's score).
  moving average      average of r_k over the 5 steps ending at K (GeCO's 'moving average', w = 5).
  sum                 r_1 + ... + r_K (times eta = the length of the path the sampler walks).
  bucket (one-sided)  sum over k <= K of max(0, z_k), z_k = (r_k - m_k) / s_k, where m_k and s_k are the median and
                      the spread (1.4826 * median absolute deviation) of r_k over the calibration states. This is
                      GeCO's leaky bucket (leak 0) with a per-step threshold m_k: it adds up how much slower than
                      normal the sampler is at each step.
  bucket (two-sided)  sum over k <= K of |z_k|: also counts steps where the sampler is FASTER than normal (part 1
                      found OOD states are slower at K = 20 but faster at K = 5-10 and K = 200).
Over time: windows of W consecutive states of one episode; the window's score is the average of its states' scores.
Baseline: distance from the state to the closest dataset state (normalized), averaged the same way.

Sets (all windows of 25 consecutive states inside one episode):
  ID calibration   512 windows of the ID data (only used for m_k, s_k).
  ID test          512 other windows of the ID data. D4RL: training rows (searches skip the own episode);
                   OGBench: the held-out val dataset.
  OOD              512 windows of each --ood_envs dataset; ID test windows + one constant offset per window, drawn
                   from a normal distribution with std 0.3 or 1.0 per state number (a shifted sensor).
Numbers: AUROC (0.5 = chance, 1 = perfect) and, in brackets, % of the OOD set caught when 5 % of ID test is flagged.

Run from algorithms/EqM_RL/ (same envs as ood_check.py):
  LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin D4RL_SUPPRESS_IMPORT_ERROR=1 /workspace/miniconda3/envs/hawk/bin/python \
    diagnostics/ood_accumulate.py --env antmaze-umaze-v2 --ood_envs antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-large-diverse-v2
  /workspace/miniconda3/envs/ogbench/bin/python diagnostics/ood_accumulate.py --env humanoidmaze-medium-navigate-singletask-v0
Output: diagnostics/out/ood_accumulate/<env>/ results.json, summary.md.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ood_check import HERE, ConditionalMLP, auroc, caught, find_run, nearest, ogbench_data

KMAX = 1000
W = 25
WS = [1, 5, 10, 25]
NOISE = [0.3, 1.0]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-umaze-v2")
    p.add_argument("--ood_envs", default="")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--eqm_prefix", default=None)
    p.add_argument("--checkpoint", default="checkpoint_500000.pt")
    p.add_argument("--n", type=int, default=512, help="windows per set")
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--batch", type=int, default=4096, help="states per sampler batch")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.eqm_prefix = a.eqm_prefix or ("Exp14_BC" if ogbench_data.is_ogbench(a.env) else "Exp10_BC")
    return a


def windows(ep, n, rng):
    """n random windows of W consecutive rows inside one episode -> (n, W) row indices."""
    N = len(ep)
    ok = np.flatnonzero(ep[:N - W + 1] == ep[W - 1:])
    return rng.choice(ok, n, replace=False)[:, None] + np.arange(W)


def load(a, rng):
    """-> train obs, train ep, mean, std, {set: (obs (n, W, d), ep (n, W) or None)}; ID sets are 'cal' and 'ID'."""
    sets = {}
    if ogbench_data.is_ogbench(a.env):
        import ogbench
        _, train, val = ogbench.make_env_and_datasets(a.env, dataset_dir=os.path.expanduser(ogbench_data.OGBENCH_DATA),
                                                      compact_dataset=False)
        obs = train["observations"]
        mean, std = obs.mean(0), obs.std(0) + 1e-3
        ends = train["terminals"].astype(bool)
        ep = np.concatenate([[0], np.cumsum(ends)[:-1]])
        vends = val["terminals"].astype(bool)
        vep = np.concatenate([[0], np.cumsum(vends)[:-1]])
        idx = windows(vep, 2 * a.n, rng)
        sets["cal"], sets["ID"] = (val["observations"][idx[:a.n]], None), (val["observations"][idx[a.n:]], None)
    else:
        from offline_data import load_d4rl
        _, data, mean, std, ep = load_d4rl(a.env)
        obs = data["observations"]
        idx = windows(ep, 2 * a.n, rng)
        sets["cal"], sets["ID"] = (obs[idx[:a.n]], ep[idx[:a.n]]), (obs[idx[a.n:]], ep[idx[a.n:]])
        for name in [x for x in a.ood_envs.split(",") if x]:
            _, d2, _, _, ep2 = load_d4rl(name)
            sets[name] = (d2["observations"][windows(ep2, a.n, rng)], None)
    base, base_ep = sets["ID"]
    for sd in NOISE:
        off = rng.standard_normal((a.n, 1, base.shape[-1])) * sd * std     # constant per window, normalized std sd
        sets[f"offset {sd}"] = (base + off, base_ep)
    return obs, ep, mean, std, sets


@torch.no_grad()
def traces(model, s, starts, eta, mu, batch, dev):
    """|f| trace (n, KMAX), average over `starts` sampler runs per state, plus the single-start value at K = 20."""
    out, single = [], []
    adim = model.output_dim
    for b in range(0, len(s), batch):
        sb = s[b:b + batch]
        n = len(sb)
        st = sb.repeat_interleave(starts, 0)
        g = torch.Generator(device=dev).manual_seed(b)
        x = torch.randn(n * starts, adim, device=dev, generator=g)
        mom = torch.zeros_like(x)
        r = torch.empty(n, KMAX, device=dev)
        for k in range(KMAX):
            mom = model(x - eta * mom * mu, st)
            x = x - eta * mom
            fk = model(x, st).norm(dim=1).view(n, starts)
            r[:, k] = fk.mean(1)
            if k == 19:
                single.append(fk[:, 0].clone())
        out.append(r)
    return torch.cat(out), torch.cat(single)


def state_scores(r, m, sd):
    z = (r - m) / sd
    sc = {"|f| at K = 20": r[:, 19]}
    for K in (10, 20, 50):
        sc[f"moving average, K = {K}"] = r[:, K - 5:K].mean(1)
    for K in (20, 50, 200, 1000):
        sc[f"sum, K = {K}"] = r[:, :K].sum(1)
    for K in (20, 50, 200, 1000):
        sc[f"bucket one-sided, K = {K}"] = z[:, :K].clamp(min=0).sum(1)
    for K in (20, 50, 200, 1000):
        sc[f"bucket two-sided, K = {K}"] = z[:, :K].abs().sum(1)
    return sc


def evaluate(scores, oods, n):
    """scores: {set: {name: (n*W,) tensor}} -> per-state and per-window AUROC / caught."""
    res = {"state": {}, "window": {}}
    for name in scores["ID"]:
        res["state"][name] = {o: dict(auroc=auroc(scores["ID"][name].cpu(), scores[o][name].cpu()),
                                      caught=caught(scores["ID"][name].cpu().numpy(), scores[o][name].cpu().numpy()))
                              for o in oods}
        res["window"][name] = {}
        for w in WS:
            avg = {k: v[name].view(n, W)[:, :w].mean(1).cpu().numpy() for k, v in scores.items()}
            res["window"][name][w] = {o: dict(auroc=auroc(avg["ID"], avg[o]), caught=caught(avg["ID"], avg[o]))
                                      for o in oods}
    return res


def main():
    a = args()
    t0 = time.time()
    dev = a.device
    out = Path(a.out) if a.out else HERE / "out" / "ood_accumulate" / a.env
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    obs, ep, mean, std, sets = load(a, rng)
    norm = lambda x: torch.tensor(((x - mean) / std).astype(np.float32), device=dev)  # noqa: E731
    S, ep_t = norm(obs), torch.tensor(ep, device=dev)
    oods = [k for k in sets if k not in ("cal", "ID")]
    flat = {k: norm(v.reshape(-1, v.shape[-1])) for k, (v, _) in sets.items()}
    print(f"{a.env}: {len(S)} training rows; sets {list(sets)} ({a.n} windows of {W} states each)", flush=True)

    dist = {}
    for k, (_, e) in sets.items():
        e_q = torch.full((len(flat[k]),), -1, device=dev) if e is None else torch.tensor(e.reshape(-1), device=dev)
        dist[k] = torch.cat([nearest(S, ep_t, flat[k][i:i + 2048], e_q[i:i + 2048], 1)[0][:, 0]
                             for i in range(0, len(flat[k]), 2048)])
    del S
    base = evaluate({k: {"distance to closest dataset state": v} for k, v in dist.items() if k != "cal"}, oods, a.n)
    print(f"  distances done ({time.time() - t0:.0f} s)", flush=True)

    R = dict(baseline=base, eqm=[])
    for seed in [int(x) for x in a.seeds.split(",")]:
        run, cfg = find_run(a.eqm_prefix, a.env, "eqm", seed, a.checkpoint)
        sd_ = torch.load(os.path.join(run, a.checkpoint), map_location=dev)["model"]
        in_dim = sd_[[k for k in sd_ if k.endswith("weight")][0]].shape[1]       # state + action numbers
        model = ConditionalMLP(in_dim, in_dim - obs.shape[1], cfg["hidden_dim"], cfg["num_hidden_layers"]).to(dev)
        model.load_state_dict(sd_)
        model.eval().requires_grad_(False)
        eta, mu = float(cfg["actor"]["step_size"]), float(cfg["actor"]["moment"])
        tr, single = {}, {}
        for k in sets:
            tr[k], single[k] = traces(model, flat[k], a.starts, eta, mu, a.batch, dev)
        m = tr["cal"].median(0).values
        s = 1.4826 * (tr["cal"] - m).abs().median(0).values + 1e-6
        scores = {k: state_scores(tr[k], m, s) for k in sets if k != "cal"}
        for k in scores:
            scores[k]["|f| at K = 20, 1 start"] = single[k]
        res = evaluate(scores, oods, a.n)
        res.update(seed=seed, run=run, median_trace={k: tr[k].median(0).values[[0, 4, 9, 19, 49, 199, 999]].tolist()
                                                     for k in sets})
        R["eqm"].append(res)
        print(f"  seed {seed} ({time.time() - t0:.0f} s): " + "; ".join(
            f"{n}: " + ", ".join(f"{o} {res['state'][n][o]['auroc']:.2f}" for o in oods)
            for n in ("|f| at K = 20", "sum, K = 1000", "bucket two-sided, K = 200")), flush=True)
    json.dump(dict(args=vars(a), results=R), open(out / "results.json", "w"), indent=1, default=float)
    (out / "summary.md").write_text(summary(a, R, oods))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


def fmt(v, d=2):
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def summary(a, R, oods):
    E, B = R["eqm"], R["baseline"]
    L = [f"# OOD check, accumulated |f|: `{a.env}` ({a.eqm_prefix} EqM actors, seeds {a.seeds})", "",
         "Evaluation only; script and definitions: `diagnostics/ood_accumulate.py`. Cells: AUROC, mean ± std over seeds "
         "(0.5 = chance, 1 = perfect), and in brackets the % of the OOD set caught when 5 % of the ID test set is flagged.",
         "", f"## One state at a time ({a.n * W} states per set)", "", "| score | " + " | ".join(oods) + " |",
         "|---|" + "---|" * len(oods)]
    for name in E[0]["state"]:
        L.append(f"| {name} | " + " | ".join(
            f"{fmt([r['state'][name][o]['auroc'] for r in E])} ({fmt([r['state'][name][o]['caught'] for r in E], 0)})"
            for o in oods) + " |")
    bn = "distance to closest dataset state"
    L.append(f"| {bn} | " + " | ".join(f"{B['state'][bn][o]['auroc']:.2f} ({B['state'][bn][o]['caught']:.0f})"
                                       for o in oods) + " |")
    L += ["", f"## Averaged over W consecutive states of one episode ({a.n} windows per set)", "",
          "| score | W | " + " | ".join(oods) + " |", "|---|---|" + "---|" * len(oods)]
    for name in ("|f| at K = 20", "sum, K = 1000", "bucket one-sided, K = 200", "bucket two-sided, K = 200"):
        for w in WS:
            L.append(f"| {name} | {w} | " + " | ".join(
                f"{fmt([r['window'][name][w][o]['auroc'] for r in E])} "
                f"({fmt([r['window'][name][w][o]['caught'] for r in E], 0)})" for o in oods) + " |")
    for w in WS:
        L.append(f"| {bn} | {w} | " + " | ".join(
            f"{B['window'][bn][w][o]['auroc']:.2f} ({B['window'][bn][w][o]['caught']:.0f})" for o in oods) + " |")
    L += ["", "Median |f| trace (average of 16 starts), seed 1:", "", "| set | k=1 | 5 | 10 | 20 | 50 | 200 | 1000 |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in E[0]["median_trace"].items():
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
