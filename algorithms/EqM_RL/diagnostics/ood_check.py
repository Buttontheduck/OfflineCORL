"""OOD check: can the trained BC actors tell in-distribution (ID) from out-of-distribution (OOD) inputs?
Evaluation only: no training, no episodes, no wandb. Tests the two scores of the papers in papers/:

  GeCO (state score)   run the actor's own sampler from noise for K steps at state s, then read |f(a_K | s)|.
                       High = the sampler did not settle = s is OOD. GeCO: AUROC 0.93, LIBERO-Goal -> Spatial.
  EqM paper (point score)  score the given point itself. The paper used an explicit energy model (EqM-E); our
                       actors are implicit (they output f only), so two stand-ins are used:
                         |f(a|s)|      the field length at the action itself;
                         energy drop   sum over 200 plain gradient steps started at a of eta * |f|^2. For a true
                                       gradient field this is E(a) - E(end point), the energy above the minimum.

Baselines (no EqM involved):
  state distance       distance from s to the closest dataset state (normalized state numbers).
  Gaussian BC          for actions: |(a - mean(s)) / std|, the Gaussian BC actor's distance in std units
                       (= its -log-likelihood up to a constant).
  action distance      for actions: distance from a to the closest of the 64 data actions taken in the 64 closest
                       dataset states.
For D4RL the ID rows are training rows, so all nearest-state searches skip the query's own episode (else the next
time step of the same episode is always ~0 away). For OGBench the ID rows come from the held-out val dataset.

ID: n dataset (state, action) pairs. OOD sets:
  states   other datasets of the same robot (--ood_envs, e.g. other AntMaze mazes); ID states + normal noise of
           std 0.3 and 1.0 (normalized units, every state number).
  actions  (all at the ID states) data action + noise std 0.1 / 0.3 (clipped); the data action of another ID row
           ("swapped": a real action, but for a different state); uniform in the action box; the EqM actor's own
           samples at K = 20 and K = 1000 (sample 0 of each state).
Numbers: AUROC (probability that a random OOD input scores higher than a random ID input; 0.5 = no better than
chance, 1 = perfect) and "caught at 5 %": share of the OOD set flagged when the threshold flags 5 % of the ID set.

Run from algorithms/EqM_RL/:
  D4RL:    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin D4RL_SUPPRESS_IMPORT_ERROR=1 /workspace/miniconda3/envs/hawk/bin/python \
             diagnostics/ood_check.py --env antmaze-umaze-v2 --ood_envs antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-large-diverse-v2
  OGBench: /workspace/miniconda3/envs/ogbench/bin/python diagnostics/ood_check.py --env humanoidmaze-medium-navigate-singletask-v0
Output: diagnostics/out/ood_check/<env>/ results.json, summary.md.
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
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Aware"))
import ogbench_data  # noqa: E402
from gaussian_actor import GaussianActor  # noqa: E402
from utils.networks import ConditionalMLP  # noqa: E402

CKPT_ROOT = "/workspace/Trained_Models/checkpoints"
KS = [5, 10, 20, 50, 200, 1000]
STATE_NOISE = [0.3, 1.0]
ACTION_NOISE = [0.1, 0.3]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-umaze-v2")
    p.add_argument("--ood_envs", default="", help="comma list of other datasets of the same robot (same state size)")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--eqm_prefix", default=None, help="default: Exp14_BC (OGBench), Exp10_BC (D4RL)")
    p.add_argument("--gauss_prefix", default=None, help="default: Exp17_BC (humanoidmaze), else eqm_prefix")
    p.add_argument("--checkpoint", default="checkpoint_500000.pt")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--starts", type=int, default=16, help="sampler starts per state for the GeCO score")
    p.add_argument("--n_nn", type=int, default=64)
    p.add_argument("--gd_steps", type=int, default=200)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ogb = ogbench_data.is_ogbench(a.env)
    a.eqm_prefix = a.eqm_prefix or ("Exp14_BC" if ogb else "Exp10_BC")
    a.gauss_prefix = a.gauss_prefix or ("Exp17_BC" if a.env.startswith("humanoidmaze") else a.eqm_prefix)
    return a


def find_run(prefix, env, kind, seed, ckpt):
    pat = os.path.join(CKPT_ROOT, f"{prefix}_env_name{env}_actor{kind}_seed{seed}-{env}-*")
    dirs = [d for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, ckpt))]
    assert len(dirs) == 1, f"expected one run with {ckpt} for {pat}, found {dirs}"
    cfg = yaml.safe_load(open(os.path.join(dirs[0], "config.yaml")))
    assert cfg["env_name"] == env and cfg["seed"] == seed and (cfg.get("actor") or {}).get("kind") in (None, kind)
    return dirs[0], cfg


def load(a):
    """-> train obs, train actions, train episode ids, mean, std, (low, high), ID rows dict, ood state sets dict."""
    rng = np.random.default_rng(0)
    ood = {}
    if ogbench_data.is_ogbench(a.env):
        import ogbench
        env, train, val = ogbench.make_env_and_datasets(a.env, dataset_dir=os.path.expanduser(ogbench_data.OGBENCH_DATA),
                                                        compact_dataset=False)
        obs, act = train["observations"], train["actions"]
        mean, std = obs.mean(0), obs.std(0) + 1e-3                    # as ogbench_data.load_ogbench
        ends = train["terminals"].astype(bool)
        ep = np.concatenate([[0], np.cumsum(ends)[:-1]])
        idx = np.sort(rng.choice(len(val["observations"]), a.n, replace=False))
        id_rows = dict(s=val["observations"][idx], a=val["actions"][idx], ep=np.full(a.n, -1), source="val dataset (held out)")
    else:
        from offline_data import load_d4rl
        env, data, mean, std, ep = load_d4rl(a.env)
        obs, act = data["observations"], data["actions"]
        idx = np.sort(rng.choice(len(obs), a.n, replace=False))
        id_rows = dict(s=obs[idx], a=act[idx], ep=ep[idx], source="training rows (own episode skipped in searches)")
        for name in [x for x in a.ood_envs.split(",") if x]:
            _, d2, _, _, _ = load_d4rl(name)
            o2 = d2["observations"]
            assert o2.shape[1] == obs.shape[1], name
            ood[name] = o2[np.sort(rng.choice(len(o2), a.n, replace=False))]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    return obs, act, ep, mean, std, (low, high), id_rows, ood


@torch.no_grad()
def nearest(S, ep_t, Q, ep_q, k, chunk=200_000):
    """k nearest training rows of each query (rows of the query's own episode skipped) -> (dist, idx) on GPU."""
    best_d = best_i = None
    for a in range(0, len(S), chunk):
        d = torch.cdist(Q, S[a:a + chunk])
        d = d.masked_fill(ep_q[:, None] == ep_t[None, a:a + chunk], float("inf"))
        dk, ik = d.topk(k, largest=False)
        ik = ik + a
        if best_d is not None:
            dk, ik = torch.cat([best_d, dk], 1), torch.cat([best_i, ik], 1)
        o = dk.topk(k, largest=False)
        best_d, best_i = o.values, ik.gather(1, o.indices)
    return best_d, best_i


def auroc(id_s, ood_s):
    id_s, ood_s = np.asarray(id_s, np.float64), np.asarray(ood_s, np.float64)
    r = rankdata(np.concatenate([id_s, ood_s]))
    n0, n1 = len(id_s), len(ood_s)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def caught(id_s, ood_s, fpr=0.05):
    return float(100 * (np.asarray(ood_s) > np.quantile(id_s, 1 - fpr)).mean())


def compare(scores, id_key):
    """scores: {set: 1-D array}. -> {set: dict(auroc, caught, median)} against scores[id_key]."""
    out = {}
    for k, v in scores.items():
        out[k] = dict(median=float(np.median(v)))
        if k != id_key:
            out[k].update(auroc=auroc(scores[id_key], v), caught=caught(scores[id_key], v))
    return out


@torch.no_grad()
def sampler_fnorm(model, s, starts, eta, mu, adim, dev):
    """Actor._implicit (NAG) from noise, `starts` runs per state. -> {K: |f| (n, starts)}, {K: sample 0 (n, adim)}."""
    n = len(s)
    st = s.repeat_interleave(starts, 0)
    torch.manual_seed(0)
    x = torch.randn(n * starts, adim, device=dev)
    mom = torch.zeros_like(x)
    fn, x0 = {}, {}
    for k in range(1, max(KS) + 1):
        mom = model(x - eta * mom * mu, st)
        x = x - eta * mom
        if k in KS:
            fn[k] = model(x, st).norm(dim=1).view(n, starts)
            x0[k] = x.view(n, starts, adim)[:, 0].clone()
    return fn, x0


@torch.no_grad()
def energy_drop(model, s, x, eta, steps):
    """Plain gradient descent from x: sum of eta*|f|^2 (= E(x) - E(end) for a true gradient field)."""
    e = torch.zeros(len(x), device=x.device)
    for _ in range(steps):
        f = model(x, s)
        e += eta * (f ** 2).sum(1)
        x = x - eta * f
    return e


def main():
    a = args()
    t0 = time.time()
    dev = a.device
    out = Path(a.out) if a.out else HERE / "out" / "ood_check" / a.env
    out.mkdir(parents=True, exist_ok=True)
    obs, act, ep, mean, std, (low, high), id_rows, ood_env_states = load(a)
    norm = lambda x: torch.tensor(((x - mean) / std).astype(np.float32), device=dev)  # noqa: E731
    S, A = norm(obs), torch.tensor(act.astype(np.float32), device=dev)
    ep_t = torch.tensor(ep, device=dev)
    adim = A.shape[1]
    n = a.n
    print(f"{a.env}: {len(S)} training rows, state {S.shape[1]}-D, action {adim}-D; ID = {id_rows['source']}", flush=True)

    # ---- state sets
    s_id = norm(id_rows["s"])
    ep_id = torch.tensor(id_rows["ep"], device=dev)
    g = torch.Generator(device=dev).manual_seed(1)
    states = {"ID": (s_id, ep_id)}
    for name, o in ood_env_states.items():
        states[name] = (norm(o), torch.full((n,), -1, device=dev))
    for sd in STATE_NOISE:
        states[f"noise {sd}"] = (s_id + sd * torch.randn(s_id.shape, generator=g, device=dev), ep_id)
    state_dist, nn_idx = {}, None
    for k, (s, e) in states.items():
        d, i = nearest(S, ep_t, s, e, a.n_nn if k == "ID" else 1)
        state_dist[k] = d[:, 0].cpu().numpy()
        if k == "ID":
            nn_idx = i
    print(f"  nearest-state searches done ({time.time() - t0:.0f} s)", flush=True)

    # ---- action sets at the ID states
    a_id = torch.tensor(id_rows["a"].astype(np.float32), device=dev)
    actions = {"ID": a_id}
    for sd in ACTION_NOISE:
        actions[f"noise {sd}"] = (a_id + sd * torch.randn(a_id.shape, generator=g, device=dev)).clamp(low, high)
    actions["swapped"] = a_id[torch.randperm(n, generator=g, device=dev)]
    actions["uniform"] = low + (high - low) * torch.rand(a_id.shape, generator=g, device=dev)
    nn_act = A[nn_idx]                                                          # (n, n_nn, adim)

    def act_dist(x):
        return (nn_act - x[:, None]).norm(dim=-1).min(1).values.cpu().numpy()

    R = dict(state_distance=compare(state_dist, "ID"), eqm=[], gaussian=[])
    act_dist_scores = {k: act_dist(v) for k, v in actions.items()}

    for seed in [int(x) for x in a.seeds.split(",")]:
        run, cfg = find_run(a.eqm_prefix, a.env, "eqm", seed, a.checkpoint)
        model = ConditionalMLP(S.shape[1] + adim, adim, cfg["hidden_dim"], cfg["num_hidden_layers"]).to(dev)
        model.load_state_dict(torch.load(os.path.join(run, a.checkpoint), map_location=dev)["model"])
        model.eval().requires_grad_(False)
        eta, mu = float(cfg["actor"]["step_size"]), float(cfg["actor"]["moment"])
        assert cfg["actor"]["opt_type"] == "nag"
        r = dict(seed=seed, run=run, eta=eta, momentum=mu, lam=cfg.get("scale_gradient"), geco={}, geco_single={})
        fn = {}
        for k, (s, _) in states.items():
            fn[k], x0 = sampler_fnorm(model, s, a.starts, eta, mu, adim, dev)
            if k == "ID":
                samples = x0
        for K in KS:
            r["geco"][K] = compare({k: fn[k][K].mean(1).cpu().numpy() for k in states}, "ID")
            r["geco_single"][K] = compare({k: fn[k][K][:, 0].cpu().numpy() for k in states}, "ID")
        acts = dict(actions)
        acts["EqM sample K=20"], acts["EqM sample K=1000"] = samples[20].clamp(low, high), samples[1000].clamp(low, high)
        f_at = {k: model(v, s_id).norm(dim=1).cpu().numpy() for k, v in acts.items()}
        e_at = {k: energy_drop(model, s_id, v.clone(), eta, a.gd_steps).cpu().numpy() for k, v in acts.items()}
        ad = dict(act_dist_scores)
        for k in ("EqM sample K=20", "EqM sample K=1000"):
            ad[k] = act_dist(acts[k])
        r["f_at_action"], r["energy_drop"], r["action_distance"] = compare(f_at, "ID"), compare(e_at, "ID"), compare(ad, "ID")
        R["eqm"].append(r)
        print(f"  eqm seed {seed}: GeCO K=20 AUROC " + ", ".join(
            f"{k} {v['auroc']:.2f}" for k, v in r["geco"][20].items() if k != "ID") + " | |f| at action AUROC " + ", ".join(
            f"{k} {v['auroc']:.2f}" for k, v in r["f_at_action"].items() if k != "ID"), flush=True)

        grun, gcfg = find_run(a.gauss_prefix, a.env, "gaussian", seed, a.checkpoint)
        lo_std, hi_std = gcfg.get("min_log_std", -20.0), gcfg.get("max_log_std", 2.0)
        gm = GaussianActor(S.shape[1], adim, gcfg["hidden_dim"], gcfg.get("num_hidden_layers", 3), lo_std, hi_std).to(dev)
        gm.load_state_dict(torch.load(os.path.join(grun, a.checkpoint), map_location=dev)["actor"])
        gm.eval().requires_grad_(False)
        with torch.no_grad():
            gstd = gm._log_std.clamp(lo_std, hi_std).exp()
            mu_s = gm._mlp(s_id)
            gs = {k: ((v - mu_s) / gstd).norm(dim=1).cpu().numpy() for k, v in acts.items()}
        R["gaussian"].append(dict(seed=seed, run=grun, std=gstd.tolist(), score=compare(gs, "ID")))
    json.dump(dict(args=vars(a), id_source=id_rows["source"], ks=KS, results=R), open(out / "results.json", "w"),
              indent=1, default=float)
    (out / "summary.md").write_text(summary(a, id_rows["source"], R, list(states), list(acts)))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


def fmt(vals, d=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.{d}f} ± {np.std(vals):.{d}f}" if len(vals) > 1 else f"{vals[0]:.{d}f}"


def summary(a, src, R, state_sets, act_sets):
    oods = [k for k in state_sets if k != "ID"]
    L = [f"# OOD check: `{a.env}` (EqM: {a.eqm_prefix}, Gaussian: {a.gauss_prefix}, seeds {a.seeds})", "",
         f"Evaluation only; script and definitions: `diagnostics/ood_check.py`. ID = {a.n} rows from the {src}. "
         "Cells: AUROC, mean ± std over seeds (0.5 = chance, 1 = perfect), then in brackets the % of the OOD set caught "
         "when 5 % of the ID set is flagged.", "",
         "## States: GeCO score (|f| after K sampler steps from noise) vs distance to the closest dataset state", "",
         "| score | " + " | ".join(oods) + " |", "|---|" + "---|" * len(oods)]
    for K in KS:
        for name, key in (("mean of 16 starts", "geco"), ("1 start", "geco_single")):
            if key == "geco_single" and K not in (10, 20):
                continue
            L.append(f"| GeCO, K = {K}, {name} | " + " | ".join(
                f"{fmt([r[key][K][o]['auroc'] for r in R['eqm']])} ({fmt([r[key][K][o]['caught'] for r in R['eqm']], 0)})"
                for o in oods) + " |")
    sd = R["state_distance"]
    L.append("| distance to closest dataset state | " + " | ".join(f"{sd[o]['auroc']:.2f} ({sd[o]['caught']:.0f})" for o in oods) + " |")
    L += ["", "Median GeCO score (mean of 16 starts) per set:", "",
          "| K | " + " | ".join(state_sets) + " |", "|---|" + "---|" * len(state_sets)]
    for K in KS:
        L.append(f"| {K} | " + " | ".join(fmt([r["geco"][K][k]["median"] for r in R["eqm"]]) for k in state_sets) + " |")
    L.append("| distance to closest state | " + " | ".join(f"{sd[k]['median']:.2f}" for k in state_sets) + " |")
    aoods = [k for k in act_sets if k != "ID"]
    L += ["", "## Actions at the ID states: EqM-paper-style scores vs baselines", "",
          "| score | " + " | ".join(aoods) + " |", "|---|" + "---|" * len(aoods)]

    def arow(name, items):
        return f"| {name} | " + " | ".join(
            f"{fmt([x[o]['auroc'] for x in items])} ({fmt([x[o]['caught'] for x in items], 0)})" for o in aoods) + " |"
    L.append(arow("EqM: |f| at the action", [r["f_at_action"] for r in R["eqm"]]))
    L.append(arow(f"EqM: energy drop ({a.gd_steps} GD steps)", [r["energy_drop"] for r in R["eqm"]]))
    L.append(arow("Gaussian BC: distance in std units", [r["score"] for r in R["gaussian"]]))
    L.append(arow(f"distance to closest data action ({a.n_nn} closest states)", [r["action_distance"] for r in R["eqm"]]))
    L += ["", "Median score per set:", "", "| score | ID | " + " | ".join(aoods) + " |", "|---|---|" + "---|" * len(aoods)]
    for name, items in (("EqM |f|", [r["f_at_action"] for r in R["eqm"]]),
                        ("EqM energy drop", [r["energy_drop"] for r in R["eqm"]]),
                        ("Gaussian distance (std units)", [r["score"] for r in R["gaussian"]]),
                        ("distance to closest data action", [r["action_distance"] for r in R["eqm"]])):
        L.append(f"| {name} | " + " | ".join(fmt([x[k]["median"] for x in items]) for k in ["ID"] + aoods) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
