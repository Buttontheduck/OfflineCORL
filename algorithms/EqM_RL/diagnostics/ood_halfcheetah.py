"""OOD check on D4RL HalfCheetah, where the actor and the critic both learn (Aware 95.5 on medium-expert at 1M
steps, 46.9 on medium-replay). Evaluation only: no training, no episodes, no wandb. Same tests as
diagnostics/ood_check.py (part 1) and diagnostics/ood_accumulate.py (part 2), plus the critics.

Actors: Aware runs (EqM actor + its two critics) and CORL AWAC runs (Gaussian actor + its two critics) on the same
dataset, seeds 1-3, latest checkpoint.

States (windows of 25 consecutive states inside one episode, 512 windows per set):
  ID calibration / ID test   training windows (searches skip the own episode); calibration only sets the
                             per-step reference of the bucket scores.
  OOD                        --ood_envs datasets (e.g. halfcheetah-random-v2); --expert_half: the expert episodes of
                             medium-expert (for an actor trained on medium-replay: a better runner than any data);
                             ID test windows + one constant offset per window (std 0.3 / 1.0 per normalized state number).
  Scores: |f| after K steps of the sampler from noise (GeCO; mean of 16 starts, and 1 start at K = 20); sum of |f| over
  steps 1..K; bucket one/two-sided (ood_accumulate.py); distance to the closest dataset state. Over time: average
  over W = 1/5/10/25 consecutive states.

Actions (at 1024 random training states; searches skip the own episode):
  ID = the data action. OOD sets: data action + noise 0.1 / 0.3; swapped (the data action of another state);
  uniform; EqM sample at K = 20 (the action Aware executes) and K = 1000; the AWAC mean action (the action AWAC
  executes); Q-ascent 0.2 / 1.0: the K = 20 EqM sample moved uphill on Aware's own critic min(Q1, Q2) in steps of
  0.05 (4 / 20 steps, clipped to the box): the actions a policy that maximizes this critic would pick.
  Scores: EqM |f| at the action; EqM energy drop (200 plain GD steps, sum of eta |f|^2); AWAC Gaussian distance
  |(a - mean(s)) / std|; distance to the closest of the 64 data actions of the 64 closest states.
  Critics: Q(s, a) - Q(s, data action) for Aware's critic and for AWAC's critic (median), and the share of
  actions that the critic rates above the data action.

Run from algorithms/EqM_RL/ (hawk env):
  LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin D4RL_SUPPRESS_IMPORT_ERROR=1 /workspace/miniconda3/envs/hawk/bin/python \
    diagnostics/ood_halfcheetah.py --env halfcheetah-medium-expert-v2 --eqm_runs 'Aware_1M_seed{seed}-{env}-*' \
    --ood_envs halfcheetah-random-v2,halfcheetah-medium-replay-v2
  ... --env halfcheetah-medium-replay-v2 --eqm_runs 'Aware_env_name{env}_seed{seed}-{env}-*' \
    --ood_envs halfcheetah-random-v2 --expert_half
Output: diagnostics/out/ood_halfcheetah/<env>/ results.json, summary.md.
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

import ood_accumulate as acc
from ood_check import HERE, ConditionalMLP, auroc, caught, compare, energy_drop, nearest
from common import latest_checkpoint

sys.path.insert(0, str(HERE.parent.parent / "offline"))
import awac as corl_awac  # noqa: E402  (CORL's algorithms/offline/awac.py)
from critic_models import DeterministicCritic  # noqa: E402  (Aware/, on the path through ood_check)

CKPT_ROOT = "/workspace/Trained_Models/checkpoints"
KS_SINGLE = [5, 10, 20, 50, 200, 1000]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="halfcheetah-medium-expert-v2")
    p.add_argument("--eqm_runs", default="Aware_1M_seed{seed}-{env}-*")
    p.add_argument("--gauss_runs", default="AWAC_env_name{env}_seed{seed}-{env}-*")
    p.add_argument("--ood_envs", default="halfcheetah-random-v2,halfcheetah-medium-replay-v2")
    p.add_argument("--expert_half", action="store_true")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--n_win", type=int, default=512)
    p.add_argument("--n_act", type=int, default=1024)
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--n_nn", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_dir(pattern, env, seed):
    dirs = sorted(glob.glob(os.path.join(CKPT_ROOT, pattern.format(env=env, seed=seed))))
    assert len(dirs) == 1, (pattern, dirs)
    cfg = yaml.safe_load(open(os.path.join(dirs[0], "config.yaml")))
    assert cfg["env_name"] == env and cfg["seed"] == seed, dirs[0]
    return Path(dirs[0]), cfg


def load_aware(d, cfg, sdim, adim, dev):
    ck = latest_checkpoint(d)
    sd = torch.load(ck, map_location=dev)
    model = ConditionalMLP(sdim + adim, adim, cfg["hidden_dim"], cfg["num_hidden_layers"]).to(dev)
    model.load_state_dict(sd["model"])
    crit = []
    for key in ("critic_1", "critic_2"):
        c = DeterministicCritic(sdim, adim, cfg["hidden_dim"], cfg["num_hidden_layers"],
                                cfg.get("critic_layer_norm", False)).to(dev)
        c.load_state_dict(sd[key])
        crit.append(c)
    for m in [model] + crit:
        m.eval().requires_grad_(False)
    return model, crit, ck.name


def load_awac(d, cfg, sdim, adim, dev):
    ck = latest_checkpoint(d)
    sd = torch.load(ck, map_location=dev)
    actor = corl_awac.Actor(sdim, adim, cfg["hidden_dim"]).to(dev)
    actor.load_state_dict(sd["actor"])
    crit = []
    for key in ("critic_1", "critic_2"):
        c = corl_awac.Critic(sdim, adim, cfg["hidden_dim"]).to(dev)
        c.load_state_dict(sd[key])
        crit.append(c)
    for m in [actor] + crit:
        m.eval().requires_grad_(False)
    return actor, crit, ck.name


def qmin(crit, s, a):
    return torch.min(crit[0](s, a), crit[1](s, a)).reshape(-1)


def q_ascent(crit, s, a, steps, low, high, step=0.05):
    a = a.clone()
    for _ in range(steps):
        with torch.enable_grad():
            a.requires_grad_(True)
            g, = torch.autograd.grad(qmin(crit, s, a).sum(), a)
        a = (a.detach() + step * g / (g.norm(dim=1, keepdim=True) + 1e-8)).clamp(low, high)
    return a


def main():
    a = args()
    t0 = time.time()
    dev = a.device
    out = Path(a.out) if a.out else HERE / "out" / "ood_halfcheetah" / a.env
    out.mkdir(parents=True, exist_ok=True)
    from offline_data import load_d4rl
    env, data, mean, std, ep = load_d4rl(a.env)
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    obs, act = data["observations"], data["actions"]
    sdim, adim = obs.shape[1], act.shape[1]
    norm = lambda x: torch.tensor(((x - mean) / std).astype(np.float32), device=dev)  # noqa: E731
    S, A, ep_t = norm(obs), torch.tensor(act.astype(np.float32), device=dev), torch.tensor(ep, device=dev)
    rng = np.random.default_rng(0)

    # ---------------- state windows
    W, n = acc.W, a.n_win
    idx = acc.windows(ep, 2 * n, rng)
    sets = {"cal": (obs[idx[:n]], ep[idx[:n]]), "ID": (obs[idx[n:]], ep[idx[n:]])}
    for name in [x for x in a.ood_envs.split(",") if x]:
        _, d2, _, _, ep2 = load_d4rl(name)
        sets[name] = (d2["observations"][acc.windows(ep2, n, rng)], None)
    if a.expert_half:
        _, d2, _, _, ep2 = load_d4rl("halfcheetah-medium-expert-v2")
        keep = ep2 >= (ep2.max() + 1) // 2                       # the last 1,000 episodes: the expert policy
        o2, e2 = d2["observations"][keep], ep2[keep]
        sets["expert episodes"] = (o2[acc.windows(e2, n, rng)], None)
    base, base_ep = sets["ID"]
    for sd in acc.NOISE:
        sets[f"offset {sd}"] = (base + rng.standard_normal((n, 1, sdim)) * sd * std, base_ep)
    oods = [k for k in sets if k not in ("cal", "ID")]
    flat = {k: norm(v.reshape(-1, sdim)) for k, (v, _) in sets.items()}
    dist = {}
    for k, (_, e) in sets.items():
        e_q = torch.full((len(flat[k]),), -1, device=dev) if e is None else torch.tensor(e.reshape(-1), device=dev)
        dist[k] = torch.cat([nearest(S, ep_t, flat[k][i:i + 2048], e_q[i:i + 2048], 1)[0][:, 0]
                             for i in range(0, len(flat[k]), 2048)])
    base_res = acc.evaluate({k: {"distance to closest dataset state": v} for k, v in dist.items() if k != "cal"}, oods, n)

    # ---------------- action rows
    rows = np.sort(rng.choice(len(obs), a.n_act, replace=False))
    s_a, a_id = S[rows], A[rows]
    ep_a = ep_t[rows]
    nn_d, nn_i = nearest(S, ep_t, s_a, ep_a, a.n_nn)
    nn_act = A[nn_i]
    g = torch.Generator(device=dev).manual_seed(1)
    base_acts = {"ID": a_id}
    for sd in (0.1, 0.3):
        base_acts[f"noise {sd}"] = (a_id + sd * torch.randn(a_id.shape, generator=g, device=dev)).clamp(low, high)
    base_acts["swapped"] = a_id[torch.randperm(a.n_act, generator=g, device=dev)]
    base_acts["uniform"] = low + (high - low) * torch.rand(a_id.shape, generator=g, device=dev)
    print(f"{a.env}: {len(obs)} rows; state sets {list(sets)}; searches done ({time.time() - t0:.0f} s)", flush=True)

    R = dict(baseline=base_res, eqm=[])
    for seed in [int(x) for x in a.seeds.split(",")]:
        d, cfg = run_dir(a.eqm_runs, a.env, seed)
        model, crit, ck = load_aware(d, cfg, sdim, adim, dev)
        gd, gcfg = run_dir(a.gauss_runs, a.env, seed)
        gactor, gcrit, gck = load_awac(gd, gcfg, sdim, adim, dev)
        eta, mu = float(cfg["actor"]["step_size"]), float(cfg["actor"]["moment"])
        r = dict(seed=seed, run=str(d), checkpoint=ck, gauss_run=str(gd), gauss_checkpoint=gck)

        # states
        tr, single = {}, {}
        for k in sets:
            tr[k], single[k] = acc.traces(model, flat[k], a.starts, eta, mu, 4096, dev)
        m = tr["cal"].median(0).values
        s_ = 1.4826 * (tr["cal"] - m).abs().median(0).values + 1e-6
        scores = {}
        for k in sets:
            if k == "cal":
                continue
            z = (tr[k] - m) / s_
            sc = {f"|f| at K = {K}": tr[k][:, K - 1] for K in KS_SINGLE}
            sc["|f| at K = 20, 1 start"] = single[k]
            sc["moving average, K = 20"] = tr[k][:, 15:20].mean(1)
            for K in (20, 50, 1000):
                sc[f"sum, K = {K}"] = tr[k][:, :K].sum(1)
            for K in (20, 50, 200):
                sc[f"bucket one-sided, K = {K}"] = z[:, :K].clamp(min=0).sum(1)
                sc[f"bucket two-sided, K = {K}"] = z[:, :K].abs().sum(1)
            scores[k] = sc
        r["states"] = acc.evaluate(scores, oods, n)
        r["median_trace"] = {k: tr[k].median(0).values[[0, 4, 9, 19, 49, 199, 999]].tolist() for k in sets}
        del tr

        # actions
        torch.manual_seed(0)
        x = torch.randn(a.n_act, adim, device=dev)
        mom = torch.zeros_like(x)
        with torch.no_grad():
            for k in range(1, 1001):
                mom = model(x - eta * mom * mu, s_a)
                x = x - eta * mom
                if k == 20:
                    x20 = x.clamp(low, high)
            x1000 = x.clamp(low, high)
            gstd = gactor._log_std.clamp(gactor._min_log_std, gactor._max_log_std).exp()
            gmean = gactor._mlp(s_a)
        acts = dict(base_acts)
        acts["EqM sample K=20"], acts["EqM sample K=1000"] = x20, x1000
        acts["AWAC mean action"] = gmean.clamp(low, high)
        acts["Q-ascent 0.2"] = q_ascent(crit, s_a, x20, 4, low, high)
        acts["Q-ascent 1.0"] = q_ascent(crit, s_a, x20, 20, low, high)
        with torch.no_grad():
            f_at = {k: model(v, s_a).norm(dim=1).cpu().numpy() for k, v in acts.items()}
            e_at = {k: energy_drop(model, s_a, v.clone(), eta, 200).cpu().numpy() for k, v in acts.items()}
            gs = {k: ((v - gmean) / gstd).norm(dim=1).cpu().numpy() for k, v in acts.items()}
            ad = {k: (nn_act - v[:, None]).norm(dim=-1).min(1).values.cpu().numpy() for k, v in acts.items()}
            q0, g0 = qmin(crit, s_a, a_id), qmin(gcrit, s_a, a_id)
            qa = {k: (qmin(crit, s_a, v) - q0).cpu().numpy() for k, v in acts.items()}
            qg = {k: (qmin(gcrit, s_a, v) - g0).cpu().numpy() for k, v in acts.items()}
        r["actions"] = dict(f_at_action=compare(f_at, "ID"), energy_drop=compare(e_at, "ID"), gaussian=compare(gs, "ID"),
                            action_distance=compare(ad, "ID"),
                            q_aware={k: dict(median=float(np.median(v)), above=float(100 * (v > 0).mean())) for k, v in qa.items()},
                            q_awac={k: dict(median=float(np.median(v)), above=float(100 * (v > 0).mean())) for k, v in qg.items()},
                            q_scale=dict(aware=float(q0.abs().median()), awac=float(g0.abs().median())))
        R["eqm"].append(r)
        print(f"  seed {seed} ({time.time() - t0:.0f} s): states |f| K=20 " + ", ".join(
            f"{o} {r['states']['state']['|f| at K = 20'][o]['auroc']:.2f}" for o in oods) + " | actions |f| " + ", ".join(
            f"{k} {v['auroc']:.2f}" for k, v in r["actions"]["f_at_action"].items() if k != "ID"), flush=True)
    json.dump(dict(args=vars(a), results=R), open(out / "results.json", "w"), indent=1, default=float)
    (out / "summary.md").write_text(summary(a, R, oods, list(acts)))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


def fmt(v, d=2):
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def summary(a, R, oods, act_sets):
    E, B = R["eqm"], R["baseline"]
    bn = "distance to closest dataset state"
    L = [f"# OOD check on `{a.env}` (EqM: `{a.eqm_runs}`, Gaussian: `{a.gauss_runs}`, seeds {a.seeds})", "",
         "Evaluation only; script and definitions: `diagnostics/ood_halfcheetah.py`. Checkpoints: "
         + ", ".join(f"seed {r['seed']}: {r['checkpoint']} / {r['gauss_checkpoint']}" for r in E)
         + ". Cells: AUROC, mean ± std over seeds (0.5 = chance, 1 = perfect), and in brackets the % of the OOD set "
         "caught when 5 % of the ID set is flagged.", "",
         f"## States, one at a time ({a.n_win * acc.W} states per set)", "", "| score | " + " | ".join(oods) + " |",
         "|---|" + "---|" * len(oods)]
    for name in E[0]["states"]["state"]:
        L.append(f"| {name} | " + " | ".join(
            f"{fmt([r['states']['state'][name][o]['auroc'] for r in E])} "
            f"({fmt([r['states']['state'][name][o]['caught'] for r in E], 0)})" for o in oods) + " |")
    L.append(f"| {bn} | " + " | ".join(f"{B['state'][bn][o]['auroc']:.2f} ({B['state'][bn][o]['caught']:.0f})"
                                       for o in oods) + " |")
    L += ["", f"## States averaged over W consecutive states ({a.n_win} windows per set)", "",
          "| score | W | " + " | ".join(oods) + " |", "|---|---|" + "---|" * len(oods)]
    for name in ("|f| at K = 20", "sum, K = 50", "bucket two-sided, K = 50"):
        for w in acc.WS:
            L.append(f"| {name} | {w} | " + " | ".join(
                f"{fmt([r['states']['window'][name][w][o]['auroc'] for r in E])} "
                f"({fmt([r['states']['window'][name][w][o]['caught'] for r in E], 0)})" for o in oods) + " |")
    for w in acc.WS:
        L.append(f"| {bn} | {w} | " + " | ".join(
            f"{B['window'][bn][w][o]['auroc']:.2f} ({B['window'][bn][w][o]['caught']:.0f})" for o in oods) + " |")
    L += ["", "Median |f| trace (mean of 16 starts), seed 1, and median distance to the closest dataset state:", "",
          "| set | k=1 | 5 | 10 | 20 | 50 | 200 | 1000 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in E[0]["median_trace"].items():
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    aoods = [k for k in act_sets if k != "ID"]
    L += ["", f"## Actions at {a.n_act} training states", "", "| score | " + " | ".join(aoods) + " |",
          "|---|" + "---|" * len(aoods)]
    for name, key in (("EqM: |f| at the action", "f_at_action"), ("EqM: energy drop", "energy_drop"),
                      ("AWAC Gaussian: distance in std units", "gaussian"),
                      (f"distance to closest data action ({a.n_nn} closest states)", "action_distance")):
        L.append(f"| {name} | " + " | ".join(
            f"{fmt([r['actions'][key][o]['auroc'] for r in E])} ({fmt([r['actions'][key][o]['caught'] for r in E], 0)})"
            for o in aoods) + " |")
    L += ["", "Median per set (score values) and the critics:", "", "| | ID | " + " | ".join(aoods) + " |",
          "|---|---|" + "---|" * len(aoods)]
    for name, key in (("EqM |f|", "f_at_action"), ("AWAC Gaussian distance", "gaussian"),
                      ("distance to closest data action", "action_distance")):
        L.append(f"| {name} | " + " | ".join(fmt([r["actions"][key][k]["median"] for r in E]) for k in ["ID"] + aoods) + " |")
    for name, key in (("Aware critic: Q − Q(data action), median", "q_aware"), ("AWAC critic: Q − Q(data action), median", "q_awac")):
        L.append(f"| {name} | 0 | " + " | ".join(fmt([r["actions"][key][k]["median"] for r in E], 1) for k in aoods) + " |")
    for name, key in (("Aware critic: % rated above the data action", "q_aware"), ("AWAC critic: % rated above the data action", "q_awac")):
        L.append(f"| {name} | – | " + " | ".join(fmt([r["actions"][key][k]["above"] for r in E], 0) for k in aoods) + " |")
    L += ["", "Typical |Q(s, data action)|: Aware " + fmt([r["actions"]["q_scale"]["aware"] for r in E], 0)
          + ", AWAC " + fmt([r["actions"]["q_scale"]["awac"] for r in E], 0) + "."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
