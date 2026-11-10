"""Experiment 19: would an OOD score of the EqM actor help against the critic's false peak? Markdown to stdout.

Background (IFQL.md, Section 12.7): in seed 1 from 800k and in seed 3 at 1M, the critic's V is too high in a side
corridor of the maze (cells (4,2), (4,1), (5,1)), and the policy walks there instead of to the goal.

Two OOD scores of the EqM field f, as in diagnostics/ood_check.py and ood_accumulate.py. The sampler makes a candidate
by K = 20 NAG steps from noise, x_1 .. x_20:
  last+1        |f(x_20 | s)|: the norm of the field one evaluation after the last step (GeCO's score)
  accumulated   |f(x_1 | s)| + ... + |f(x_20 | s)|: the sum of the norms over all steps
Each candidate has its own two scores. A state's score is the average over its 32 candidates.
Baseline for states: the distance to the nearest dataset state (normalized states).

Part A, measured during rollouts without any filter (frozen EqM-BC actor of the seed, best of 32 by the critic):
  A1  states: the share of visited states whose score is above the 95th percentile of held-out dataset states (OGBench's
      val split; 5 % of those are above it by construction). Given for all visited states and for those in the corridor.
  A2  actions: where the chosen candidate's score lies among the 32 candidates (0 = lowest score, 1 = highest;
      0.5 on average if the critic's choice has nothing to do with the score).
Part B, the direct test: the same rollouts, but only the --keep candidates with the lowest score are offered to the
critic. If OOD helped, the bad critics' success would rise.

    python diagnostics/exp19_ood_check.py > diagnostics/out/exp19_ood_check/summary.md
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EQM_RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQM_RL / "Aware"))
import ogbench_data  # noqa: E402
import train_ieqm as T  # noqa: E402
from utils.networks import ConditionalMLP  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--env_name", default="humanoidmaze-medium-navigate-singletask-v0")
ap.add_argument("--root", default="/workspace/Trained_Models/checkpoints")
ap.add_argument("--prefix", default="Exp19_IEQM")
ap.add_argument("--critics", default="1:700000,1:800000,1:900000,3:900000,3:1000000", help="seed:step pairs")
ap.add_argument("--corridor", default="4:2,4:1,5:1,6:1", help="the cells of the false peak")
ap.add_argument("--keep", type=int, default=16, help="part B: candidates offered to the critic (of 32)")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--n_ref", type=int, default=5000, help="held-out dataset states for the 95th percentiles")
ap.add_argument("--test_seed", type=int, default=69)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
dev = args.device
critics = [tuple(int(x) for x in c.split(":")) for c in args.critics.split(",")]
corridor = {tuple(int(x) for x in c.split(":")) for c in args.corridor.split(",")}

import ogbench  # noqa: E402

env, train, val = ogbench.make_env_and_datasets(args.env_name, dataset_dir=ogbench_data.OGBENCH_DATA)
u = env.unwrapped
obs = train["observations"]
mean, std = obs.mean(0), obs.std(0) + 1e-3
state_dim, action_dim = obs.shape[1], train["actions"].shape[1]
low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
std_t = torch.tensor(std, dtype=torch.float32, device=dev)
S_train = (torch.tensor(obs, dtype=torch.float32, device=dev) - mean_t) / std_t


def run_dir(seed: int) -> str:
    d = glob.glob(f"{args.root}/{args.prefix}_env_name{args.env_name}_seed{seed}-{args.env_name}-*")
    assert len(d) == 1, f"expected one run for seed {seed}, found {d}"
    return d[0]


def load(seed: int, step: int):
    """-> critic of the checkpoint, the seed's frozen EqM field, its step size and momentum, K, N."""
    cfg = yaml.safe_load(open(f"{run_dir(seed)}/config.yaml"))
    ck = torch.load(f"{run_dir(seed)}/checkpoint_{step}.pt", map_location=dev)
    critic = T.IFQLCritic(state_dim, action_dim, cfg["critic_hidden_dim"], cfg["critic_n_hidden"]).to(dev)
    critic.load_state_dict(ck["critic"])
    bc = cfg["frozen_bc_config"]
    assert bc["actor"]["opt_type"] == "nag"
    model = ConditionalMLP(state_dim + action_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"]).to(dev)
    model.load_state_dict(torch.load(f"{cfg['frozen_bc_run']}/{cfg['bc_checkpoint']}", map_location=dev)["model"])
    return (critic.eval().requires_grad_(False), model.eval().requires_grad_(False), float(bc["actor"]["step_size"]),
            float(bc["actor"]["moment"]), int(cfg["num_step"]), int(cfg["eval_candidates"]))


@torch.no_grad()
def propose(model, s_norm, N, K, eta, mu, gen):
    """Actor._descend (NAG) from noise, N candidates per state. -> candidates (B, N, A) clipped to the action box,
    last+1 score (B, N), accumulated score (B, N)."""
    st = s_norm.repeat_interleave(N, 0)
    x = torch.randn(len(st), action_dim, device=dev, generator=gen)
    mom, acc = torch.zeros_like(x), torch.zeros(len(st), device=dev)
    for _ in range(K):
        mom = model(x - eta * mom * mu, st)
        x = x - eta * mom
        fk = model(x, st).norm(dim=1)                 # |f(x_k | s)|
        acc += fk
    B = len(s_norm)
    return x.clamp(low, high).view(B, N, action_dim), fk.view(B, N), acc.view(B, N)


@torch.no_grad()
def nearest(q_norm: torch.Tensor) -> np.ndarray:
    """Distance of each normalized query state to its nearest training state."""
    out = []
    for a in range(0, len(q_norm), 1000):
        q, best = q_norm[a:a + 1000], None
        for b in range(0, len(S_train), 200000):
            d = torch.cdist(q, S_train[b:b + 200000]).min(1).values
            best = d if best is None else torch.minimum(best, d)
        out.append(best)
    return torch.cat(out).cpu().numpy()


def rollout(seed: int, step: int, filt: str, envs: list) -> dict:
    """filt: 'none', 'last' or 'acc' (offer only the args.keep candidates with the lowest score to the critic)."""
    critic, model, eta, mu, K, N = load(seed, step)
    gen = torch.Generator(device=dev).manual_seed(seed + 2000)
    n = len(envs)
    glob_rng = np.random.get_state()
    ob = np.stack([ogbench_data.seeded_reset(e, args.test_seed + 1000 * k + j)
                   for j, (k, e) in enumerate(envs)]).astype(np.float64)
    done, success, length = np.zeros(n, bool), np.zeros(n), np.zeros(n, int)
    rec = dict(last=[], acc=[], rank_last=[], rank_acc=[], xy=[], keep=[])
    kept_states, t = [], 0
    with torch.no_grad():
        while not done.all():
            act = np.flatnonzero(~done)
            s = torch.tensor(ob[act], dtype=torch.float32, device=dev)
            s_norm = (s - mean_t) / std_t
            cand, last, acc = propose(model, s_norm, N, K, eta, mu, gen)
            q = critic(s[:, None].expand(-1, N, -1), cand)
            if filt != "none":
                score = last if filt == "last" else acc
                kth = score.kthvalue(args.keep, dim=1, keepdim=True).values
                q = q.masked_fill(score > kth, float("-inf"))
            best = q.argmax(1)
            ar = torch.arange(len(s), device=dev)
            rec["last"].append(last.mean(1).cpu().numpy())
            rec["acc"].append(acc.mean(1).cpu().numpy())
            rec["rank_last"].append((last < last[ar, best][:, None]).float().mean(1).cpu().numpy() * N / (N - 1))
            rec["rank_acc"].append((acc < acc[ar, best][:, None]).float().mean(1).cpu().numpy() * N / (N - 1))
            rec["xy"].append(ob[act, :2].copy())
            sub = np.zeros(len(act), bool)
            if t % 20 == 0:                           # every 20th step: states kept for the nearest-state distance
                kept_states.append(s_norm.clone())
                sub[:] = True
            rec["keep"].append(sub)
            a = cand[ar, best].cpu().numpy()
            for m, i in enumerate(act):
                o, _, term, trunc, info = envs[i][1].step(a[m])
                ob[i] = o
                length[i] += 1
                if term or trunc:
                    done[i], success[i] = True, float(info["success"])
            t += 1
    np.random.set_state(glob_rng)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    cells = np.array([u.xy_to_ij(p) for p in out["xy"]], dtype=int)
    out["corridor"] = np.array([tuple(c) in corridor for c in cells])
    out["dist"] = nearest(torch.cat(kept_states))     # for the rows where out["keep"] is True
    out.update(success=100 * success.mean(), length=length.mean())
    return out


# ---------------------------------------------------------------- reference: held-out dataset states (95th percentiles)
rng = np.random.default_rng(0)
vi = rng.choice(len(val["observations"]), args.n_ref, replace=False)
val_norm = (torch.tensor(val["observations"][vi], dtype=torch.float32, device=dev) - mean_t) / std_t
ref = {}
for seed in sorted({c[0] for c in critics}):
    _, model, eta, mu, K, N = load(seed, critics[0][1] if (seed, critics[0][1]) in critics else
                                   [c[1] for c in critics if c[0] == seed][0])
    gen = torch.Generator(device=dev).manual_seed(123)
    last, acc = [], []
    for a in range(0, len(val_norm), 1000):
        _, l, c = propose(model, val_norm[a:a + 1000], N, K, eta, mu, gen)
        last.append(l.mean(1))
        acc.append(c.mean(1))
    last, acc = torch.cat(last).cpu().numpy(), torch.cat(acc).cpu().numpy()
    ref[seed] = dict(last=np.percentile(last, 95), acc=np.percentile(acc, 95), last_med=np.median(last),
                     acc_med=np.median(acc))
dist_ref = nearest(val_norm)
d95, d_med = np.percentile(dist_ref, 95), np.median(dist_ref)

tasks = ogbench_data.parse_tasks("own", env)
envs = ogbench_data.make_eval_envs(args.env_name, tasks, args.episodes)
R = {}
for seed, step in critics:
    for filt in ("none", "last", "acc"):
        R[(seed, step, filt)] = rollout(seed, step, filt, envs)
        print(f"done seed {seed} {step} {filt}: {R[(seed, step, filt)]['success']:.0f} %", file=sys.stderr, flush=True)

# ---------------------------------------------------------------- report
print("# Experiment 19: would an OOD score of the EqM actor help against the critic's false peak?\n")
print(f"`{args.env_name}`, runs `{args.prefix}`. Each rollout: {args.episodes} episodes of task 1, the seed's frozen EqM-BC "
      "actor proposes 32 candidates at K = 20, the critic of the named checkpoint picks one. **last+1** = |f| one "
      "evaluation after the last sampler step. **accumulated** = the sum of |f| over the 20 steps. A state's score is the "
      "average over its 32 candidates. **distance** = distance to the nearest dataset state (normalized states).\n")
print(f"Reference: {args.n_ref:,} held-out dataset states (OGBench's val split). Their median and 95th percentile: "
      + "; ".join(f"seed {sd}: last+1 {r['last_med']:.3f} / {r['last']:.3f}, accumulated {r['acc_med']:.1f} / {r['acc']:.1f}"
                  for sd, r in ref.items()) + f"; distance {d_med:.2f} / {d95:.2f}.\n")

print("## A1. Are the visited states flagged as OOD?\n")
print("Share of visited states with a score above the 95th percentile of the held-out dataset states (5 % = like the "
      "data). **corridor** = only the visited states in the cells " + ", ".join(f"({i},{j})" for i, j in sorted(corridor))
      + " (the false peak). The distance is computed on every 20th step.\n")
print("| seed | critic of | success | steps in the corridor | last+1: all / corridor | accumulated: all / corridor | "
      "distance: all / corridor |")
print("|---|---|---|---|---|---|---|")
for seed, step in critics:
    r = R[(seed, step, "none")]
    c, k = r["corridor"], r["keep"]

    def share(x, thr, m):
        return f"{100 * (x[m] > thr).mean():.0f} %" if m.sum() else "–"

    ck = c[k]
    print(f"| {seed} | {step // 1000}k | {r['success']:.0f} % | {100 * c.mean():.0f} % | "
          f"{share(r['last'], ref[seed]['last'], np.ones_like(c))} / {share(r['last'], ref[seed]['last'], c)} | "
          f"{share(r['acc'], ref[seed]['acc'], np.ones_like(c))} / {share(r['acc'], ref[seed]['acc'], c)} | "
          f"{share(r['dist'], d95, np.ones_like(ck))} / {share(r['dist'], d95, ck)} |")

print("\n## A2. Does the critic choose candidates with a high OOD score?\n")
print("Position of the chosen candidate's score among the 32 candidates of its state: 0 = the lowest score, 1 = the "
      "highest, 0.5 = no relation. Mean over all steps; **corridor** = steps in the corridor cells only.\n")
print("| seed | critic of | last+1: all / corridor | accumulated: all / corridor |")
print("|---|---|---|---|")
for seed, step in critics:
    r = R[(seed, step, "none")]
    c = r["corridor"]
    f = lambda x: f"{x.mean():.2f} / " + (f"{x[c].mean():.2f}" if c.sum() else "–")  # noqa: E731
    print(f"| {seed} | {step // 1000}k | {f(r['rank_last'])} | {f(r['rank_acc'])} |")

print(f"\n## B. Does filtering the candidates by the OOD score rescue the policy?\n")
print(f"Only the {args.keep} of 32 candidates with the lowest score are offered to the critic. Success in % "
      f"({args.episodes} episodes; one such eval is noisy by about ±10 points), and in brackets the share of all steps "
      "spent in the corridor.\n")
print("| seed | critic of | no filter | filter by last+1 | filter by accumulated |")
print("|---|---|---|---|---|")
for seed, step in critics:
    cell = lambda f: f"{R[(seed, step, f)]['success']:.0f} % ({100 * R[(seed, step, f)]['corridor'].mean():.0f} %)"  # noqa: E731
    print(f"| {seed} | {step // 1000}k | {cell('none')} | {cell('last')} | {cell('acc')} |")
