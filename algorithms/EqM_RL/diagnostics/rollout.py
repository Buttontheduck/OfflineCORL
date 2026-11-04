"""Roll out one trained Minari run and record return, flips and speed (EqM_RL.md,
Experiment 2, diagnostic C). A flip is HalfCheetah's body angle obs[1] beyond 2 rad;
obs[8] is the forward speed.

By default the run is an Aware_Minari run (one EqM sample per step, K sampler steps). With
--awac it is a Gaussian AWAC run of Awac_Minari.py, evaluated with its mean action as
Awac_Minari.py does (EqM_RL.md, "Planned next experiments", section 5b).

    python diagnostics/rollout.py <run checkpoint dir> diagnostics/out/roll/<name>_K20.json --K 20
    python diagnostics/rollout.py <Awac_Minari run dir> diagnostics/out/roll/<name>_awac.json --awac --n_eps 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from common import EQM_RL, Policy, latest_checkpoint

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path, help="checkpoint folder of one run (has config.yaml)")
ap.add_argument("out", type=Path, help="output JSON")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--awac", action="store_true", help="the run is an Awac_Minari.py (Gaussian AWAC) run")
ap.add_argument("--K", type=int, default=20, help="sampler steps (Aware runs)")
ap.add_argument("--n_eps", type=int, default=10, help="episodes (seeds seed0, seed0 + 1, ...)")
ap.add_argument("--seed0", type=int, default=1000, help="env seed of the first episode")
ap.add_argument("--max_steps", type=int, default=None, help="episode length cap (default: the env's own)")
ap.add_argument("--flip_angle", type=float, default=2.0, help="|body angle| in rad that counts as a flip")
args = ap.parse_args()

torch.manual_seed(0)
if args.awac:
    sys.path.insert(0, str(EQM_RL))
    from Awac_Minari import Actor, compute_mean_std  # noqa: E402
    from minari_utils import prepare_minari_data  # noqa: E402

    cfg = yaml.safe_load(open(args.run_dir / "config.yaml"))
    _, data, minari_dataset = prepare_minari_data(cfg["env_name"])
    mean, std = compute_mean_std(data["observations"], eps=1e-3)  # as in Awac_Minari.py
    state_dim, action_dim = data["observations"].shape[1], data["actions"].shape[1]
    ckpt = args.run_dir / args.checkpoint if args.checkpoint else latest_checkpoint(args.run_dir)
    actor = Actor(state_dim, action_dim, cfg["hidden_dim"]).to("cuda").eval()
    actor.load_state_dict(torch.load(ckpt, map_location="cuda")["actor"])
    env0 = minari_dataset.recover_environment()
    low, high = float(env0.action_space.low[0]), float(env0.action_space.high[0])

    @torch.no_grad()
    def act(o):
        s = torch.tensor((o - mean) / std, dtype=torch.float32, device="cuda")
        return actor._mlp(s).clamp(low, high).cpu().numpy()  # the Gaussian mean

    info = dict(run=args.run_dir.name, checkpoint=ckpt.name, env=cfg["env_name"].split("/")[-1],
                seed=cfg["seed"], method="awac", awac_lambda=cfg.get("awac_lambda"))
else:
    pol = Policy(args.run_dir, args.checkpoint)
    minari_dataset = pol.minari_dataset

    def act(o):
        return pol.sample(pol.states(o), args.K)[0].clamp(pol.low, pol.high).cpu().numpy()

    info = dict(pol.info(), method="aware", K=args.K)

envs = [minari_dataset.recover_environment() for _ in range(args.n_eps)]
o = [e.reset(seed=args.seed0 + i)[0] for i, e in enumerate(envs)]
n = args.n_eps
ret, done = np.zeros(n), np.zeros(n, bool)
flip_step, ret_at_flip = np.full(n, -1), np.full(n, np.nan)
speeds = [[] for _ in range(n)]
t = 0
while not done.all() and (args.max_steps is None or t < args.max_steps):
    a = act(np.stack(o))
    for i, e in enumerate(envs):
        if done[i]:
            continue
        o[i], r, term, trunc, _ = e.step(a[i])
        ret[i] += r
        speeds[i].append(o[i][8])
        if flip_step[i] < 0 and abs(o[i][1]) > args.flip_angle:
            flip_step[i], ret_at_flip[i] = t, ret[i]
        done[i] = term or trunc
    t += 1

res = dict(info, returns=ret.tolist(), flip_step=flip_step.tolist(),
           ret_at_flip=ret_at_flip.tolist(),
           speed_before_flip=[float(np.mean(sp[:fs] if fs > 0 else sp)) for sp, fs in zip(speeds, flip_step)])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res))
print("done", args.run_dir.name, info["method"], f"flips {np.mean(flip_step >= 0):.0%}", f"return {ret.mean():.0f}")
