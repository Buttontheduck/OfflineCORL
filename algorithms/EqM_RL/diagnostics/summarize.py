"""Print the tables of EqM_RL.md, Experiment 2, from the JSON files that diag.py and
rollout.py wrote (means over seeds).

    python diagnostics/summarize.py --diag diagnostics/out/diag --roll diagnostics/out/roll
"""

import argparse
import collections
import glob
import json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--diag", help="folder with diag.py JSON files")
ap.add_argument("--roll", help="folder with rollout.py JSON files")
args = ap.parse_args()
pd.set_option("display.width", 250)

if args.diag:
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{args.diag}/*.json"))]
    d = pd.DataFrame([dict(env=r["env"], lam=r["lam"], seed=r["seed"], K=int(K), ret=np.mean(v), worst=np.min(v))
                      for r in R for K, v in r["returns"].items()])
    g = d.groupby(["env", "lam", "K"])
    print("A: mean return vs sampler steps K (mean over seeds of the mean over episodes)")
    print(g.ret.mean().unstack("K").round(0).to_string())
    print("\nA: std over seeds")
    print(g.ret.std().unstack("K").round(0).to_string())
    print("\nA: worst single episode")
    print(g.worst.min().unstack("K").round(0).to_string())

    a = pd.DataFrame([dict(env=r["env"], lam=r["lam"], seed=r["seed"], K=int(K), **v)
                      for r in R for K, v in r["act"].items()])
    for m in ["f_norm", "spread", "rmse", "bias", "clip", "sat", "move_from_prev"]:
        print(f"\nB: {m}")
        print(a.groupby(["env", "lam", "K"])[m].mean().unstack("K").round(3).to_string())
    ref = pd.DataFrame([dict(env=r["env"], lam=r["lam"], f_at_data=r["f_at_data"], f_at_uniform=r["f_at_uniform"],
                             data_cond_std=r["data_cond_std_nn"], nn_state_dist=r["nn_state_dist_median"],
                             data_sat=r["data_sat_frac"], data_marginal_std=r["data_action_std_marginal"])
                        for r in R])
    print("\nB: references (field at dataset / uniform actions; dataset spread and saturation)")
    print(ref.groupby(["env", "lam"]).mean().round(3).to_string())

if args.roll:
    G = collections.defaultdict(list)
    for f in glob.glob(f"{args.roll}/*.json"):
        r = json.load(open(f))
        G[(r["env"], r["lam"], r["K"])].append(r)
    print(f"\nC: rollouts\n{'setting':26s} {'eps':>3s} {'return':>7s} {'min':>6s} {'flips':>6s} "
          f"{'ret no flip':>11s} {'ret flip':>9s} {'speed*':>7s}")
    for k in sorted(G):
        Rt = np.concatenate([r["returns"] for r in G[k]])
        F = np.concatenate([r["flip_step"] for r in G[k]]) >= 0
        sp = np.concatenate([r["speed_before_flip"] for r in G[k]])
        nf = f"{Rt[~F].mean():.0f}" if (~F).any() else "-"
        ff = f"{Rt[F].mean():.0f}" if F.any() else "-"
        name = f"{k[0]} lam={k[1]} K={k[2]}"
        print(f"{name:26s} {len(Rt):3d} {Rt.mean():7.0f} {Rt.min():6.0f} {F.mean() * 100:5.0f}% {nf:>11s} {ff:>9s} "
              f"{sp.mean():7.2f}")
    print("* mean forward speed (m/s) before the first flip, or over the whole episode without one")
