"""Tables (markdown) from the JSON files of drift.py, averaged over seeds.

Files are grouped by name without the seed suffix, e.g. me_T1_s1.json, me_T1_s2.json -> me_T1.

    python diagnostics/summarize_drift.py diagnostics/out/drift [--mu 0.3]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", type=Path)
ap.add_argument("--mu", type=float, default=0.3, help="momentum for the eta x K tables (trained: 0.3)")
ap.add_argument("--eta_train", type=float, default=0.01)
ap.add_argument("--K_train", type=int, default=20)
args = ap.parse_args()

groups = defaultdict(list)
for f in sorted(args.folder.glob("*.json")):
    r = json.loads(f.read_text())
    # Early files stored the rollouts under "env" (overwriting the dataset name) and "awac_env"
    if isinstance(r.get("env"), list):
        r["rollouts"] = r.pop("env")
        r["env"] = re.search(r"halfcheetah-[a-z-]+?-v2", r["run"]).group(0)
    if "awac_env" in r:
        r["awac_rollouts"] = r.pop("awac_env")
    groups[re.sub(r"_s\d+$", "", f.stem)].append(r)


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def pooled(runs, part, key, **match):
    """values of `key` over all seeds (and episodes) of the entries matching `match`"""
    vals = []
    for r in runs:
        for e in r[part] if isinstance(r[part], list) else [r[part]]:
            if all(abs(e[m] - v) < 1e-9 for m, v in match.items()):
                v = e[key]
                vals += v if isinstance(v, list) else [v]
    return np.array(vals, dtype=float)


def ep_stats(runs, part, **match):
    ret = pooled(runs, part, "returns", **match)
    flip = pooled(runs, part, "flip_step", **match) >= 0
    q, g = pooled(runs, part, "q", **match), pooled(runs, part, "g", **match)
    d = dict(ret=ret.mean(), ret_seed_std=np.std([np.mean(pooled([r], part, "returns", **match)) for r in runs]),
             flip=flip.mean(), speed=pooled(runs, part, "speed", **match).mean(),
             outside=pooled(runs, part, "outside", **match).mean(), qg=q.mean() / g.mean(),
             state=pooled(runs, part, "state_dist", **match).mean(), act=pooled(runs, part, "act_dist", **match).mean(),
             n=len(ret))
    if "expert_region" in (runs[0][part][0] if isinstance(runs[0][part], list) else runs[0][part]):
        d["expert"] = pooled(runs, part, "expert_region", **match).mean()
    return d, ret, flip


def fmt_stats(d):
    return [f"{d['ret']:,.0f} ± {d['ret_seed_std']:,.0f}", f"{100 * d['flip']:.0f}%", f"{d['speed']:.1f}",
            f"{100 * d['outside']:.1f}%", f"{d['qg']:.2f}", f"{d['state']:.2f}", f"{d['act']:.2f}",
            f"{100 * d['expert']:.0f}%" if "expert" in d else "–"]


STATS_HEAD = ["return ± seed std", "flips", "speed", "outside box", "Q / return", "state dist", "action dist",
              "expert region"]

for name, runs in groups.items():
    r0 = runs[0]
    print(f"## {name}  ({len(runs)} seeds; {r0['env']}, T = {r0['temperature']}, weighting {r0['weighting']})\n")
    print(f"Data reference: actions at bounds {100 * np.mean([r['data']['at_bounds'] for r in runs]):.1f}%, "
          f"nearest-neighbour action distance {np.mean([r['data']['nn_dist'] for r in runs]):.3f}, "
          f"nearest-state distance {np.mean([r['data']['nn_state_dist'] for r in runs]):.3f}"
          + (f", neighbours of medium / expert states that are expert: "
             f"{100 * np.mean([r['data']['nb_expert_share_by_state']['medium'] for r in runs]):.1f}% / "
             f"{100 * np.mean([r['data']['nb_expert_share_by_state']['expert'] for r in runs]):.1f}%"
             if "nb_expert_share_by_state" in r0["data"] else "") + "\n")

    if "rollouts" in r0:
        etas = sorted({e["eta"] for e in r0["rollouts"]})
        Ks = sorted({e["K"] for e in r0["rollouts"]})
        mus = sorted({e["mu"] for e in r0["rollouts"]})
        for title, f in (("return", lambda d: f"{d['ret']:,.0f}"), ("flip rate", lambda d: f"{100 * d['flip']:.0f}%"),
                         ("actions outside the box (visited states)", lambda d: f"{100 * d['outside']:.0f}%"),
                         ("critic Q / actual discounted return (visited states)", lambda d: f"{d['qg']:.2f}")):
            print(f"**A: {title}, momentum {args.mu}** (rows η, columns K; trained: η = {args.eta_train}, K = {args.K_train})\n")
            rows = []
            for eta in etas:
                row = [f"{eta:g}"]
                for K in Ks:
                    d = ep_stats(runs, "rollouts", eta=eta, K=K, mu=args.mu)[0]
                    cell = f(d)
                    row.append(f"**{cell}**" if (abs(eta - args.eta_train) < 1e-9 and K == args.K_train) else cell)
                rows.append(row)
            table(["η \\ K"] + [str(K) for K in Ks], rows)

        # Best settings over the whole grid, and the trained one
        cells = []
        for eta in etas:
            for K in Ks:
                for mu in mus:
                    d = ep_stats(runs, "rollouts", eta=eta, K=K, mu=mu)[0]
                    cells.append(((eta, K, mu), d))
        cells.sort(key=lambda c: -c[1]["ret"])
        rows = []
        trained = ep_stats(runs, "rollouts", eta=args.eta_train, K=args.K_train, mu=0.3)[0]
        rows.append(["trained: η 0.01, K 20, μ 0.3", "0.2"] + fmt_stats(trained))
        for (eta, K, mu), d in cells[:5]:
            rows.append([f"best: η {eta:g}, K {K}, μ {mu:g}", f"{eta * K:g}"] + fmt_stats(d))
        if "awac_rollouts" in r0:
            rows.append(["Gaussian AWAC (mean action, own critic)", "–"] + fmt_stats(ep_stats(runs, "awac_rollouts")[0]))
        print("**A: trained setting, best settings of the grid, and AWAC** (per episode, pooled over seeds; "
              "state / action dist: ratio to the same distance for dataset states)\n")
        table(["setting", "η·K"] + STATS_HEAD, rows)

        # Flip vs no-flip episodes at the trained setting
        d, ret, flip = ep_stats(runs, "rollouts", eta=args.eta_train, K=args.K_train, mu=0.3)
        rows = []
        for lab, m in (("no flip", ~flip), ("flip", flip)):
            if m.any():
                sub = {key: pooled(runs, "rollouts", key, eta=args.eta_train, K=args.K_train, mu=0.3)[m]
                       for key in ("returns", "q", "g", "outside", "state_dist", "act_dist", "speed")}
                exp = (pooled(runs, "rollouts", "expert_region", eta=args.eta_train, K=args.K_train, mu=0.3)[m]
                       if "expert" in d else None)
                rows.append([lab, int(m.sum()), f"{sub['returns'].mean():,.0f}", f"{sub['speed'].mean():.1f}",
                             f"{100 * sub['outside'].mean():.1f}%", f"{sub['q'].mean() / sub['g'].mean():.2f}",
                             f"{sub['state_dist'].mean():.2f}", f"{sub['act_dist'].mean():.2f}",
                             f"{100 * exp.mean():.0f}%" if exp is not None else "–"])
        print("**A: episodes at the trained setting, by flip**\n")
        table(["episodes", "n", "return", "speed", "outside box", "Q / return", "state dist", "action dist",
               "expert region"], rows)

        # Across all settings: does the flip rate follow the outside-box share?
        fr = np.array([c[1]["flip"] for c in cells])
        ob = np.array([c[1]["outside"] for c in cells])
        rt = np.array([c[1]["ret"] for c in cells])
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        print(f"Across all {len(cells)} settings: Spearman(flip rate, outside box) = "
              f"{np.corrcoef(rank(fr), rank(ob))[0, 1]:.2f}; Spearman(return, outside box) = "
              f"{np.corrcoef(rank(rt), rank(ob))[0, 1]:.2f}\n")

    # B: dataset states
    acts = r0["act"]
    etas_b = sorted({e["eta"] for e in acts})
    Ks_b = sorted({e["K"] for e in acts})

    def b(key, **m):
        return np.mean(pooled(runs, "act", key, **m))

    print(f"**B: dataset states, η = {args.eta_train}, momentum {args.mu}, over K** "
          "(nn ratio: distance to the nearest neighbour action / same for the data action; "
          "ΔQ: Q(sample) − Q(data action))\n")
    rows = []
    for K in Ks_b:
        m = dict(eta=args.eta_train, mu=args.mu, K=K)
        rows.append([K, f"{b('f_norm', **m):.3f}", f"{b('move', **m):.4f}", f"{b('spread', **m):.3f}",
                     f"{100 * b('outside', **m):.1f}%", f"{b('nn_ratio', **m):.2f}", f"{b('dq', **m):+.1f}",
                     f"{b('awac_dq', **m):+.1f}" if "awac_dq" in acts[0] else "–"])
    table(["K", "‖f‖", "move / step", "spread", "outside box", "nn ratio", "ΔQ Aware critic", "ΔQ AWAC critic"], rows)

    print(f"**B: dataset states, K = {args.K_train}, momentum {args.mu}, over η**\n")
    rows = []
    for eta in etas_b:
        m = dict(eta=eta, mu=args.mu, K=args.K_train)
        rows.append([f"{eta:g}", f"{b('f_norm', **m):.3f}", f"{b('move', **m):.4f}", f"{b('spread', **m):.3f}",
                     f"{100 * b('outside', **m):.1f}%", f"{b('nn_ratio', **m):.2f}", f"{b('dq', **m):+.1f}",
                     f"{b('awac_dq', **m):+.1f}" if "awac_dq" in acts[0] else "–"])
    table(["η", "‖f‖", "move / step", "spread", "outside box", "nn ratio", "ΔQ Aware critic", "ΔQ AWAC critic"], rows)

    if "awac_policy" in r0:
        ap_ = lambda key: np.mean([r["awac_policy"][key] for r in runs])  # noqa: E731
        print(f"AWAC policy on the same dataset states: outside box {100 * ap_('outside'):.1f}%, "
              f"nn ratio {ap_('nn_ratio'):.2f}, ΔQ Aware critic {ap_('dq'):+.1f}, ΔQ AWAC critic {ap_('awac_dq'):+.1f}\n")
