"""Two-group check (EqM_RL.md, "Experiment 9 follow-up"): on states whose nearby data actions form two
separated groups, do the EqM samples at a given K split between the groups like the data?

A state is two-group when 2-means on its 128 neighbour actions gives a smaller group >= 20 % and
centres more than 3 within-group stds apart. Per K: share of those states where the samples (clipped)
put >= 10 % in each group ('covered'), mean |sample share - data share| ('split error'), and the same
for a round Gaussian with the samples' centre and spread (the null: width without shape).
Uses the four checkpoints of maze2d_mode_figures.py and the states of its npz files.
Run from algorithms/EqM_RL/ in the hawk env: python diagnostics/maze2d_two_groups.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Policy  # noqa: E402
from maze2d_mode_figures import CASES, C  # noqa: E402
from maze2d_modes import pick_checkpoint  # noqa: E402

KS = [5, 10, 15, 20, 25, 30, 100, 1000]
NPZ = Path(__file__).resolve().parent / "out/maze2d_modes/figures/T1_s1_K20_30_100.npz"


def two_means(x, it=30):
    m = x.mean(0)
    c1 = x[np.argmax(((x - m) ** 2).sum(1))]
    c2 = x[np.argmax(((x - c1) ** 2).sum(1))]
    cen = np.stack([c1, c2])
    for _ in range(it):
        lab = ((x[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        cen = np.stack([x[lab == k].mean(0) if (lab == k).any() else cen[k] for k in (0, 1)])
    return cen, lab


def assign(pts, cen):
    return ((pts[:, None] - cen[None]) ** 2).sum(-1).argmin(1)


def main():
    rng = np.random.default_rng(0)
    z = np.load(NPZ)
    D, idx = z["data_actions_nn128"], z["state_index"]
    groups = []
    for i in range(len(D)):
        cen, lab = two_means(D[i])
        share, s = lab.mean(), np.sqrt(((D[i] - cen[lab]) ** 2).sum(1).mean() / 2)
        if min(share, 1 - share) >= 0.2 and np.linalg.norm(cen[0] - cen[1]) > 3 * s:
            groups.append((i, cen, share))
    print(f"two-group states: {len(groups)} of {len(D)} (smaller group {np.mean([min(g[2], 1 - g[2]) for g in groups]):.2f} on average)")
    pol = Policy(C / CASES[0][2], checkpoint=pick_checkpoint(C / CASES[0][2], CASES[0][3]))
    s = pol.states(pol.data["observations"][idx])
    for cid, _, run, ck, _ in CASES:
        pol.model.load_state_dict(torch.load(C / run / pick_checkpoint(C / run, ck), map_location=pol.device)["model"])
        pol.model.eval()
        torch.manual_seed(0)
        _, snaps = pol.sample(s.repeat_interleave(256, 0), max(KS), snaps=set(KS))
        cells = []
        for K in KS:
            X = snaps[K].view(len(s), 256, 2).clamp(pol.low, pol.high).cpu().numpy()
            err, cov, cov0 = [], [], []
            for i, cen, share in groups:
                p = assign(X[i], cen).mean()
                g = np.clip(X[i].mean(0) + X[i].std(0).mean() * rng.standard_normal(X[i].shape), pol.low, pol.high)
                p0 = assign(g, cen).mean()
                err.append(abs(p - share)); cov.append(min(p, 1 - p) >= 0.1); cov0.append(min(p0, 1 - p0) >= 0.1)
            cells.append(f"K={K}: {100 * np.mean(cov):.0f}% ({100 * np.mean(cov0):.0f}%) / {np.mean(err):.2f}")
        print(f"{cid:13s} covered (null) / split error   " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
