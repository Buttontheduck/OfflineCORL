"""Action diagnostics for Maze2D runs (Experiments 8 and 9): does the EqM sampler collapse, cover
the data's actions, or land between them, and how does the number of sampler steps K change that
and the score? Evaluation only; nothing is trained.

Maze2D actions are 2-D, so the samples can be compared with the data directly and plotted.

For one Aware run (a checkpoint folder) and each requested checkpoint:

  A. Samples on dataset states. 256 fixed dataset states (seed 0, so the same states for every run
     on a dataset), 256 samples each from N(0, I), with snapshots at each K of --K (one sampler
     trajectory per sample, so K = 1000 is where the K = 20 sample ends up when it runs on). The
     data reference for a state is the actions of its 128 nearest dataset states (normalized state
     space), split by rank into halves A and B of 64. Per K:
       f_end        mean ||f|| at the samples (about 0 = converged, an equilibrium of the field)
       spread       per-dim std across one state's samples, clipped (as the logged sampler/spread)
       outside      share of sample dims outside the box before clipping
       at_bound     share of clipped sample dims at +-1 (data: share of data dims at +-1)
       n_modes      clusters of the state's samples (radius 0.05) holding >= 5 % of them
       top_mass     share of the samples in the largest cluster (1 = all samples at one action)
       recall       mean distance from each data action (B) to the nearest of 64 samples:
                    large = parts of the data's actions are never sampled
       precision    mean distance from each of 64 samples to the nearest data action (B):
                    large = samples where the data has no actions (e.g. between modes)
       energy       energy distance between the 64 samples and B (0 = same distribution)
       off_support  share of samples farther from B than the 95th percentile of A -> B distances
     The same four distances between the data halves A and B are the reference (what a sampler
     that draws from the data itself would get; off_support is 5 % there by construction).
     With --awac, a Gaussian AWAC run of the same dataset and seed: its mean action (clipped)
     gets precision, off_support and at_bound.

  B. Rollouts. --episodes episodes side by side for each K of --rollout-K, with the run's own
     sampler otherwise (eta, momentum; one sample per step, clipped), env seeds test_seed + i as in
     Aware/main.py. D4RL normalized score, episode std, share of executed dims at +-1, and the mean
     action change between consecutive steps (|a_t - a_t-1|, a measure of jitter).

  An SVG per checkpoint: for 6 states (3 random, 3 with the widest data spread), the data's
  actions (grey) and the samples at each K (blue), the AWAC mean action (red cross).

Run from algorithms/EqM_RL/ in the hawk env (see diagnostics/README.md):
  python diagnostics/maze2d_modes.py <aware checkpoint folder> <out.json> [--awac <awac folder>]
         [--checkpoints 500000 last] [--K 20 30 100 1000] [--rollout-K 20 30 100]
Tables over runs: python diagnostics/summarize_maze2d_modes.py <folder of the json files>
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AwacPolicy, Policy  # noqa: E402

N_STATES, N_SAMPLES, N_NEIGH, N_CMP = 256, 256, 128, 64
MODE_RADIUS, MODE_MIN_MASS = 0.05, 0.05


def available(run_dir: Path) -> dict:
    return {int(re.findall(r"\d+", p.stem)[-1]): p.name for p in run_dir.glob("checkpoint_*.pt")}


def pick_checkpoint(run_dir: Path, want: str) -> str:
    """'last', or the checkpoint whose step is nearest to the requested one."""
    av = available(run_dir)
    step = max(av) if want == "last" else min(av, key=lambda s: abs(s - int(want)))
    return av[step]


def nearest_data(pol: Policy, idx: np.ndarray):
    """Actions of the N_NEIGH nearest dataset states of each chosen state -> (A, B, radius)."""
    S = pol.states(pol.data["observations"])
    acts = torch.tensor(pol.data["actions"], dtype=torch.float32, device=pol.device)
    ii, dd = [], []
    for chunk in S[idx].split(16):
        d, i = torch.cdist(chunk, S).topk(N_NEIGH, largest=False)
        ii.append(i)
        dd.append(d[:, -1])
    na = acts[torch.cat(ii)]  # (N, 128, 2), sorted by distance
    return na[:, 0::2], na[:, 1::2], torch.cat(dd).median().item()


def energy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy distance per state between point sets x (N, n, d) and y (N, m, d)."""
    def mean_offdiag(z):
        n = z.shape[1]
        return torch.cdist(z, z).sum((1, 2)) / (n * (n - 1))
    return 2 * torch.cdist(x, y).mean((1, 2)) - mean_offdiag(x) - mean_offdiag(y)


def modes(x: torch.Tensor):
    """Greedy clustering of each state's samples (N, n, 2): repeatedly take the point with the most
    unassigned neighbours within MODE_RADIUS as a cluster. -> n_modes, top_mass per state."""
    N, n, _ = x.shape
    near = torch.cdist(x, x) <= MODE_RADIUS
    free = torch.ones(N, n, dtype=torch.bool, device=x.device)
    masses = []
    for _ in range(20):
        cnt = (near & free[:, None, :]).sum(-1).masked_fill(~free, -1)  # free neighbours of each free point
        c = cnt.argmax(-1)
        members = near[torch.arange(N), c] & free & (cnt.max(-1).values > 0)[:, None]
        masses.append(members.sum(-1).float() / n)
        free &= ~members
        if not free.any():
            break
    m = torch.stack(masses, -1)
    return (m >= MODE_MIN_MASS).sum(-1).float(), m.max(-1).values


def compare(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, thr: float) -> dict:
    """Distances between point sets x (N, 64, 2) and the data half B, and at_bound of x."""
    to_b = torch.cdist(x, B).min(-1).values
    return dict(recall=torch.cdist(B, x).min(-1).values.mean().item(), precision=to_b.mean().item(),
                energy=energy(x, B).mean().item(), off_support=(to_b > thr).float().mean().item())


@torch.no_grad()
def sampler_block(pol, s, A, B, Ks, thr) -> dict:
    torch.manual_seed(0)
    s_rep = s.repeat_interleave(N_SAMPLES, 0)
    _, snaps = pol.sample(s_rep, max(Ks), snaps=set(Ks))
    out = {}
    for K in Ks:
        x = snaps[K]
        f = pol.model(x, s_rep).norm(dim=-1)
        x = x.view(len(s), N_SAMPLES, -1)
        xc = x.clamp(pol.low, pol.high)
        n_modes, top = modes(x)
        out[K] = dict(f_end=f.mean().item(), f_end_median=f.median().item(),
                      spread=xc.std(1).mean().item(), outside=((x < pol.low) | (x > pol.high)).float().mean().item(),
                      at_bound=(xc.abs() >= 0.999).float().mean().item(),
                      n_modes=n_modes.mean().item(), n_modes_1=(n_modes == 1).float().mean().item(),
                      top_mass=top.mean().item(), **compare(xc[:, :N_CMP], A, B, thr))
        out[K]["_xy"] = xc  # for the SVG, dropped before saving
    return out


@torch.no_grad()
def rollouts(pol, K: int, episodes: int) -> dict:
    torch.manual_seed(K)
    envs = [pol.make_env() for _ in range(episodes)]
    seed = pol.cfg.get("test_seed", 0)
    obs = [pol.reset(e, seed + i) for i, e in enumerate(envs)]
    ret, done = np.zeros(episodes), np.zeros(episodes, bool)
    prev = np.full((episodes, pol.action_dim), np.nan)
    dact, bound = [], []
    while not done.all():
        act_i = np.flatnonzero(~done)
        a = pol.sample(pol.states(np.stack([obs[i] for i in act_i])), K)[0]
        a = a.clamp(pol.low, pol.high).cpu().numpy()
        bound.append(np.abs(a) >= 0.999)
        d = np.abs(a - prev[act_i]).sum(-1)
        dact.append(d[~np.isnan(d)])
        prev[act_i] = a
        for j, i in enumerate(act_i):
            obs[i], r, done[i] = pol.step(envs[i], a[j])
            ret[i] += r
    norm = np.array([pol.env.get_normalized_score(r) * 100 for r in ret])
    return dict(score=float(norm.mean()), score_std=float(norm.std()), score_min=float(norm.min()),
                at_bound=float(np.concatenate(bound).mean()), action_change=float(np.concatenate(dact).mean()))


def svg(path: Path, title: str, rows, Ks, A, B, awac_xy):
    """rows: list of (state index, label); panels: data, then samples at each K."""
    P, M, T = 150, 18, 40  # panel size, gap, title height
    W = 110 + (len(Ks) + 1) * (P + M)
    H = T + 20 + len(rows) * (P + M)
    to = lambda v: (v + 1.1) / 2.2 * P  # [-1.1, 1.1] -> [0, P]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>', f'<text x="10" y="18" font-size="13">{title}</text>']
    heads = ["data (128 nearest states)"] + [f"samples, K = {K}" for K, _ in Ks]
    for c, h in enumerate(heads):
        out.append(f'<text x="{110 + c * (P + M)}" y="{T + 8}">{h}</text>')
    data = torch.cat([A, B], 1).cpu().numpy()
    for r, (i, label) in enumerate(rows):
        y0 = T + 16 + r * (P + M)
        out.append(f'<text x="6" y="{y0 + P / 2}">{label}</text>')
        for c in range(len(Ks) + 1):
            x0 = 110 + c * (P + M)
            g = [f'<g transform="translate({x0},{y0})">',
                 f'<rect x="{to(-1):.1f}" y="{to(-1):.1f}" width="{to(1) - to(-1):.1f}" height="{to(1) - to(-1):.1f}" fill="none" stroke="#999"/>',
                 f'<line x1="{to(0):.1f}" y1="0" x2="{to(0):.1f}" y2="{P}" stroke="#ddd"/>',
                 f'<line x1="0" y1="{to(0):.1f}" x2="{P}" y2="{to(0):.1f}" stroke="#ddd"/>']
            g += [f'<circle cx="{to(a):.1f}" cy="{P - to(b):.1f}" r="2" fill="#888" fill-opacity="{0.7 if c == 0 else 0.3}"/>'
                  for a, b in data[i]]
            if c > 0:
                xs = Ks[c - 1][1][i].cpu().numpy()
                g += [f'<circle cx="{to(a):.1f}" cy="{P - to(b):.1f}" r="1.6" fill="#1f5fbf" fill-opacity="0.35"/>' for a, b in xs]
            if awac_xy is not None:
                a, b = awac_xy[i].tolist()
                g.append(f'<text x="{to(a) - 4:.1f}" y="{P - to(b) + 4:.1f}" fill="#d62728" font-size="13">×</text>')
            out += g + ["</g>"]
    out.append("</svg>")
    path.write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--awac", type=Path, default=None, help="Gaussian AWAC checkpoint folder (same dataset)")
    ap.add_argument("--awac-checkpoint", default="500000", help="step (nearest is used) or 'last'")
    ap.add_argument("--checkpoints", nargs="+", default=["500000", "last"])
    ap.add_argument("--K", nargs="+", type=int, default=[20, 30, 100, 1000])
    ap.add_argument("--rollout-K", nargs="+", type=int, default=[20, 30, 100])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--no-svg", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    pol = Policy(args.run_dir)
    idx = np.random.default_rng(0).choice(len(pol.data["observations"]), N_STATES, replace=False)
    s = pol.states(pol.data["observations"][idx])
    A, B, radius = nearest_data(pol, idx)
    ref_to_b = torch.cdist(A, B).min(-1).values
    thr = torch.quantile(ref_to_b.flatten(), 0.95).item()
    data = dict(nn_radius_median=radius, spread=torch.cat([A, B], 1).std(1).mean().item(),
                at_bound=(torch.cat([A, B], 1).abs() >= 0.999).float().mean().item(),
                ref=compare(A, A, B, thr), off_support_threshold=thr)
    spread_rank = torch.cat([A, B], 1).std(1).mean(-1).argsort(descending=True).tolist()
    show = [0, 1, 2] + [i for i in spread_rank if i > 2][:3]

    awac, awac_xy, awac_info = None, None, None
    if args.awac:
        awac = AwacPolicy(args.awac, pol, pick_checkpoint(args.awac, args.awac_checkpoint))
        awac_xy = awac.mean(s).clamp(pol.low, pol.high)
        to_b = torch.cdist(awac_xy[:, None], B).min(-1).values
        awac_info = dict(run=args.awac.name, checkpoint=awac.checkpoint.name, precision=to_b.mean().item(),
                         off_support=(to_b > thr).float().mean().item(),
                         at_bound=(awac_xy.abs() >= 0.999).float().mean().item())

    result = dict(info=pol.info(), data=data, awac=awac_info, K=args.K, checkpoints={})
    for want in args.checkpoints:
        name = pick_checkpoint(args.run_dir, want)
        if name in result["checkpoints"]:
            continue
        pol.model.load_state_dict(torch.load(args.run_dir / name, map_location=pol.device)["model"])
        pol.model.eval()
        smp = sampler_block(pol, s, A, B, args.K, thr)
        if not args.no_svg:
            rows = [(i, f"state {i}" + (" (wide)" if j >= 3 else "")) for j, i in enumerate(show)]
            svg(args.out.with_name(args.out.stem + f"_{Path(name).stem}.svg"),
                f"{args.run_dir.name}, {name}: grey = data actions, blue = EqM samples, red x = Gaussian AWAC mean",
                rows, [(K, smp[K]["_xy"]) for K in args.K], A, B, awac_xy)
        for K in args.K:
            del smp[K]["_xy"]
        roll = {K: rollouts(pol, K, args.episodes) for K in args.rollout_K}
        result["checkpoints"][name] = dict(sampler=smp, rollouts=roll)
        print(f"{name}: " + " | ".join(f"K={K} spread {smp[K]['spread']:.3f} modes {smp[K]['n_modes']:.2f}"
                                       for K in args.K) + " || " +
              " | ".join(f"K={K} score {r['score']:.1f}" for K, r in roll.items()), flush=True)

    result["seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} ({result['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
