"""Episode statistics of Minari HalfCheetah datasets: return spread, speed, flips and action
saturation (EqM_RL.md, Experiment 2, diagnostic C). A flip is the body angle obs[1] beyond
2 rad at some step; obs[8] is the forward speed.

    python diagnostics/dataset_stats.py mujoco/halfcheetah/simple-v0 mujoco/halfcheetah/medium-v0
"""

import argparse

import minari
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("datasets", nargs="*", default=["mujoco/halfcheetah/simple-v0", "mujoco/halfcheetah/medium-v0"])
ap.add_argument("--flip_angle", type=float, default=2.0)
args = ap.parse_args()

for name in args.datasets:
    if name not in minari.list_local_datasets():
        minari.download_dataset(name)
    ds = minari.load_dataset(name)
    R, flip, speed, maxang, sat = [], [], [], [], []
    for ep in ds.iterate_episodes():
        o = ep.observations[:-1]
        ang = np.abs(o[:, 1])
        R.append(ep.rewards.sum())
        maxang.append(ang.max())
        flip.append(ang.max() > args.flip_angle)
        speed.append(o[:, 8].mean())
        sat.append((np.abs(ep.actions) > 0.99).mean())
    R, flip, speed, maxang = map(np.array, (R, flip, speed, maxang))
    pct = " / ".join(f"{np.percentile(R, p):.0f}" for p in (1, 5, 25, 50, 75, 95))
    print(f"\n{name}: {len(R)} episodes")
    print(f"  return {R.mean():.0f} ± {R.std():.0f}; percentiles 1/5/25/50/75/95: {pct}")
    print(f"  episodes with return < 5000: {(R < 5000).mean() * 100:.1f}%, < 9000: {(R < 9000).mean() * 100:.1f}%")
    flipped = f"{R[flip].mean():.0f}" if flip.any() else "-"
    print(f"  flipped episodes: {flip.mean() * 100:.1f}%; return flipped {flipped} vs not {R[~flip].mean():.0f}")
    print(f"  forward speed {speed.mean():.2f} m/s (episodes without a flip: {speed[~flip].mean():.2f}); "
          f"max |body angle| median {np.median(maxang):.2f} rad")
    print(f"  action dims at the bounds (|a| > 0.99): {np.mean(sat) * 100:.1f}%")
