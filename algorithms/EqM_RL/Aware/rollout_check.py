"""Does the compiled EqM sampler (config `compile: true`) drive real episodes exactly like the eager one?
The companion of check_compile.py, which checks single sampler calls and training steps on random data.
This one loads a trained Aware policy and plays HalfCheetah (or any D4RL/Minari env of the run):

  A. lockstep  N episodes follow the compiled actions. At every step the eager action is also computed,
               from the same state and the same starting noise. Prints the largest |compiled - eager|
               over all steps (about 1e-7 to 1e-6 means: the same policy, only rounding differs).
  B. rates     M episodes led by the eager sampler, then M led by the compiled one (same episode seeds and
               noise stream). Prints mean and median return, failed episodes (return below half the
               median) and flipped ones (|obs[1]| > 2, the cheetah on its back). The trajectories differ
               (a 1000-step episode amplifies 1e-7), so compare the counts, not single episodes.

Written for the jit_check question (2026-11-07): single failed eval episodes in a compiled run. Results:
medium-expert T = 4 seed 1 at 100k: max diff 7.8e-7; failed 22 vs 22, flipped 62 vs 63 of 200.
medium-replay seed 1 at 150k: max diff 4.2e-7; failed 0 vs 0.

Run in the hawk env, with MuJoCo on the library path:
    export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH D4RL_SUPPRESS_IMPORT_ERROR=1
    python rollout_check.py <run_dir> <checkpoint> [N, default 50] [M, default 200]
    python rollout_check.py /workspace/Trained_Models/checkpoints/Aware_1M_T4_seed1-halfcheetah-medium-expert-v2-91e62406 checkpoint_100000.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from common import Policy  # noqa: E402  (loads the run's config, dataset, env, normalization and EqM field)
from actor_models import Actor  # noqa: E402

run_dir, ckpt = Path(sys.argv[1]), sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
M = int(sys.argv[4]) if len(sys.argv) > 4 else 200
pol = Policy(run_dir, ckpt)
a = pol.cfg["actor"]


def actor(compiled: bool) -> Actor:
    """Aware's sampler around the run's field, with the run's own K, step size and momentum."""
    act = Actor(model=pol.model, critic_1=None, critic_2=None, action_dim=pol.action_dim, min_action=pol.low,
                max_action=pol.high, ebm=None, opt_type="nag", step_size=a["step_size"], num_step=a["num_step"],
                moment=a["moment"], sampler_type="implicit", ood_threshold=0.01, early_stop=None)
    return act.compile_sampler() if compiled else act


eager, comp = actor(False), actor(True)
envs = [pol.make_env() for _ in range(max(N, M))]


def run(n: int, lead: Actor, check: Actor = None, seed0: int = 1000):
    """n episodes side by side, driven by `lead`. With `check`, also computes check's action at every step
    from the same state and noise. -> (returns, flipped, largest |lead - check| action difference)"""
    obs = np.stack([pol.reset(envs[i], seed0 + i) for i in range(n)])
    ret, done, flip = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    g = torch.Generator(device="cuda").manual_seed(seed0)
    worst = 0.0
    while not done.all():
        act_idx = np.flatnonzero(~done)
        s = pol.states(obs[act_idx])
        x0 = torch.randn(len(act_idx), pol.action_dim, device="cuda", generator=g)
        x = lead._implicit(x0, s)
        if check is not None:
            worst = max(worst, (x - check._implicit(x0, s)).abs().max().item())
        x = x.clamp(pol.low, pol.high).cpu().numpy()
        for k, i in enumerate(act_idx):
            obs[i], r, done[i] = pol.step(envs[i], x[k])
            ret[i] += r
            flip[i] |= abs(obs[i][1]) > 2.0
    return ret, flip, worst


print(f"{run_dir.name}/{ckpt}  (K = {a['num_step']}, eta = {a['step_size']}, momentum = {a['moment']})")
_, _, worst = run(N, comp, check=eager)
print(f"A. lockstep, {N} episodes following the compiled actions:")
print(f"   largest |compiled - eager| action difference over all steps: {worst:.2e}")
for name, lead in (("eager", eager), ("compiled", comp)):
    r, f, _ = run(M, lead, seed0=5000)
    fail = r < 0.5 * np.median(r)
    print(f"B. {name:<8} {M} episodes: mean return {r.mean():8.1f}, median {np.median(r):8.1f}, "
          f"failed (< half the median) {fail.sum():3d}, flipped {f.sum():3d}")
