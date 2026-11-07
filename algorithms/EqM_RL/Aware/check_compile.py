"""Check that the torch.compile'd EqM sampler (config `compile: true`, Actor.compile_sampler) gives the
same results as the eager one, and how much faster it is. Four checks, all on the GPU:

  1. sampler   Actor._implicit from the same noise and states, eager vs compiled, at the batch sizes
               Aware uses (256 training, 10 eval, a smaller eval batch, 2048 diagnostics).
               Pass: max |difference| <= --atol.
  2. training  Two identical agents (same weights, same batches, same random draws) take --steps
               Otter.update steps, one with the eager sampler, one with the compiled one.
               Pass: losses and final weights agree to --rtol. They can't match bit for bit: fused
               kernels round differently, and training slowly amplifies that.
  3. speed     ms per sampler call (B = 256 and B = 10) and per full Otter.update, eager vs compiled.
  4. logger    The new tensor path of logger.py gives the same means as the old float path.

The networks are Aware's (4 x 256 field, two 4 x 256 critics). By default they are freshly
initialized at HalfCheetah sizes on random data; --checkpoint loads a trained Aware checkpoint
(model, critic_1, critic_2) instead, sizes taken from it.

    python check_compile.py
    python check_compile.py --checkpoint /workspace/Trained_Models/checkpoints/<run>/checkpoint_310000.pt
"""

import argparse
import copy
import sys
import time

import numpy as np
import torch

from actor_models import Actor
from agents import Otter
from critic_models import DeterministicCritic
from logger import _mean, global_logger as gl
from utils.networks import ConditionalMLP

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", help="an Aware checkpoint_*.pt (default: fresh networks)")
ap.add_argument("--mode", default="reduce-overhead", help="torch.compile mode ('reduce-overhead' or 'default')")
ap.add_argument("--steps", type=int, default=200, help="training steps compared in check 2")
ap.add_argument("--atol", type=float, default=1e-5, help="sampler: largest accepted |eager - compiled|")
ap.add_argument("--rtol", type=float, default=1e-3, help="training: largest accepted relative difference")
ap.add_argument("--temperature", type=float, default=1.0)
args = ap.parse_args()
dev = "cuda"
HID, LAYERS, BATCH = 256, 4, 256

# ---------------------------------------------------------------- networks, as Aware/main.py builds them
if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location=dev)
    lin = [v for k, v in ck["model"].items() if k.endswith(".weight")]   # the field's Linear layers in order
    HID, LAYERS = lin[0].shape[0], len(lin) - 1
    A = lin[-1].shape[0]                                  # head: (action_dim, hidden)
    S = lin[0].shape[1] - A                               # first layer: (hidden, action_dim + state_dim)
else:
    ck, S, A = None, 17, 6                                # HalfCheetah


def build(compiled: bool):
    """A fresh Otter with the same weights every time (seeded init, or the checkpoint)."""
    torch.manual_seed(0)
    model = ConditionalMLP(S + A, A, HID, LAYERS).to(dev)
    c1, c2 = DeterministicCritic(S, A, HID, LAYERS).to(dev), DeterministicCritic(S, A, HID, LAYERS).to(dev)
    if ck is not None:
        model.load_state_dict(ck["model"])
        c1.load_state_dict(ck["critic_1"])
        c2.load_state_dict(ck["critic_2"])
    actor = Actor(model=model, critic_1=c1, critic_2=c2, action_dim=A, min_action=-1.0, max_action=1.0, ebm=None,
                  opt_type="nag", step_size=0.01, num_step=20, moment=0.3, sampler_type="implicit",
                  ood_threshold=0.01, early_stop=None)
    if compiled:
        actor.compile_sampler(mode=args.mode)
    agent = Otter(model=model, model_optimizer=torch.optim.Adam(model.parameters(), lr=3e-4), actor=actor,
                  critic_1=c1, critic_1_optimizer=torch.optim.Adam(c1.parameters(), lr=3e-4),
                  critic_2=c2, critic_2_optimizer=torch.optim.Adam(c2.parameters(), lr=3e-4),
                  temperature=args.temperature)
    return agent, actor


# random data in Aware's units: normalized states ~ N(0, 1), actions in [-1, 1]
g = torch.Generator(device=dev).manual_seed(1)
N = 20000
DATA = [torch.randn(N, S, device=dev, generator=g), torch.rand(N, A, device=dev, generator=g) * 2 - 1,
        torch.randn(N, 1, device=dev, generator=g), torch.randn(N, S, device=dev, generator=g),
        torch.zeros(N, 1, device=dev), (torch.rand(N, 1, device=dev, generator=g) < 0.01).float()]


def batch(t: int):
    idx = torch.randint(0, N, (BATCH,), device=dev, generator=torch.Generator(device=dev).manual_seed(10_000 + t))
    return [d[idx] for d in DATA]


eager_agent, eager_actor = build(False)
comp_agent, comp_actor = build(True)
failed = []
print(f"PyTorch {torch.__version__}, {torch.cuda.get_device_name()}, compile mode '{args.mode}', "
      f"networks: {'checkpoint ' + args.checkpoint if ck else 'fresh'} (state {S}, action {A})\n")

# ---------------------------------------------------------------- 1. sampler
print("1. sampler, eager vs compiled (same noise, same states)")
t0 = time.time()
for B in (256, 10, 3, 2048):
    s = torch.randn(B, S, device=dev, generator=g)
    x0 = torch.randn(B, A, device=dev, generator=g)
    a_eager = eager_actor._implicit(x0, s)
    a_comp = comp_actor._implicit(x0, s)       # the first call per shape compiles or records a graph
    a_comp = comp_actor._implicit(x0, s)       # the replay is what training uses
    diff = (a_eager - a_comp).abs().max().item()
    ok = diff <= args.atol
    failed += [] if ok else [f"sampler B={B}"]
    print(f"   B = {B:>4}: max |diff| {diff:.2e}   (|action| up to {a_eager.abs().max().item():.2f})   "
          f"{'ok' if ok else 'FAIL'}")
print(f"   (compiling took {time.time() - t0:.0f} s)\n")

# ---------------------------------------------------------------- 2. training
print(f"2. training, {args.steps} Otter.update steps from identical agents")
loss_rel = []
for t in range(args.steps):
    b = batch(t)
    torch.manual_seed(t)                       # same rho, noise and sampler start for both agents
    r_e = eager_agent.update(b)
    torch.manual_seed(t)
    r_c = comp_agent.update(b)
    for k in r_e:
        e, c = float(r_e[k]), float(r_c[k])
        loss_rel.append(abs(e - c) / max(abs(e), 1e-8))
gl.buffer.clear()


def param_rel(m_e, m_c):
    pe = torch.cat([p.detach().flatten() for p in m_e.parameters()])
    pc = torch.cat([p.detach().flatten() for p in m_c.parameters()])
    return ((pe - pc).norm() / pe.norm()).item()


rels = {"losses (max over steps)": max(loss_rel),
        "EqM field weights": param_rel(eager_agent._model, comp_agent._model),
        "critic 1 weights": param_rel(eager_agent._critic_1, comp_agent._critic_1),
        "critic 2 weights": param_rel(eager_agent._critic_2, comp_agent._critic_2)}
for k, v in rels.items():
    ok = v <= args.rtol
    failed += [] if ok else [f"training {k}"]
    print(f"   {k:<24} relative difference {v:.2e}   {'ok' if ok else 'FAIL'}")
print()

# ---------------------------------------------------------------- 3. speed


def ms_per_call(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


print("3. speed (ms per call, GPU synchronized)")
s256, x256 = torch.randn(256, S, device=dev), torch.randn(256, A, device=dev)
s10, x10 = torch.randn(10, S, device=dev), torch.randn(10, A, device=dev)
b = batch(0)
rows = [("sampler, B = 256 (training)", lambda a: a._implicit(x256, s256)),
        ("sampler, B = 10 (eval)", lambda a: a._implicit(x10, s10)),
        ("full Otter.update", None)]
for name, fn in rows:
    if fn is None:
        te = ms_per_call(lambda: eager_agent.update(b))
        tc = ms_per_call(lambda: comp_agent.update(b))
        gl.buffer.clear()
    else:
        te, tc = ms_per_call(lambda: fn(eager_actor)), ms_per_call(lambda: fn(comp_actor))
    print(f"   {name:<28} eager {te:6.2f}   compiled {tc:6.2f}   x{te / tc:.1f}")
print()

# ---------------------------------------------------------------- 4. logger
print("4. logger: tensor values vs the old float path")
vals = torch.randn(1000, device=dev) * 3 + 1
old = float(np.mean([v.item() for v in vals]))
new = _mean([v for v in vals])
ok = abs(old - new) <= 1e-9 * max(1.0, abs(old))
failed += [] if ok else ["logger"]
print(f"   mean of 1000 values: old {old:.12f}   new {new:.12f}   {'ok' if ok else 'FAIL'}\n")

print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
