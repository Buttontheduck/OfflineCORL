"""Actions of the trained actors on the same dataset states, all seeds together (maze2d-umaze-v1).

100 dataset states (seed 0). For each state, 40 actions from each actor:
  Aware T = 0.1 and T = 1 (Experiment 9, 1M): 40 samples of its sampler from N(0, I), clipped to the
      action box as in Actor.sample, at K = 10, 20 (as trained and evaluated), 30 and 100 (snapshots of
      the same trajectories)
  Gaussian AWAC (Experiment 8, its last checkpoint): 40 draws from N(mean, std), clipped, as the
      actor's forward() does in training; plus the mean, which is what evaluation uses
for seeds 1, 2 and 3. Also each state's own dataset action and the actions of its 128 nearest dataset
states (normalized state space) as reference.

Writes <out_dir>/umaze_actions.json (for the figure page), umaze_actions.npz and umaze_actions.csv.
Run from algorithms/EqM_RL/ in the hawk env:
  python diagnostics/maze2d_action_samples.py diagnostics/out/maze2d_modes/figures
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AwacPolicy, Policy  # noqa: E402
from maze2d_modes import nearest_data, pick_checkpoint  # noqa: E402

C = Path("/workspace/Trained_Models/checkpoints")
ENV, N_STATES, N_ACT, KS = "maze2d-umaze-v1", 100, 40, [10, 20, 30, 100]


def fullest(pattern: str) -> Path:
    """The folder matching the pattern with the most checkpoints (complete run, not a first start)."""
    return max(C.glob(pattern), key=lambda d: len(list(d.glob("checkpoint_*.pt"))))


def q(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64) * 1000).astype(int).ravel().tolist()


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = {(T, s): fullest(f"Exp9_Aware_env_name{ENV}_temperature{T}_seed{s}-*") for T in ("0.1", "1") for s in (1, 2, 3)}
    awac_runs = {s: fullest(f"Exp8_AWAC_env_name{ENV}_seed{s}-*") for s in (1, 2, 3)}

    pol = Policy(runs[("1", 1)], checkpoint=pick_checkpoint(runs[("1", 1)], "last"))
    obs = pol.data["observations"]
    idx = np.random.default_rng(0).choice(len(obs), N_STATES, replace=False)
    s = pol.states(obs[idx])
    A, B, _ = nearest_data(pol, idx)
    nn = torch.cat([A, B], 1).cpu().numpy()

    res = dict(env=ENV, n_states=N_STATES, n_actions=N_ACT, K=KS, K_train=pol.cfg["actor"]["num_step"],
               states=[dict(index=int(i), obs=[round(float(v), 3) for v in obs[i]]) for i in idx],
               data_action=q(pol.data["actions"][idx]), data_nn=q(nn), methods=[])
    arrays = dict(state_index=idx, states_obs=obs[idx], data_action=pol.data["actions"][idx], data_nn128=nn)
    rows = []
    for T in ("0.1", "1"):
        m = dict(id=f"aware_T{T}", label=f"Aware, T = {T}", seeds=[])
        for sd in (1, 2, 3):
            run = runs[(T, sd)]
            name = pick_checkpoint(run, "last")
            pol.model.load_state_dict(torch.load(run / name, map_location=pol.device)["model"])
            pol.model.eval()
            torch.manual_seed(sd)
            _, snaps = pol.sample(s.repeat_interleave(N_ACT, 0), max(KS), snaps=set(KS))
            acts = {K: snaps[K].clamp(pol.low, pol.high).view(N_STATES, N_ACT, -1).cpu().numpy() for K in KS}
            m["seeds"].append(dict(seed=sd, run=run.name, checkpoint=name, actions={str(K): q(acts[K]) for K in KS}))
            for K in KS:
                arrays[f"aware_T{T}_seed{sd}_K{K}"] = acts[K]
                if K == 20:
                    rows += [f"aware_T{T},{sd},{int(idx[i])},{n},{a[0]:.4f},{a[1]:.4f}" for i in range(N_STATES) for n, a in enumerate(acts[K][i])]
        res["methods"].append(m)
    m = dict(id="awac", label="Gaussian AWAC", seeds=[])
    for sd in (1, 2, 3):
        run = awac_runs[sd]
        aw = AwacPolicy(run, pol, pick_checkpoint(run, "last"))
        mean = aw.mean(s)
        std = aw.actor._log_std.clamp(aw.actor._min_log_std, aw.actor._max_log_std).exp()
        g = torch.Generator(device=pol.device).manual_seed(sd)
        draws = mean[:, None] + std * torch.randn(N_STATES, N_ACT, pol.action_dim, device=pol.device, generator=g)
        draws = draws.clamp(pol.low, pol.high).cpu().numpy()
        mc = mean.clamp(pol.low, pol.high).cpu().numpy()
        m["seeds"].append(dict(seed=sd, run=run.name, checkpoint=aw.checkpoint.name, std=[round(float(v), 4) for v in std],
                               actions={"all": q(draws)}, mean=q(mc)))
        arrays[f"awac_seed{sd}"], arrays[f"awac_seed{sd}_mean"] = draws, mc
        rows += [f"awac,{sd},{int(idx[i])},{n},{a[0]:.4f},{a[1]:.4f}" for i in range(N_STATES) for n, a in enumerate(draws[i])]
        print(f"AWAC seed {sd}: {aw.checkpoint.name}, std {std.cpu().numpy().round(3)}", flush=True)
    res["methods"].append(m)

    (out / "umaze_actions.json").write_text(json.dumps(res, separators=(",", ":")))
    np.savez_compressed(out / "umaze_actions.npz", **arrays)
    (out / "umaze_actions.csv").write_text("method,seed,dataset_index,sample,action_x,action_y\n" + "\n".join(rows) + "\n")
    for mm in res["methods"]:
        print(mm["label"], [(x["seed"], x["checkpoint"]) for x in mm["seeds"]])
    print(f"wrote {out}/umaze_actions.{{json,npz,csv}} (Aware rows in the CSV: K = 20)")


if __name__ == "__main__":
    main()
