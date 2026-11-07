"""Data for the mode-collapse figures (EqM_RL.md, "Experiment 9 follow-up"): how the EqM sampler
moves from noise to one action per state on Maze2D. Evaluation only.

Cases (all maze2d-medium-v1): Aware T = 4 seed 1 (Experiment 8, last checkpoint), T = 1 seed 1 and
T = 0.1 seed 1 (Experiment 9, 1M), and T = 0.1 seed 2 at 500k (still in its critic-overestimation
phase). Same 256 dataset states and the same data reference as maze2d_modes.py.

Writes to <out_dir>:
  figdata.json            for the figure page: per case and K frame (0 ... 1000) the metrics over all
                          256 states; for 6 states (3 random, 3 with the widest data spread) the
                          data actions, the 256 sample trajectories at every frame, the field -f on a
                          grid over [-2, 2]^2 and the Gaussian AWAC mean action
  <case>_K20_30_100.npz   samples at K = 20, 30 and 100 (and the K = 1000 endpoints) for all 256
                          states x 256 samples, unclipped, plus the states (raw observations), the
                          128 nearest-neighbour data actions and the AWAC mean actions
  shown_states_K20_30_100.csv   the same samples for the 6 plotted states, long format

Run from algorithms/EqM_RL/ in the hawk env:
  python diagnostics/maze2d_mode_figures.py diagnostics/out/maze2d_modes/figures
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AwacPolicy, Policy  # noqa: E402
from maze2d_modes import N_CMP, N_SAMPLES, N_STATES, compare, modes, nearest_data, pick_checkpoint  # noqa: E402

C = Path("/workspace/Trained_Models/checkpoints")
CASES = [  # id, label, run folder, checkpoint, AWAC folder (same seed)
    ("T4_s1", "T = 4 (near BC), seed 1, 540k", "Exp8_Aware-T4_env_namemaze2d-medium-v1_seed1-maze2d-medium-v1-a8d94620", "last",
     "Exp8_AWAC_env_namemaze2d-medium-v1_seed1-maze2d-medium-v1-d97d00c4"),
    ("T1_s1", "T = 1, seed 1, 1M", "Exp9_Aware_env_namemaze2d-medium-v1_temperature1_seed1-maze2d-medium-v1-dc0308a3", "last",
     "Exp8_AWAC_env_namemaze2d-medium-v1_seed1-maze2d-medium-v1-d97d00c4"),
    ("T0.1_s1", "T = 0.1, seed 1, 1M", "Exp9_Aware_env_namemaze2d-medium-v1_temperature0.1_seed1-maze2d-medium-v1-a787b75d", "last",
     "Exp8_AWAC_env_namemaze2d-medium-v1_seed1-maze2d-medium-v1-d97d00c4"),
    ("T0.1_s2_500k", "T = 0.1, seed 2, 500k (critic overestimating)",
     "Exp9_Aware_env_namemaze2d-medium-v1_temperature0.1_seed2-maze2d-medium-v1-e2f01930", "500000",
     "Exp8_AWAC_env_namemaze2d-medium-v1_seed2-maze2d-medium-v1-f9892b66"),
]
FRAMES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000]
GRID, LIM = 25, 2.0


def q(a) -> list:
    """Compact JSON: values x 1000 as ints."""
    return np.round(np.asarray(a, dtype=np.float64) * 1000).astype(int).ravel().tolist()


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    pol = Policy(C / CASES[0][2], checkpoint=pick_checkpoint(C / CASES[0][2], CASES[0][3]))
    obs = pol.data["observations"]
    idx = np.random.default_rng(0).choice(len(obs), N_STATES, replace=False)  # as maze2d_modes.py
    s = pol.states(obs[idx])
    A, B, radius = nearest_data(pol, idx)
    data_all = torch.cat([A, B], 1)
    thr = torch.quantile(torch.cdist(A, B).min(-1).values.flatten(), 0.95).item()
    rank = data_all.std(1).mean(-1).argsort(descending=True).tolist()
    show = [0, 1, 2] + [i for i in rank if i > 2][:3]

    g = torch.linspace(-LIM, LIM, GRID, device=pol.device)
    grid = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)  # x fastest

    fig = dict(frames=FRAMES, grid=dict(n=GRID, lim=LIM), env=pol.cfg["env_name"],
               sampler=dict(eta=pol.eta, momentum=pol.mu, K_train=pol.cfg["actor"]["num_step"]),
               data=dict(spread=data_all.std(1).mean().item(),
                         at_bound=(data_all.abs() >= 0.999).float().mean().item(),
                         ref=compare(A, A, B, thr), nn_radius_median=radius),
               states=[dict(index=int(idx[i]), obs=[round(float(v), 3) for v in obs[idx[i]]],
                            wide=j >= 3, data=q(data_all[i].cpu())) for j, i in enumerate(show)],
               cases=[])
    rows = []
    for cid, label, run, ck, awac_run in CASES:
        name = pick_checkpoint(C / run, ck)
        sd = torch.load(C / run / name, map_location=pol.device)
        pol.model.load_state_dict(sd["model"])
        pol.model.eval()
        awac = AwacPolicy(C / awac_run, pol, pick_checkpoint(C / awac_run, "500000"))
        awac_xy = awac.mean(s).clamp(pol.low, pol.high)

        torch.manual_seed(0)
        s_rep = s.repeat_interleave(N_SAMPLES, 0)
        x0 = torch.randn(len(s_rep), pol.action_dim, device=pol.device)
        _, snaps = pol.sample(s_rep, max(FRAMES), snaps=set(FRAMES), x0=x0)
        snaps[0] = x0

        metrics, traj = [], {i: [] for i in show}
        for K in FRAMES:
            x = snaps[K]
            f = pol.model(x, s_rep).norm(dim=-1)
            x = x.view(N_STATES, N_SAMPLES, -1)
            xc = x.clamp(pol.low, pol.high)
            n_modes, top = modes(x)
            metrics.append(dict(K=K, f_end=f.mean().item(), spread=xc.std(1).mean().item(),
                                one_mode=(n_modes == 1).float().mean().item(), top_mass=top.mean().item(),
                                at_bound=(xc.abs() >= 0.999).float().mean().item(),
                                **compare(xc[:, :N_CMP], A, B, thr)))
            for i in show:
                traj[i].append(q(x[i].cpu()))

        fields = []
        for i in show:
            fv = -pol.model(grid, s[i].expand(len(grid), -1))  # the direction the sampler moves
            fields.append(q(fv.cpu()))
        fig["cases"].append(dict(id=cid, label=label, run=run, checkpoint=name,
                                 temperature=pol_temperature(C / run), metrics=metrics,
                                 traj=[traj[i] for i in show], field=fields,
                                 awac=[q(awac_xy[i].cpu()) for i in show], awac_checkpoint=awac.checkpoint.name))

        keep = {K: snaps[K].view(N_STATES, N_SAMPLES, -1).cpu().numpy() for K in (20, 30, 100, 1000)}
        np.savez_compressed(out / f"{cid}_K20_30_100.npz", K20=keep[20], K30=keep[30], K100=keep[100],
                            K1000=keep[1000], states_obs=obs[idx], state_index=idx,
                            data_actions_nn128=data_all.cpu().numpy(), awac_mean=awac_xy.cpu().numpy(),
                            action_low=pol.low, action_high=pol.high)
        for j, i in enumerate(show):
            for K in (20, 30, 100):
                for n, (a0, a1) in enumerate(keep[K][i]):
                    rows.append(f"{cid},{int(idx[i])},{K},{n},{a0:.4f},{a1:.4f}")
        print(f"{cid}: " + ", ".join(f"K={m['K']} spread {m['spread']:.3f}" for m in metrics if m["K"] in (20, 30, 100, 1000)),
              flush=True)

    (out / "figdata.json").write_text(json.dumps(fig, separators=(",", ":")))
    (out / "shown_states_K20_30_100.csv").write_text(
        "case,dataset_index,K,sample,action_x,action_y\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}")


def pol_temperature(run_dir: Path):
    import yaml
    return yaml.safe_load(open(run_dir / "config.yaml")).get("temperature")


if __name__ == "__main__":
    main()
