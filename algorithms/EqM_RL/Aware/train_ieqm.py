"""IEQM: IFQL (IFQL.md) with our EqM actor in place of IFQL's flow network. OGBench single-task data only
(run with the ogbench conda env). Experiment 19.

What is IFQL's, unchanged (IFQL.md, Sections 3, 4, 5 and 7; FQL's agents/ifql.py):
  networks   V(s), Q1(s, a), Q2(s, a): n_hidden x (Linear -> GELU -> LayerNorm), then Linear -> 1. Glorot-uniform
             weights, zero biases. Target copies of Q1 and Q2 only (V has none).
  V loss     d = min(Q1_target, Q2_target)(s, a) - V(s) at the dataset action;
             expectile * d^2 where d >= 0 (V too low), (1 - expectile) * d^2 where d < 0. Mean over the batch.
  Q loss     y = r + gamma * mask * V(s') with the current V (no gradient into V);
             (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2, mean over the batch.
  one step   one Adam step on V loss + Q loss + actor loss (each loss reaches only its own network), then
             target <- (1 - tau) * target + tau * Q with the Q weights from BEFORE this step, as in FQL's code.
  inputs     raw states for V and Q (no state normalization), rewards as given (no scaling), dataset actions
             clipped to [low + action_clip_eps, high - action_clip_eps], no clipping of the TD target.
  acting     eval_candidates (32) actions from the actor, clipped to the action box; the one with the highest
             min(Q1, Q2) of the trained critic (not the target copies) is executed.
The critic never sees an action made by the actor during training, so it does not depend on the actor at all.

Speed (no change of the maths): Adam runs as one fused kernel, and compile=true runs the three losses as one compiled
CUDA graph (torch.compile, mode reduce-overhead): 2.6 instead of 8.9 ms per step. Checked on 6,000 steps: with and
without compile the critic's losses agree to 4 digits at step 3,000. The evals are not compiled.

What is ours (the only change of the algorithm):
  actor      the EqM field (ConditionalMLP, Mish), trained by behaviour cloning on the same batches with Aware's EqM
             loss (Otter._actor_loss, every weight 1), on normalized states as all our EqM actors. It never sees Q
             or V. Sampled with Aware's sampler: num_step (K = 20) NAG steps from N(0, I).

Optional second actor, evaluated with the same critic (frozen_bc_prefix, e.g. Exp14_BC): the frozen EqM-BC actor
of fit_actor.py with this seed, <bc_root>/<frozen_bc_prefix>_env_name<env>_actoreqm_seed<seed>-*/<bc_checkpoint>.
It is not trained here. Its eval is the critic comparison with Experiment 15 (same actor, QIPO critic).

Eval every eval_frequency steps: n_test_episodes episodes per task in eval_tasks (ogbench_data.evaluate, seeded
starts). Logged under eval/ (trained actor) and eval_frozen/ (frozen actor): normalized_score = success in %,
score = return, episode_length, q_start = Q of the chosen action at the start states, q_gain = mean over all steps of
Q(chosen) - mean Q(candidates). One line 'RESULT {json}' per eval and actor, and at the end one 'FINAL {json}' per
actor: the mean success of the last final_evals evals (FQL's score: the evals at 800k, 900k and 1M).
diagnostics/summarize_exp19.py reads these lines.

Checkpoints: <checkpoints_path>/<name>/checkpoint_<step>.pt = {"model": EqM field, "value": V, "critic": Q1 and Q2,
"target": their target copies}, with an Aware-style config.yaml, so diagnostics/common.py's Policy loads the actor.

    python train_ieqm.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --seed=1 --frozen_bc_prefix=Exp14_BC
"""

import glob
import json
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
import yaml

import ogbench_data
from actor_models import Actor
from agents import Otter
from train_critic import set_seed
from utils.networks import ConditionalMLP


@dataclass
class TrainConfig:
    project: str = "E"
    group: str = "Exp19_IEQM"
    name: str = "IEQM"
    env_name: str = "humanoidmaze-medium-navigate-singletask-v0"
    # critic: IFQL's settings for humanoidmaze (IFQL.md, Section 7)
    gamma: float = 0.995                  # FQL: 0.995 for humanoidmaze, antmaze-giant and antsoccer, else 0.99
    expectile: float = 0.9
    tau: float = 0.005                    # target update rate (Q only)
    critic_hidden_dim: int = 512          # V, Q1 and Q2
    critic_n_hidden: int = 4
    action_clip_eps: float = 1e-5         # dataset actions are clipped to the action box minus this
    # optimisation (one Adam for all networks, as IFQL)
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_train_ops: int = 1000001          # the extra 1 makes the last eval (and checkpoint) happen at step 1,000,000
    compile: bool = False                 # true = torch.compile the three losses (one CUDA graph per step): the same
                                          # maths, about 3x faster per step (8.5 -> 2.5 ms). The evals are not compiled
    # actor: the EqM field, trained by BC on the same batches (EqM settings as Experiment 14's EqM-BC)
    hidden_dim: int = 512                 # IFQL's actor size is 4 x 512 (Experiment 14's EqM-BC: 4 x 256)
    num_hidden_layers: int = 4
    scale_gradient: float = 8.0
    flag_type: str = "trun"
    flag_threshold: float = 0.8
    step_size: float = 0.01
    num_step: int = 20                    # K: sampler steps, for both actors
    moment: float = 0.3
    # optional frozen EqM-BC actor of fit_actor.py, evaluated with the same critic ("" = none)
    frozen_bc_prefix: str = ""            # e.g. Exp14_BC
    bc_root: str = "/workspace/Trained_Models/checkpoints/"
    bc_checkpoint: str = "checkpoint_500000.pt"
    # evaluation and saving
    eval_candidates: int = 32             # N (IFQL on humanoidmaze: 32)
    eval_frequency: int = 100000
    n_test_episodes: int = 50             # per task (OGBench: 50)
    eval_tasks: str = "own"               # 'own' (the dataset's task, whose rewards the critic learned), 'all' or "1,3"
    final_evals: int = 3                  # the score is the mean success of this many last evals
    test_seed: int = 69                   # as fit_actor.py, train_critic.py and eval_best_of_n.py
    ogbench_dir: str = ogbench_data.OGBENCH_DATA
    log_frequency: int = 5000
    stats_every: int = 10                 # the train/ logs average every this many steps' batches
    checkpoints_path: Optional[str] = "/workspace/Trained_Models/checkpoints/"
    checkpoint_every: int = 100000
    seed: int = 1
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"


class IFQLMLP(nn.Module):
    """FQL's value MLP (utils/networks.py, MLP with layer_norm=True): n_hidden x (Linear -> GELU -> LayerNorm),
    then Linear -> 1. Glorot-uniform weights and zero biases (FQL's default_init). GELU is the tanh form and
    LayerNorm's epsilon is 1e-6, the flax defaults."""

    def __init__(self, in_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(approximate="tanh"), nn.LayerNorm(hidden_dim, eps=1e-6)]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class IFQLCritic(nn.Module):
    """Q1 and Q2: two IFQLMLPs on [state, action], with different random starting weights."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.q1 = IFQLMLP(state_dim + action_dim, hidden_dim, n_hidden)
        self.q2 = IFQLMLP(state_dim + action_dim, hidden_dim, n_hidden)

    def both(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x), self.q2(x)

    def forward(self, s, a):
        return torch.min(*self.both(s, a))


def ifql_losses(value, critic, target, s, a, r, d, s2, gamma: float, expectile: float):
    """IFQL's V and Q losses on one batch (FQL's agents/ifql.py: value_loss and critic_loss). d is the task
    terminal (1 - mask). Returns v_loss, q_loss and the tensors the logs use (no gradient needed from those)."""
    with torch.no_grad():
        q_t = target(s, a)                                   # min of the target copies at the dataset action
        y = r + gamma * (1.0 - d) * value(s2)                # the current V, as a fixed number
    v = value(s)
    diff = q_t - v
    v_loss = (torch.where(diff >= 0, expectile, 1.0 - expectile) * diff ** 2).mean()
    q1, q2 = critic.both(s, a)
    q_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
    return v_loss, q_loss, dict(v=v, q1=q1, q2=q2, y=y, diff=diff)


def find_frozen_bc(cfg: TrainConfig) -> str:
    pattern = os.path.join(cfg.bc_root, f"{cfg.frozen_bc_prefix}_env_name{cfg.env_name}_actoreqm_seed{cfg.seed}"
                                        f"-{cfg.env_name}-*")
    dirs = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, cfg.bc_checkpoint))]
    assert len(dirs) == 1, f"expected one BC run with {cfg.bc_checkpoint} for {pattern}, found {dirs}"
    return dirs[0]


@pyrallis.wrap()
def train(cfg: TrainConfig):
    assert ogbench_data.is_ogbench(cfg.env_name), "train_ieqm.py supports OGBench single-task data only"
    dev = cfg.device
    env, data, mean, std, _, data_info = ogbench_data.load_ogbench(cfg.env_name, cfg.ogbench_dir)
    set_seed(cfg.seed, env)
    n, state_dim, action_dim = len(data["observations"]), data["observations"].shape[1], data["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])

    # ------------------------------------------------------------ dataset: raw states, rewards as given
    S = torch.tensor(data["observations"], dtype=torch.float32, device=dev)
    S2 = torch.tensor(data["next_observations"], dtype=torch.float32, device=dev)
    A = torch.tensor(np.clip(data["actions"], low + cfg.action_clip_eps, high - cfg.action_clip_eps),
                     dtype=torch.float32, device=dev)
    R = torch.tensor(data["rewards"], dtype=torch.float32, device=dev)
    D = torch.tensor(data["terminals"], dtype=torch.float32, device=dev)      # task terminal = 1 - masks
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)              # the actors take (s - mean) / std
    std_t = torch.tensor(std, dtype=torch.float32, device=dev)
    r_min = float(R.min())
    data_info.update(reward_min=r_min, reward_max=float(R.max()),
                     q_floor=r_min / (1 - cfg.gamma) if r_min < 0 else r_min)
    print(f"data: {data_info}", flush=True)

    # ------------------------------------------------------------ networks (critic first: its starting weights
    # then don't depend on the actor's size) and one Adam for all of them
    value = IFQLMLP(state_dim, cfg.critic_hidden_dim, cfg.critic_n_hidden).to(dev)
    critic = IFQLCritic(state_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_n_hidden).to(dev)
    target = deepcopy(critic).requires_grad_(False)
    model = ConditionalMLP(state_dim + action_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(dev)
    opt = torch.optim.Adam([*value.parameters(), *critic.parameters(), *model.parameters()], lr=cfg.learning_rate,
                           fused=dev.startswith("cuda"))    # fused: the same Adam update, computed in one kernel

    def make_sampler(m, opt_type, step_size, moment):
        return Actor(model=m, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low, max_action=high,
                     ebm=None, opt_type=opt_type, step_size=step_size, num_step=cfg.num_step, moment=moment,
                     sampler_type="implicit", ood_threshold=0.01, early_stop=None)

    sampler = make_sampler(model, "nag", cfg.step_size, cfg.moment)
    # Otter only for its EqM loss (_actor_loss with given weights queries no critic), as fit_actor.py
    eqm = Otter(model=model, model_optimizer=opt, actor=sampler, critic_1=None, critic_1_optimizer=None,
                critic_2=None, critic_2_optimizer=None, flag_type=cfg.flag_type, flag_threshold=cfg.flag_threshold,
                scale_gradient=cfg.scale_gradient, num_actions=1)
    ONES = torch.ones(cfg.batch_size, 1, device=dev)
    actors = {"trained": sampler}

    frozen_dir, frozen_cfg = None, None
    if cfg.frozen_bc_prefix:      # built last: the trained networks' starting weights are the same with and without it
        frozen_dir = find_frozen_bc(cfg)
        frozen_cfg = yaml.safe_load(open(os.path.join(frozen_dir, "config.yaml")))
        assert (frozen_cfg["env_name"] == cfg.env_name and frozen_cfg["seed"] == cfg.seed
                and frozen_cfg["actor"]["kind"] == "eqm"), frozen_dir
        fmodel = ConditionalMLP(state_dim + action_dim, action_dim, frozen_cfg["hidden_dim"],
                                frozen_cfg["num_hidden_layers"]).to(dev)
        fmodel.load_state_dict(torch.load(os.path.join(frozen_dir, cfg.bc_checkpoint), map_location=dev)["model"])
        fmodel.eval().requires_grad_(False)
        actors["frozen"] = make_sampler(fmodel, frozen_cfg["actor"]["opt_type"], frozen_cfg["actor"]["step_size"],
                                        frozen_cfg["actor"]["moment"])
        print(f"frozen BC actor {frozen_dir}/{cfg.bc_checkpoint}", flush=True)

    t_params, c_params = list(target.parameters()), list(critic.parameters())

    def total_loss(s, a, r, d, s2):
        v_loss, q_loss, out = ifql_losses(value, critic, target, s, a, r, d, s2, cfg.gamma, cfg.expectile)
        a_loss = eqm._actor_loss((s - mean_t) / std_t, a, weights=ONES)
        return v_loss + q_loss + a_loss, v_loss, q_loss, a_loss, out

    loss_fn = torch.compile(total_loss, mode="reduce-overhead") if cfg.compile else total_loss

    def update(idx, with_stats):
        loss, v_loss, q_loss, a_loss, out = loss_fn(S[idx], A[idx], R[idx], D[idx], S2[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():     # before opt.step(): the target moves toward the Q weights from before this step
            torch._foreach_lerp_(t_params, c_params, cfg.tau)             # target <- (1 - tau) target + tau Q
        opt.step()
        if not with_stats:
            return None
        with torch.no_grad():     # new tensors: a compiled graph overwrites its outputs at the next step
            q = torch.min(out["q1"], out["q2"])
            return {"train/v_loss": v_loss.detach().clone(), "train/q_loss": q_loss.detach().clone(),
                    "train/actor_loss": a_loss.detach().clone(), "train/v_data": out["v"].mean(),
                    "train/q_data": q.mean(), "train/q_data_min": q.min(), "train/q_data_max": q.max(),
                    "train/target": out["y"].mean(), "train/adv": out["diff"].mean(),
                    "train/adv_std": out["diff"].std(), "train/adv_pos_share": (out["diff"] >= 0).float().mean()}

    # ------------------------------------------------------------ evaluation: best of N candidates of an actor
    tasks = ogbench_data.parse_tasks(cfg.eval_tasks, env)
    eval_envs = ogbench_data.make_eval_envs(cfg.env_name, tasks, cfg.n_test_episodes)
    data_info.update(eval_tasks=tasks, eval_episodes_per_task=cfg.n_test_episodes,
                     score="success rate in %, mean over eval_tasks (OGBench)")
    eval_gen = torch.Generator(device=dev).manual_seed(cfg.seed + 2000)   # its own stream: evals don't change training
    N = cfg.eval_candidates

    @torch.no_grad()
    def evaluate(actor_sampler) -> dict:
        """One eval of one actor: ogbench_data.evaluate's entries plus q_start and q_gain."""
        st = {"q_start": None, "gain": 0.0, "n": 0}

        def act(s_np):            # raw states (evaluate() is called with mean 0 and std 1)
            s = torch.tensor(s_np, device=dev)
            x = torch.randn(len(s) * N, action_dim, device=dev, generator=eval_gen)
            cand = actor_sampler._implicit(x=x, state=((s - mean_t) / std_t).repeat_interleave(N, 0))
            cand = cand.clamp(low, high).view(len(s), N, action_dim)
            q = critic(s[:, None].expand(-1, N, -1), cand)                # (B, N), the trained critic
            qb, best = q.max(1)
            if st["q_start"] is None:
                st["q_start"] = qb.mean().item()
            st["gain"] += (qb - q.mean(1)).sum().item()
            st["n"] += len(s)
            return cand[torch.arange(len(s), device=dev), best].cpu().numpy()

        ev = ogbench_data.evaluate(eval_envs, act, 0.0, 1.0, cfg.test_seed, low, high)
        ev.update(q_start=st["q_start"], q_gain=st["gain"] / max(st["n"], 1))
        return ev

    # ------------------------------------------------------------ run
    run_dir = os.path.join(cfg.checkpoints_path, cfg.name) if cfg.checkpoints_path else None
    full = dict(asdict(cfg), data_info=data_info, frozen_bc_run=frozen_dir, frozen_bc_config=frozen_cfg,
                # Aware-style keys of the trained EqM field, as fit_actor.py saves them (diagnostics/common.py)
                actor=dict(opt_type="nag", step_size=cfg.step_size, num_step=cfg.num_step, moment=cfg.moment,
                           kind="eqm"),
                agent=dict(scale_gradient=cfg.scale_gradient, weighting="bc"), temperature=None)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(full, f, sort_keys=False)
    wandb.init(config=full, project=cfg.project, group=cfg.group, name=cfg.name, id=str(uuid.uuid4()))

    rng = torch.Generator(device=dev).manual_seed(cfg.seed + 1000)        # the batches: its own stream
    prefix = {"trained": "eval", "frozen": "eval_frozen"}
    acc, t_log, step_log, history = {}, time.time(), 0, {k: {} for k in actors}
    try:
        for t in range(1, cfg.num_train_ops):
            do_eval = t % cfg.eval_frequency == 0 or t == cfg.num_train_ops - 1
            do_log = t % cfg.log_frequency == 0 or do_eval
            stats = update(torch.randint(0, n, (cfg.batch_size,), device=dev, generator=rng),
                           t % cfg.stats_every == 0 or do_log)
            for k, v in (stats or {}).items():
                acc.setdefault(k, []).append(v)
            if do_log:
                log = {k: torch.stack(v).mean().item() for k, v in acc.items()}
                log["train/ms_per_step"] = 1000 * (time.time() - t_log) / (t - step_log)
                acc, step_log = {}, t
                if do_eval:
                    for kind, smp in actors.items():
                        te = time.time()
                        ev = evaluate(smp)
                        ev["seconds"] = time.time() - te
                        history[kind][t] = ev["normalized_score"]
                        log.update({(f"{prefix[kind]}_tasks/" if "_task" in k else f"{prefix[kind]}/") + k: v
                                    for k, v in ev.items()})
                        print("RESULT " + json.dumps(dict(
                            env_name=cfg.env_name, seed=cfg.seed, step=t, actor=kind, K=cfg.num_step, N=N,
                            episodes=cfg.n_test_episodes * len(tasks), success=ev["normalized_score"],
                            score=ev["score"], episode_length=ev["episode_length"], q_start=ev["q_start"],
                            q_gain=ev["q_gain"], seconds=ev["seconds"], v_data=log["train/v_data"],
                            q_data=log["train/q_data"], q_loss=log["train/q_loss"], v_loss=log["train/v_loss"],
                            actor_loss=log["train/actor_loss"], ms_per_step=log["train/ms_per_step"])), flush=True)
                wandb.log(log, step=t)
                t_log = time.time()
            if run_dir and (t % cfg.checkpoint_every == 0 or t == cfg.num_train_ops - 1):
                torch.save({"model": model.state_dict(), "value": value.state_dict(), "critic": critic.state_dict(),
                            "target": target.state_dict()}, os.path.join(run_dir, f"checkpoint_{t}.pt"))
        for kind, h in history.items():       # FQL's score: the mean of the last evals (800k, 900k and 1M)
            steps = sorted(h)[-cfg.final_evals:]
            if steps:
                final = float(np.mean([h[s] for s in steps]))
                wandb.run.summary[f"final/{prefix[kind]}_score"] = final
                print("FINAL " + json.dumps(dict(env_name=cfg.env_name, seed=cfg.seed, actor=kind, steps=steps,
                                                 success=final)), flush=True)
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
