"""Fit an actor alone on fixed per-transition weights, with no critic (EqM_RL.md, "Planned next
experiments", section 2): the 2 x 2 frozen-weights study and the expert-only BC control.

  actor=eqm       the EqM field of Aware (ConditionalMLP), trained with Otter._actor_loss and the
                  given weights, so the EqM loss is exactly Aware's; evaluated with Aware's
                  sampler (Actor.sample: NAG, K steps from N(0, I), clipped).
  actor=gaussian  a Gaussian actor as CORL's awac.py (state-independent std), trained with the
                  weighted log-likelihood -w * log pi(a|s); evaluated with its mean, clipped.
                  Same depth and width as the EqM field (num_hidden_layers x hidden_dim).
  weights=ones    every weight is 1 (behavior cloning); otherwise the weight file
                  <weights_dir>/<weights>_seed<seed>.npz written by make_weights.py.
  data_filter     'all', or 'expert_half' (medium-expert: the expert policy's episodes).

Data loading and state normalization are those of Aware/main.py (the normalization uses the whole
dataset, also with a filter, so every arm sees the same states). EqM checkpoints are
saved as {"model": ...} with an Aware-style config.yaml, so diagnostics/common.py's Policy loads
them; Gaussian checkpoints are {"actor": ...} (no critics) and load with its AwacPolicy, which
rebuilds the saved depth from config.yaml (actor.kind = gaussian).

Two benchmarks, chosen by env_name (run each in its conda env):
  D4RL (hawk)       e.g. antmaze-umaze-v2. n_test_episodes episodes per eval. eval/score = mean return,
                    eval/normalized_score = D4RL's normalized score of that mean (AntMaze: success in %),
                    eval/flip_rate on HalfCheetah (body angle obs[1] beyond flip_angle).
  OGBench (ogbench) a single-task name, e.g. antmaze-large-navigate-singletask-v0 (ogbench_data.py).
                    Data from ogbench_dir. Every eval runs n_test_episodes episodes for each task in
                    eval_tasks (OGBench: 50). eval/normalized_score = success rate in %, the mean over the
                    tasks (OGBench's score); eval/score = mean return (-1 per step until the goal),
                    eval/episode_length. Per task, in the wandb section eval_tasks/: success_task<k> (%),
                    return_task<k>, length_task<k>.
  Both: eval/normalized_score_K<k> for the extra EqM sampler step counts in eval_num_steps, and the
  wandb config holds data_info (dataset size, dims, and for OGBench the tasks and episode limit).

    python fit_actor.py --actor=eqm --weights=iql --seed=1
    python fit_actor.py --actor=gaussian --weights=ones --data_filter=expert_half --seed=1
    python fit_actor.py --actor=eqm --env_name=antmaze-large-navigate-singletask-v0 --n_test_episodes=50 \\
        --eval_frequency=50000      # with /workspace/miniconda3/envs/ogbench/bin/python
"""

import os
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import json
import numpy as np
import pyrallis
import torch
import wandb
import yaml

import ogbench_data
from actor_models import Actor
from agents import Otter
from gaussian_actor import GaussianActor
from offline_data import data_filter, load_d4rl
from utils.networks import ConditionalMLP


@dataclass
class TrainConfig:
    project: str = "E"
    group: str = "exp4_fit_actor"
    name: str = "Fit"
    env_name: str = "halfcheetah-medium-expert-v2"
    actor: str = "eqm"                    # 'eqm' or 'gaussian'
    weights: str = "ones"                 # 'ones' (BC) or a weight-file name: <weights_dir>/<weights>_seed<seed>.npz
    weights_dir: str = "/workspace/Trained_Models/weights"
    data_filter: str = "all"              # 'all' or 'expert_half'
    num_train_ops: int = 310001           # the extra 1 makes the last eval happen at step 310000
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    # EqM field and sampler, as Aware/config.yaml
    scale_gradient: float = 8.0
    flag_type: str = "trun"
    flag_threshold: float = 0.8
    step_size: float = 0.01
    num_step: int = 20
    moment: float = 0.3
    # Gaussian actor, as awac.py
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    # evaluation and saving
    eval_frequency: int = 1000
    n_test_episodes: int = 10             # D4RL: episodes per eval; OGBench: episodes per task (OGBench uses 50)
    test_seed: int = 69
    eval_tasks: str = "all"               # OGBench only: 'all' (tasks 1-5), 'own' (the dataset's task) or e.g. "1,3"
    ogbench_dir: str = ogbench_data.OGBENCH_DATA   # OGBench only: where the datasets are stored
    flip_angle: float = 2.0               # HalfCheetah only: an episode 'flips' when |obs[1]| exceeds it
    eval_num_steps: str = ""              # EqM only: extra sampler step counts K to evaluate at, e.g. "10,30";
                                          # logged as eval/normalized_score_K<k>. The main score uses num_step.
    checkpoints_path: Optional[str] = "/workspace/Trained_Models/checkpoints/"
    checkpoint_every: int = 10000
    seed: int = 1
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def set_seed(seed: int, env=None):   # env: a gym (D4RL) or gymnasium (OGBench) environment
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def weight_stats(w: np.ndarray, cap: float = 100.0) -> dict:
    rng = np.random.default_rng(0)
    wb = w[rng.integers(0, len(w), (1000, 256))]
    m, var = wb.mean(1), wb.var(1)
    return dict(batch_share=float((m ** 2 / (m ** 2 + var)).mean()), cap_share=float((w >= cap).mean()),
                near_zero_share=float((w < 0.01).mean()), median=float(np.median(w)), mean=float(w.mean()))


@pyrallis.wrap()
def train(cfg: TrainConfig):
    dev = cfg.device
    ogb = ogbench_data.is_ogbench(cfg.env_name)
    if ogb:
        env, data, mean, std, ep, data_info = ogbench_data.load_ogbench(cfg.env_name, cfg.ogbench_dir)
        assert cfg.data_filter == "all", "OGBench: only data_filter='all'"
    else:
        env, data, mean, std, ep = load_d4rl(cfg.env_name)
        data_info = dict(benchmark="d4rl", transitions=len(data["observations"]), episodes=int(ep.max()) + 1,
                         state_dim=int(data["observations"].shape[1]), action_dim=int(data["actions"].shape[1]))
    print(f"data: {data_info}", flush=True)
    set_seed(cfg.seed, env)
    state_dim, action_dim = data["observations"].shape[1], data["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])

    # ------------------------------------------------------------ weights and data subset
    n = len(data["observations"])
    if cfg.weights == "ones":
        w, wmeta = np.ones(n, np.float32), dict(source="ones")
    else:
        f = np.load(os.path.join(cfg.weights_dir, f"{cfg.weights}_seed{cfg.seed}.npz"))
        w, wmeta = f["weights"], json.loads(str(f["meta"]))
        assert wmeta["env_name"] == cfg.env_name and len(w) == n, "weight file doesn't match the dataset"
    mask, frep = data_filter(cfg.data_filter, ep, data["rewards"])
    print(f"data filter {cfg.data_filter}: {mask.sum()} of {n} transitions; {frep}")
    S = torch.tensor((data["observations"][mask] - mean) / std, dtype=torch.float32, device=dev)
    A = torch.tensor(data["actions"][mask], dtype=torch.float32, device=dev)
    W = torch.tensor(w[mask], dtype=torch.float32, device=dev)[:, None]
    wst = weight_stats(w[mask])
    print(f"weights {cfg.weights}: {wst}")

    # ------------------------------------------------------------ actor
    if cfg.actor == "eqm":
        model = ConditionalMLP(state_dim + action_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        sampler = Actor(model=model, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low,
                        max_action=high, ebm=None, opt_type="nag", step_size=cfg.step_size, num_step=cfg.num_step,
                        moment=cfg.moment, sampler_type="implicit", ood_threshold=0.01, early_stop=None)
        agent = Otter(model=model, model_optimizer=opt, actor=sampler, critic_1=None, critic_1_optimizer=None,
                      critic_2=None, critic_2_optimizer=None, flag_type=cfg.flag_type,
                      flag_threshold=cfg.flag_threshold, scale_gradient=cfg.scale_gradient, num_actions=1)

        def loss_fn(s, a, wt):
            return agent._actor_loss(s, a, weights=wt)

        @torch.no_grad()
        def act(s_np):
            return sampler.sample(s_np)

        def ckpt_state():
            return {"model": model.state_dict()}
    elif cfg.actor == "gaussian":
        model = GaussianActor(state_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers,
                              cfg.min_log_std, cfg.max_log_std).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        def loss_fn(s, a, wt):
            return (-model.log_prob(s, a) * wt.squeeze(-1)).mean()

        @torch.no_grad()
        def act(s_np):
            s = torch.tensor(s_np, dtype=torch.float32, device=dev)
            return model._mlp(s).clamp(low, high).cpu().numpy()

        def ckpt_state():
            return {"actor": model.state_dict()}
    else:
        raise ValueError(f"actor must be 'eqm' or 'gaussian', got {cfg.actor!r}")

    # ------------------------------------------------------------ evaluation (episodes side by side)
    track_flips = "halfcheetah" in cfg.env_name
    extra_K = [int(k) for k in cfg.eval_num_steps.split(",") if k.strip()] if cfg.actor == "eqm" else []
    if ogb:
        tasks = ogbench_data.parse_tasks(cfg.eval_tasks, env)
        eval_envs = ogbench_data.make_eval_envs(cfg.env_name, tasks, cfg.n_test_episodes)
        data_info.update(eval_tasks=tasks, eval_episodes_per_task=cfg.n_test_episodes,
                         score="success rate in %, mean over eval_tasks (OGBench)")
    else:
        import gym

        eval_envs = [env] + [gym.make(cfg.env_name) for _ in range(cfg.n_test_episodes - 1)]
        data_info.update(eval_episodes=cfg.n_test_episodes, score="D4RL normalized score of the mean return")

    def evaluate_at(K: int) -> dict:
        """evaluate() with the EqM sampler set to K steps."""
        sampler.num_step = K
        try:
            return evaluate()
        finally:
            sampler.num_step = cfg.num_step

    def evaluate() -> dict:
        """The eval/ log entries (without the prefix) of one evaluation."""
        model.eval()
        try:
            if ogb:
                return ogbench_data.evaluate(eval_envs, act, mean, std, cfg.test_seed, low, high)
            obs = []
            for i, e in enumerate(eval_envs):
                e.seed(cfg.test_seed + i)  # old gym API (D4RL): seed before reset
                obs.append(e.reset())
            obs = np.stack(obs).astype(np.float64)
            ret, done = np.zeros(len(eval_envs)), np.zeros(len(eval_envs), bool)
            flipped = np.zeros(len(eval_envs), bool)
            while not done.all():
                active = np.flatnonzero(~done)
                a = act(((obs[active] - mean) / std).astype(np.float32))
                for k, i in enumerate(active):
                    obs[i], r, done[i], _ = eval_envs[i].step(a[k])
                    ret[i] += r
                    flipped[i] |= abs(obs[i][1]) > cfg.flip_angle
            out = dict(score=ret.mean(), score_std=ret.std(), score_min=ret.min(),
                       normalized_score=env.get_normalized_score(ret.mean()) * 100.0)
            if track_flips:
                out["flip_rate"] = flipped.mean()
            return out
        finally:
            model.train()

    # ------------------------------------------------------------ run
    full = asdict(cfg)
    full.update(weights_meta=wmeta, weights_stats=wst, data_filter_report=frep, data_info=data_info)
    if cfg.checkpoints_path is not None:
        os.makedirs(cfg.checkpoints_path, exist_ok=True)
        saved = dict(full, hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers, gamma=0.99,
                     temperature=wmeta.get("temperature"),
                     actor=dict(opt_type="nag", step_size=cfg.step_size, num_step=cfg.num_step, moment=cfg.moment,
                                kind=cfg.actor),
                     agent=dict(scale_gradient=cfg.scale_gradient, weighting=f"fixed:{cfg.weights}"))
        with open(os.path.join(cfg.checkpoints_path, "config.yaml"), "w") as f:
            yaml.safe_dump(saved, f, sort_keys=False)
    wandb.init(config=full, project=cfg.project, group=cfg.group, name=cfg.name, id=str(uuid.uuid4()))
    wandb.run.summary.update({f"weights/{k}": v for k, v in wst.items()})

    rng = torch.Generator(device=dev).manual_seed(cfg.seed)
    losses = []
    try:
        for t in range(cfg.num_train_ops):
            idx = torch.randint(0, len(S), (cfg.batch_size,), device=dev, generator=rng)
            loss = loss_fn(S[idx], A[idx], W[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if t % cfg.eval_frequency == 0:
                ev = evaluate()
                # per-task entries (OGBench) in their own wandb section, eval_tasks/; the headline numbers in eval/
                log = {"train/actor_loss": float(np.mean(losses)),
                       **{(f"eval_tasks/{k}" if "_task" in k else f"eval/{k}"): v for k, v in ev.items()}}
                for K in extra_K:
                    log[f"eval/normalized_score_K{K}"] = evaluate_at(K)["normalized_score"]
                wandb.log(log, step=t)
                losses = []
                per_task = "".join(f" t{k[len('success_task'):]} {v:.0f}" for k, v in ev.items()
                                   if k.startswith("success_task"))
                print(f"{t:>7}  score {log['eval/score']:8.2f}  normalized {log['eval/normalized_score']:6.1f}"
                      + (f"  (per task:{per_task})" if per_task else "")
                      + (f"  flips {log['eval/flip_rate']:.0%}" if track_flips else "")
                      + "".join(f"  K={K}: {log[f'eval/normalized_score_K{K}']:.1f}" for K in extra_K), flush=True)
            if cfg.checkpoints_path is not None and (t % cfg.checkpoint_every == 0 or t == cfg.num_train_ops - 1):
                torch.save(ckpt_state(), os.path.join(cfg.checkpoints_path, f"checkpoint_{t}.pt"))
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
