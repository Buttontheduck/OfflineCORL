"""A critic for a frozen BC actor of Experiment 10, with QIPO's (and QGPO's) in-support softmax
Q-learning (EqM_RL.md, "Experiment 11"). This is phase 2 of QIPO: the actor stays frozen and only
proposes the candidate actions; the critic is trained alone and saved for a later actor phase.

    Q(s, a) <- r + gamma * (1 - d) * sum_j softmax_j(beta * Q'(s', a'_j)) * Q'(s', a'_j)

  a'_0      the dataset's next action (the next row of the same episode), when there is one
  a'_1..M   M actions sampled once from the frozen BC actor at every next state s' (QGPO's
            precomputed support set). EqM: Aware's sampler with support_K steps from N(0, I);
            Gaussian: draws from N(mean(s'), std), both clipped to the action box.
  Q'        min of the twin target critics (soft update tau). The softmax is over the candidates
            of one state, so beta = 3 makes the target a soft maximum over in-data actions.

Rewards are r - 1 on AntMaze (CORL iql.py), so Q lies in [-100, 0]; d is the dataset's terminal
flag (set at the goal). The critic is IQL's / QGPO's TwinQ: two ReLU MLPs of n_hidden x hidden_dim.

Eval (the critic's use, without training an actor): at every step of the episode, N candidates
from the frozen BC actor (the same sampler), and the one with the highest min(Q1, Q2) is taken.
Logged as eval/normalized_score (success rate in %, same test seeds as fit_actor.py), and
eval/q_start (the chosen candidate's Q at the start states). Training logs, averaged over
log_frequency steps (every stats_every-th batch): the TD loss, Q at dataset actions, and how the softmax used the candidates
(support/w_data: weight on the dataset next action; support/eff_n: exp(entropy); support/q_range:
max - min Q over candidates; support/lift: target minus the target with the dataset next action
alone, i.e. what the soft maximum adds over SARSA).

The BC run is found by name in bc_root (fit_actor.py's run folder, e.g.
Exp10_BC_env_nameantmaze-umaze-v2_actoreqm_seed1-antmaze-umaze-v2-<id>). If its bc_checkpoint
isn't there yet (the BC run is still training), the script waits for it before touching the GPU.

OGBench single-task (env_name like humanoidmaze-medium-navigate-singletask-v0; run with the ogbench conda env):
data from ogbench_data.load_ogbench, rewards used as given (mazes -1 per step, manipulation -(unsolved
subtasks), 0 at the goal), d = the task terminal (1 - masks); the eval runs n_test_episodes episodes of each task
in eval_tasks ('own' = the dataset's task) with ogbench_data.evaluate, eval/normalized_score = success in %.
Options for both benchmarks, off by default (Experiment 11 ran without them):
  layer_norm   LayerNorm after every hidden Linear of both Q networks (as FQL's value networks).
  target_clip  clip the TD target to the range Q can take, [r_min / (1 - gamma), r_max / (1 - gamma)] (with
               rewards <= 0: [r_min / (1 - gamma), 0]); against the soft maximum's runaway overestimation
               that diverged every umaze-diverse critic in Experiment 11.

    python train_critic.py --env_name=antmaze-umaze-v2 --bc_actor=eqm --seed=1
"""

import fcntl
import glob
import os
import random
import subprocess
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

import ogbench_data
from actor_models import Actor
from gaussian_actor import GaussianActor
from offline_data import load_d4rl
from utils.networks import ConditionalMLP


@dataclass
class TrainConfig:
    project: str = "E"
    group: str = "Exp11_Critic"
    name: str = "Critic"
    env_name: str = "antmaze-umaze-v2"
    # the frozen BC actor: <bc_root>/<bc_prefix>_env_name<env>_actor<bc_actor>_seed<seed>-*/<bc_checkpoint>
    bc_root: str = "/workspace/Trained_Models/checkpoints/"
    bc_prefix: str = "Exp10_BC"
    bc_actor: str = "eqm"                 # 'eqm' or 'gaussian'
    bc_checkpoint: str = "checkpoint_500000.pt"
    wait_s: float = 60.0                  # poll interval while the BC checkpoint is missing
    min_free_mb: float = 2500.0           # then also wait until the GPU has this much free memory (nvidia-smi)
    gpu_lock: str = "/tmp/train_critic_gpu.lock"  # critics take turns from that wait until their memory is allocated
    # support set and target
    support_num: int = 16                 # M sampled actions per next state (QIPO / QGPO: 16)
    support_K: int = 10                   # EqM sampler steps for the candidates (Exp 10: K = 10 is the best BC)
    include_data_next: bool = True        # a'_0 = the dataset's next action (QIPO, Algorithm 4)
    beta: float = 3.0                     # softmax inverse temperature of the target (QIPO's guidance scale)
    # critic and optimisation
    hidden_dim: int = 256
    n_hidden: int = 2                     # IQL's / QGPO's TwinQ: 2 hidden layers
    layer_norm: bool = False              # LayerNorm after every hidden Linear (FQL's critics)
    target_clip: bool = False             # clip the TD target to [r_min, r_max] / (1 - gamma) (see the docstring)
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    num_train_ops: int = 1000001          # the extra 1 makes the last eval (and checkpoint) happen at step 1,000,000
    log_frequency: int = 5000
    stats_every: int = 10                 # the train/ and support/ logs average every this many steps' batches
    # evaluation and saving
    eval_frequency: int = 100000          # no eval at step 0 (an untrained critic)
    n_test_episodes: int = 100            # D4RL: episodes per eval; OGBench: episodes per task (OGBench: 50)
    eval_candidates: int = 16             # N candidates per step, the highest-Q one is taken
    eval_tasks: str = "own"               # OGBench only: 'own' (the dataset's task), 'all' or e.g. "1,3"
    ogbench_dir: str = ogbench_data.OGBENCH_DATA
    test_seed: int = 69
    checkpoints_path: Optional[str] = "/workspace/Trained_Models/checkpoints/"
    checkpoint_every: int = 100000
    seed: int = 1                         # also picks the BC run (the critic of seed s uses BC seed s)
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden, layer_norm=False):
        super().__init__()

        def mlp():
            layers, d = [], state_dim + action_dim
            for _ in range(n_hidden):
                layers += [nn.Linear(d, hidden_dim)] + ([nn.LayerNorm(hidden_dim)] if layer_norm else []) + [nn.ReLU()]
                d = hidden_dim
            return nn.Sequential(*layers, nn.Linear(d, 1))

        self.q1, self.q2 = mlp(), mlp()

    def both(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, s, a):
        return torch.min(*self.both(s, a))


def set_seed(seed: int, env=None):   # env: a gym (D4RL) or gymnasium (OGBench) environment
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def find_bc_run(cfg: TrainConfig) -> str:
    pattern = os.path.join(cfg.bc_root,
                           f"{cfg.bc_prefix}_env_name{cfg.env_name}_actor{cfg.bc_actor}_seed{cfg.seed}-{cfg.env_name}-*")
    t0 = time.time()
    while True:
        dirs = sorted(glob.glob(pattern))
        assert len(dirs) <= 1, f"more than one BC run matches {pattern}: {dirs}"
        if dirs:
            f = os.path.join(dirs[0], cfg.bc_checkpoint)
            if os.path.exists(f) and time.time() - os.path.getmtime(f) > 30:   # fully written
                return dirs[0]
        if time.time() - t0 < 1:
            print(f"waiting for {pattern}/{cfg.bc_checkpoint} ...", flush=True)
        time.sleep(cfg.wait_s)


def wait_for_gpu(cfg: TrainConfig):
    """Wait until GPU 0 has min_free_mb free (a critic needs about 1.1 GB), with a random offset so
    critics whose BC runs end together don't all allocate at once."""
    time.sleep(random.uniform(0, 30))
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            free = float(out.splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return
        if free >= cfg.min_free_mb:
            return
        print(f"waiting for GPU memory ({free:.0f} MB free)", flush=True)
        time.sleep(cfg.wait_s + random.uniform(0, 30))


@pyrallis.wrap()
def train(cfg: TrainConfig):
    bc_dir = find_bc_run(cfg)
    # One critic at a time from the memory check until its memory is allocated (released after the
    # first training steps), so critics whose BC runs end together can't all pass the check at once.
    lock = open(cfg.gpu_lock, "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    wait_for_gpu(cfg)
    bc = yaml.safe_load(open(os.path.join(bc_dir, "config.yaml")))
    assert bc["env_name"] == cfg.env_name and bc["actor"]["kind"] == cfg.bc_actor and bc["seed"] == cfg.seed, bc_dir
    print(f"BC run {bc_dir} ({cfg.bc_checkpoint})", flush=True)
    dev = cfg.device
    ogb = ogbench_data.is_ogbench(cfg.env_name)
    if ogb:
        env, data, mean, std, ep, data_info = ogbench_data.load_ogbench(cfg.env_name, cfg.ogbench_dir)
    else:
        env, data, mean, std, ep = load_d4rl(cfg.env_name)
        data_info = dict(benchmark="d4rl", transitions=len(data["observations"]))
    set_seed(cfg.seed, env)
    n, state_dim, action_dim = len(data["observations"]), data["observations"].shape[1], data["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])

    # ------------------------------------------------------------ dataset (normalized states; D4RL AntMaze r - 1)
    norm = lambda x: torch.tensor((x - mean) / std, dtype=torch.float32, device=dev)  # noqa: E731
    S, S2 = norm(data["observations"]), norm(data["next_observations"])
    A = torch.tensor(data["actions"], dtype=torch.float32, device=dev)
    R = torch.tensor(data["rewards"] - (0.0 if ogb else 1.0), dtype=torch.float32, device=dev)
    r_min, r_max = float(R.min()), float(R.max())
    q_lo = r_min / (1 - cfg.gamma) if r_min < 0 else r_min          # the range Q can take (target_clip)
    q_hi = r_max / (1 - cfg.gamma) if r_max > 0 else r_max
    data_info.update(reward_min=r_min, reward_max=r_max, q_range=[q_lo, q_hi])
    D = torch.tensor(data["terminals"], dtype=torch.float32, device=dev)
    # the dataset's next action: the next row, if it belongs to the same episode and starts at s'
    has_next = np.zeros(n, bool)
    has_next[:-1] = (ep[1:] == ep[:-1]) & np.all(np.abs(data["next_observations"][:-1] - data["observations"][1:]) < 1e-6, 1)
    next_idx = np.where(has_next, np.arange(n) + 1, np.arange(n))
    A_next = A[torch.tensor(next_idx, device=dev)]
    DATA_NEXT = torch.tensor(has_next, device=dev)
    HAS_NEXT = DATA_NEXT & cfg.include_data_next   # a'_0 enters the target
    print(f"{n} transitions, {has_next.mean():.1%} with a dataset next action, {int(data['terminals'].sum())} terminals",
          flush=True)

    # ------------------------------------------------------------ frozen BC actor and its proposals
    ck = torch.load(os.path.join(bc_dir, cfg.bc_checkpoint), map_location=dev)
    if cfg.bc_actor == "eqm":
        model = ConditionalMLP(state_dim + action_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"]).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        sampler = Actor(model=model, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low,
                        max_action=high, ebm=None, opt_type=bc["actor"]["opt_type"], step_size=bc["actor"]["step_size"],
                        num_step=cfg.support_K, moment=bc["actor"]["moment"], sampler_type="implicit",
                        ood_threshold=0.01, early_stop=None)

        @torch.no_grad()
        def propose(s, m, gen=None):
            """(B, S) -> (B, m, A): m EqM samples per state."""
            x = torch.randn(len(s) * m, action_dim, device=dev, generator=gen)
            return sampler._implicit(x=x, state=s.repeat_interleave(m, 0)).clamp(low, high).view(len(s), m, action_dim)
    elif cfg.bc_actor == "gaussian":
        model = GaussianActor(state_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"],
                              bc["min_log_std"], bc["max_log_std"]).to(dev)
        model.load_state_dict(ck["actor"])
        model.eval()
        bc_std = model._log_std.clamp(bc["min_log_std"], bc["max_log_std"]).exp()
        print(f"Gaussian BC std per action dim: {np.round(bc_std.tolist(), 3)}", flush=True)

        @torch.no_grad()
        def propose(s, m, gen=None):
            mu = model._mlp(s)[:, None]
            noise = torch.randn(len(s), m, action_dim, device=dev, generator=gen)
            return (mu + bc_std * noise).clamp(low, high)
    else:
        raise ValueError(f"bc_actor must be 'eqm' or 'gaussian', got {cfg.bc_actor!r}")

    # support set at every next state, sampled once (fp16 to save GPU memory: actions are in [-1, 1])
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    SUP = torch.empty(n, cfg.support_num, action_dim, dtype=torch.float16, device=dev)
    chunk = 8192
    spread, nn_dist, nn_dist_rand = [], [], []
    for b in range(0, n, chunk):
        sup = propose(S2[b:b + chunk], cfg.support_num, gen)
        SUP[b:b + chunk] = sup.half()
        spread.append(sup.std(1).mean(-1))
        an, hn = A_next[b:b + chunk], DATA_NEXT[b:b + chunk]
        d = (sup - an[:, None]).norm(dim=-1).min(1).values / action_dim ** 0.5
        nn_dist.append(d[hn])
        # reference: the same distance to the next action of a random other row
        other = A_next[torch.randint(0, n, (len(an),), device=dev, generator=gen)]
        nn_dist_rand.append(((sup - other[:, None]).norm(dim=-1).min(1).values / action_dim ** 0.5)[hn])
    sup_stats = {"support/spread": torch.cat(spread).mean().item(),
                 "support/nn_dist_data": torch.cat(nn_dist).mean().item(),
                 "support/nn_dist_random": torch.cat(nn_dist_rand).mean().item()}
    del sup, an, hn, d, other
    torch.cuda.empty_cache()
    print(f"support set {tuple(SUP.shape)} in {time.time() - t0:.0f}s: "
          + ", ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in sup_stats.items()), flush=True)

    # ------------------------------------------------------------ critic
    critic = TwinQ(state_dim, action_dim, cfg.hidden_dim, cfg.n_hidden, cfg.layer_norm).to(dev)
    target = deepcopy(critic).requires_grad_(False)
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)
    M1 = cfg.support_num + 1

    t_params, c_params = list(target.parameters()), list(critic.parameters())

    def update(idx, with_stats):
        s, a, r, d, s2 = S[idx], A[idx], R[idx], D[idx], S2[idx]
        cand = torch.cat([A_next[idx][:, None], SUP[idx].float()], 1)                 # (B, M + 1, A)
        with torch.no_grad():
            q_next = target(s2[:, None].expand(-1, M1, -1), cand)                         # (B, M + 1)
            valid = torch.ones_like(q_next, dtype=torch.bool)
            valid[:, 0] = HAS_NEXT[idx]
            w = torch.softmax((cfg.beta * q_next).masked_fill(~valid, float("-inf")), 1)
            v_next = (w * q_next).sum(1)
            y = r + cfg.gamma * (1.0 - d) * v_next
            if cfg.target_clip:
                y = y.clamp(q_lo, q_hi)
        q1, q2 = critic.both(s, a)
        loss = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(t_params, 1 - cfg.tau)
            torch._foreach_add_(t_params, c_params, alpha=cfg.tau)
            if not with_stats:
                return None
            hn = valid[:, 0]
            qv = q_next.masked_fill(~valid, float("nan"))
            ent = -(w * torch.log(w.clamp_min(1e-12))).sum(1)
            stats = {"train/q_loss": loss, "train/q_data": torch.min(q1, q2).mean(), "train/target": y.mean(),
                     "support/eff_n": ent.exp().mean(),
                     "support/q_range": (qv.nan_to_num(-1e9).max(1).values - qv.nan_to_num(1e9).min(1).values).mean()}
            if hn.any():
                stats["support/w_data"] = w[hn, 0].mean()
                stats["support/lift"] = (v_next - q_next[:, 0])[hn].mean()
        return stats

    # ------------------------------------------------------------ evaluation: best of N BC candidates
    if ogb:
        tasks = ogbench_data.parse_tasks(cfg.eval_tasks, env)
        eval_envs = ogbench_data.make_eval_envs(cfg.env_name, tasks, cfg.n_test_episodes)
        data_info.update(eval_tasks=tasks, eval_episodes_per_task=cfg.n_test_episodes,
                         score="success rate in %, mean over eval_tasks (OGBench)")
    else:
        import gym

        eval_envs = [env] + [gym.make(cfg.env_name) for _ in range(cfg.n_test_episodes - 1)]

    @torch.no_grad()
    def best_of(s: torch.Tensor):
        """Normalized states (B, S) -> the highest-min(Q1, Q2) of eval_candidates BC candidates, and its Q."""
        cand = propose(s, cfg.eval_candidates)
        q = critic(s[:, None].expand(-1, cfg.eval_candidates, -1), cand)
        return cand[torch.arange(len(s), device=dev), q.argmax(1)], q.max(1).values

    @torch.no_grad()
    def evaluate() -> dict:
        """The log entries of one eval: eval/ (score, normalized_score, q_start) and on OGBench eval_tasks/."""
        if ogb:
            first = {}

            def act(s_np):
                a, q = best_of(torch.tensor(s_np, device=dev))
                first.setdefault("q", q.mean().item())          # at the start states
                return a.cpu().numpy()

            ev = ogbench_data.evaluate(eval_envs, act, mean, std, cfg.test_seed, low, high)
            out = {("eval_tasks/" if "_task" in k else "eval/") + k: v for k, v in ev.items()}
            out["eval/q_start"] = first["q"]
            return out
        obs = []
        for i, e in enumerate(eval_envs):
            e.seed(cfg.test_seed + i)  # old gym API (D4RL): seed before reset
            obs.append(e.reset())
        obs = np.stack(obs).astype(np.float64)
        ret, done, q_start = np.zeros(len(eval_envs)), np.zeros(len(eval_envs), bool), None
        while not done.all():
            active = np.flatnonzero(~done)
            s = torch.tensor((obs[active] - mean) / std, dtype=torch.float32, device=dev)
            a, qb = best_of(s)
            a = a.cpu().numpy()
            if q_start is None:
                q_start = qb.mean().item()
            for k, i in enumerate(active):
                obs[i], r, done[i], _ = eval_envs[i].step(a[k])
                ret[i] += r
        return {"eval/score": ret.mean(), "eval/normalized_score": env.get_normalized_score(ret.mean()) * 100.0,
                "eval/q_start": q_start}

    # ------------------------------------------------------------ run
    run_dir = os.path.join(cfg.checkpoints_path, cfg.name) if cfg.checkpoints_path else None
    full = dict(asdict(cfg), bc_run=bc_dir, bc_config=bc, support_stats=sup_stats, data_info=data_info)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(full, f, sort_keys=False)
    wandb.init(config=full, project=cfg.project, group=cfg.group, name=cfg.name, id=str(uuid.uuid4()))
    wandb.run.summary.update(sup_stats)
    wandb.log(sup_stats, step=0)

    rng = torch.Generator(device=dev).manual_seed(cfg.seed + 1000)
    acc, t_log = {}, time.time()
    try:
        for t in range(1, cfg.num_train_ops):
            stats = update(torch.randint(0, n, (cfg.batch_size,), device=dev, generator=rng), t % cfg.stats_every == 0)
            if t == 200 and not lock.closed:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
            for k, v in (stats or {}).items():
                acc.setdefault(k, []).append(v)
            do_eval = t % cfg.eval_frequency == 0 or t == cfg.num_train_ops - 1
            if t % cfg.log_frequency == 0 or do_eval:
                log = {k: torch.stack(v).mean().item() for k, v in acc.items()}
                log["train/ms_per_step"] = 1000 * (time.time() - t_log) / (cfg.stats_every * len(acc["train/q_loss"]))
                acc = {}
                if do_eval:
                    te = time.time()
                    log.update(evaluate())
                    log["eval/seconds"] = time.time() - te
                    print(f"{t:>8}  success {log['eval/normalized_score']:5.1f}  q_start {log['eval/q_start']:7.2f}  "
                          f"q_data {log['train/q_data']:7.2f}  loss {log['train/q_loss']:.4f}  "
                          f"w_data {log.get('support/w_data', float('nan')):.2f}  eff_n {log['support/eff_n']:.1f}  "
                          f"lift {log.get('support/lift', float('nan')):.2f}  ({log['eval/seconds']:.0f}s eval, "
                          f"{log['train/ms_per_step']:.1f} ms/step)", flush=True)
                wandb.log(log, step=t)
                t_log = time.time()
            if run_dir and (t % cfg.checkpoint_every == 0 or t == cfg.num_train_ops - 1):
                torch.save({"critic": critic.state_dict(), "target": target.state_dict()},
                           os.path.join(run_dir, f"critic_{t}.pt"))
    finally:
        if not lock.closed:
            lock.close()
        wandb.finish()


if __name__ == "__main__":
    train()
