"""QIPO phase 3 for an EqM actor (EqM_RL.md, "Experiment 12"): fine-tune a frozen-critic-weighted
EqM-BC actor. The Experiment 10 EqM-BC actor and the Experiment 11 critic trained on it (same seed)
are loaded; the critic stays frozen, the EqM field is trained with QIPO's Q-weighted loss:

    L = (1/B) sum_i sum_j g_ij * EqM loss(s_i, a_ij),    g_ij = softmax_j(beta * Q(s_i, a_ij)),  j = 0..M

  a_i0      the dataset action at s_i
  a_i1..M   M actions sampled from the current actor at every dataset state (support_K sampler steps),
            renewed every renew_every steps (QIPO's support set and K_renew); between renewals the
            support set and its weights g stay fixed, as in QIPO's Algorithm 4
  Q         min(Q1, Q2) of the Experiment 11 critic, frozen

The EqM loss is Aware's (Otter._actor_loss) with per-row weights (M + 1) * g_ij, so the loss is
QIPO's sum over j, averaged over the batch states. After l renewals the target policy is
pi_l ~ mu * exp(l * beta * Q) (QIPO eq. 5.4): each renewal tilts it further. beta = 0 gives uniform
weights: the control for training on the actor's own samples with no Q.

Two pulls toward the dataset actions (Experiment 16; the defaults give plain QIPO). Experiment 14 showed the
fine-tuned field collapsing to about one action per state (the EqM sampler descends toward minima, and training on
its own samples compounds that), with the dataset action holding only 6-12 % of the weight:
  data_mult  the dataset action counts as data_mult candidates: g_i = softmax(beta * Q_i + log(data_mult) * e_0),
             so its share still depends on Q (data_mult = M: data and samples weigh the same at equal Q)
  bc_coef    + bc_coef * the plain EqM-BC loss on a separate batch of dataset (s, a), independent of Q
             (logged as train/bc_loss; train/actor_loss stays the QIPO part)

Adam at learning_rate (QIPO fine-tunes at 1e-4, fresh optimizer). An EMA copy of the field
(ema_tau, QIPO: 0.005) is what gets evaluated and checkpointed.

Eval (EMA field, n_test_episodes episodes, the test seeds of fit_actor.py and train_critic.py): one
sample per step at each K in eval_K (eval/normalized_score for the first, eval/normalized_score_K<k>
for the others), and every eval_best_every steps the best of eval_candidates samples (support_K
steps) by the critic (eval/normalized_score_best<N>). Step 0 is the BC actor itself.
At every renewal, ft/* logs how the critic sees the support set: Q at the dataset action and at the
actor's samples, the weight on the dataset action, the effective number of candidates, and how far
the samples are from the dataset action and the action box.

OGBench single-task (env_name like cube-double-play-singletask-v0; run with the ogbench conda env): data from
ogbench_data.load_ogbench, evals with ogbench_data.evaluate on the tasks in eval_tasks ('own' = the dataset's task,
whose rewards the critic learned), eval/normalized_score = success in %. The critic is rebuilt with the layer_norm
setting it was trained with (train_critic.py).

    python finetune_actor.py --env_name=antmaze-medium-diverse-v2 --beta=3 --seed=1
"""

import fcntl
import math
import glob
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import wandb
import yaml

import ogbench_data
from actor_models import Actor
from agents import Otter
from offline_data import load_d4rl
from train_critic import TwinQ, set_seed, wait_for_gpu
from utils.networks import ConditionalMLP


@dataclass
class TrainConfig:
    project: str = "E"
    group: str = "Exp12_FT"
    name: str = "FT"
    env_name: str = "antmaze-umaze-v2"
    # the frozen critic: <critic_root>/<critic_prefix>_env_name<env>_bc_actor<bc_actor>_seed<seed>-*/<critic_checkpoint>;
    # the actor starts from the BC checkpoint that critic was trained on (its config.yaml: bc_run, bc_checkpoint)
    critic_root: str = "/workspace/Trained_Models/checkpoints/"
    critic_prefix: str = "Exp11_Critic"
    bc_actor: str = "eqm"
    critic_checkpoint: str = "critic_1000000.pt"
    critic_key: str = "critic"            # 'critic' (online weights, as Exp 11's eval) or 'target' (EMA)
    # QIPO phase 3
    support_num: int = 16                 # M samples per state (QIPO: 16)
    support_K: int = 10                   # EqM sampler steps for the support samples (Exp 10/11: K = 10)
    beta: float = 3.0                     # softmax inverse temperature (QIPO: 3; 10 in its ablation; 0 = no Q)
    data_mult: float = 1.0                # the dataset action counts as this many candidates (1 = QIPO)
    bc_coef: float = 0.0                  # weight of an added plain EqM-BC loss on dataset actions (0 = QIPO)
    renew_every: int = 40000              # steps between support renewals (QIPO K_renew = 10 epochs; 1 epoch ~ 3.9k steps)
    num_train_ops: int = 400001           # ~100 epochs (QIPO K3 = 100); the extra 1 puts the last eval at 400k
    batch_size: int = 256                 # states per batch; each brings its M + 1 candidates
    learning_rate: float = 1e-4           # QIPO's fine-tuning learning rate
    ema_tau: float = 0.005                # QIPO's soft update of the policy network
    log_frequency: int = 5000
    stats_every: int = 10
    # evaluation and saving
    eval_frequency: int = 20000           # ~5 epochs (QIPO evaluates every 5 epochs)
    eval_K: str = "10,20"                 # one sample per step at these sampler steps; the first is the main score
    eval_best_every: int = 100000         # and the critic's best of eval_candidates samples
    eval_candidates: int = 16
    n_test_episodes: int = 100            # D4RL: episodes per eval; OGBench: episodes per task (OGBench: 50)
    test_seed: int = 69
    eval_tasks: str = "own"               # OGBench only: 'own' (the dataset's task), 'all' or e.g. "1,3"
    ogbench_dir: str = ogbench_data.OGBENCH_DATA
    checkpoints_path: Optional[str] = "/workspace/Trained_Models/checkpoints/"
    checkpoint_every: int = 100000
    wait_s: float = 60.0
    min_free_mb: float = 2500.0           # wait for this much free GPU memory before allocating (nvidia-smi)
    gpu_lock: str = "/tmp/train_critic_gpu.lock"  # runs take turns from that check until their memory is allocated
    seed: int = 1                         # picks the critic (and through it the BC actor) of this seed
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"


def find_critic(cfg: TrainConfig) -> str:
    pattern = os.path.join(cfg.critic_root, f"{cfg.critic_prefix}_env_name{cfg.env_name}_bc_actor{cfg.bc_actor}"
                                            f"_seed{cfg.seed}-{cfg.env_name}-*")
    dirs = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, cfg.critic_checkpoint))]
    assert len(dirs) == 1, f"expected one critic run with {cfg.critic_checkpoint} for {pattern}, found {dirs}"
    return dirs[0]


@pyrallis.wrap()
def train(cfg: TrainConfig):
    crit_dir = find_critic(cfg)
    ccfg = yaml.safe_load(open(os.path.join(crit_dir, "config.yaml")))
    bc_dir, bc = ccfg["bc_run"], ccfg["bc_config"]
    assert ccfg["env_name"] == cfg.env_name and ccfg["seed"] == cfg.seed and bc["actor"]["kind"] == "eqm", crit_dir
    print(f"critic {crit_dir}/{cfg.critic_checkpoint} ({cfg.critic_key})\nBC actor {bc_dir}/{ccfg['bc_checkpoint']}",
          flush=True)
    lock = open(cfg.gpu_lock, "a")   # one run at a time from the memory check until its memory is allocated
    fcntl.flock(lock, fcntl.LOCK_EX)
    wait_for_gpu(cfg)

    dev = cfg.device
    ogb = ogbench_data.is_ogbench(cfg.env_name)
    if ogb:
        env, data, mean, std, _, data_info = ogbench_data.load_ogbench(cfg.env_name, cfg.ogbench_dir)
    else:
        env, data, mean, std, _ = load_d4rl(cfg.env_name)
        data_info = dict(benchmark="d4rl", transitions=len(data["observations"]))
    set_seed(cfg.seed, env)
    n, state_dim, action_dim = len(data["observations"]), data["observations"].shape[1], data["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    S = torch.tensor((data["observations"] - mean) / std, dtype=torch.float32, device=dev)
    A = torch.tensor(data["actions"], dtype=torch.float32, device=dev)
    M, M1 = cfg.support_num, cfg.support_num + 1

    # ------------------------------------------------------------ frozen critic, actor, EMA
    critic = TwinQ(state_dim, action_dim, ccfg["hidden_dim"], ccfg["n_hidden"], ccfg.get("layer_norm", False)).to(dev)
    critic.load_state_dict(torch.load(os.path.join(crit_dir, cfg.critic_checkpoint), map_location=dev)[cfg.critic_key])
    critic.eval().requires_grad_(False)
    model = ConditionalMLP(state_dim + action_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(bc_dir, ccfg["bc_checkpoint"]), map_location=dev)["model"])
    ema = deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    samplers = {}

    def sampler(m, K):
        key = (id(m), K)
        if key not in samplers:
            samplers[key] = Actor(model=m, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low,
                                  max_action=high, ebm=None, opt_type=bc["actor"]["opt_type"],
                                  step_size=bc["actor"]["step_size"], num_step=K, moment=bc["actor"]["moment"],
                                  sampler_type="implicit", ood_threshold=0.01, early_stop=None)
        return samplers[key]

    @torch.no_grad()
    def sample(m, s, k, K, gen=None):
        """(B, S) -> (B, k, A): k EqM samples per state with K sampler steps, clipped to the box."""
        x = torch.randn(len(s) * k, action_dim, device=dev, generator=gen)
        return sampler(m, K)._implicit(x=x, state=s.repeat_interleave(k, 0)).clamp(low, high).view(len(s), k, action_dim)

    agent = Otter(model=model, model_optimizer=opt, actor=sampler(model, cfg.support_K), critic_1=None,
                  critic_1_optimizer=None, critic_2=None, critic_2_optimizer=None, flag_type=bc["flag_type"],
                  flag_threshold=bc["flag_threshold"], scale_gradient=bc["scale_gradient"], num_actions=1)

    # ------------------------------------------------------------ support set and its weights, renewed
    SUP = torch.empty(n, M, action_dim, dtype=torch.float16, device=dev)
    G = torch.empty(n, M1, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    @torch.no_grad()
    def renew():
        t0, acc = time.time(), {}
        for b in range(0, n, 4096):
            s = S[b:b + 4096]
            sup = sample(model, s, M, cfg.support_K, gen)
            cand = torch.cat([A[b:b + 4096, None], sup], 1)                                  # (c, M + 1, A)
            q = critic(s[:, None].expand(-1, M1, -1), cand)                                  # (c, M + 1)
            logits = cfg.beta * q
            logits[:, 0] += math.log(cfg.data_mult)                                          # the dataset action
            g = torch.softmax(logits, 1)
            SUP[b:b + 4096], G[b:b + 4096] = sup.half(), g
            for k, v in {"q_data": q[:, 0], "q_samples": q[:, 1:].mean(1), "q_best": q.max(1).values,
                         "q_weighted": (g * q).sum(1), "w_data": g[:, 0],
                         "eff_n": torch.exp(-(g * torch.log(g.clamp_min(1e-12))).sum(1)),
                         "spread": sup.std(1).mean(-1),
                         "dist_data": (sup - cand[:, :1]).norm(dim=-1).min(1).values / action_dim ** 0.5,
                         "at_bound": (sup.abs() > 0.99).float().mean((1, 2))}.items():
                acc.setdefault(k, []).append(v.sum())
        stats = {f"ft/{k}": torch.stack(v).sum().item() / n for k, v in acc.items()}
        stats["ft/renew_seconds"] = time.time() - t0
        torch.cuda.empty_cache()
        return stats

    # ------------------------------------------------------------ evaluation (EMA field)
    if ogb:
        tasks = ogbench_data.parse_tasks(cfg.eval_tasks, env)
        eval_envs = ogbench_data.make_eval_envs(cfg.env_name, tasks, cfg.n_test_episodes)
        data_info.update(eval_tasks=tasks, eval_episodes_per_task=cfg.n_test_episodes,
                         score="success rate in %, mean over eval_tasks (OGBench)")
    else:
        import gym

        eval_envs = [env] + [gym.make(cfg.env_name) for _ in range(cfg.n_test_episodes - 1)]
    eval_K = [int(k) for k in cfg.eval_K.split(",") if k.strip()]

    @torch.no_grad()
    def policy(s, K, best_of):
        """Normalized states (B, S) -> actions of the EMA field: one sample at K steps, or the critic's best of N."""
        if best_of:
            cand = sample(ema, s, best_of, K)
            q = critic(s[:, None].expand(-1, best_of, -1), cand)
            return cand[torch.arange(len(s), device=dev), q.argmax(1)]
        return sample(ema, s, 1, K)[:, 0]

    @torch.no_grad()
    def evaluate(K, best_of=0) -> dict:
        """{'normalized_score': ..., 'score': mean return, ...} of one eval."""
        if ogb:
            return ogbench_data.evaluate(
                eval_envs, lambda s_np: policy(torch.tensor(s_np, device=dev), K, best_of).cpu().numpy(),
                mean, std, cfg.test_seed, low, high)
        obs = []
        for i, e in enumerate(eval_envs):
            e.seed(cfg.test_seed + i)  # old gym API (D4RL): seed before reset
            obs.append(e.reset())
        obs = np.stack(obs).astype(np.float64)
        ret, done = np.zeros(len(eval_envs)), np.zeros(len(eval_envs), bool)
        while not done.all():
            active = np.flatnonzero(~done)
            s = torch.tensor((obs[active] - mean) / std, dtype=torch.float32, device=dev)
            a = policy(s, K, best_of).cpu().numpy()
            for k, i in enumerate(active):
                obs[i], r, done[i], _ = eval_envs[i].step(a[k])
                ret[i] += r
        return {"normalized_score": env.get_normalized_score(ret.mean()) * 100.0, "score": ret.mean()}

    def eval_log(t):
        te, log = time.time(), {}
        for j, K in enumerate(eval_K):
            ev = evaluate(K)
            log["eval/normalized_score" if j == 0 else f"eval/normalized_score_K{K}"] = ev["normalized_score"]
            if j == 0:   # the main K: also the return (and on OGBench the episode length)
                log.update({f"eval/{k}": ev[k] for k in ("score", "episode_length") if k in ev})
        if t % cfg.eval_best_every == 0 or t == cfg.num_train_ops - 1:
            log[f"eval/normalized_score_best{cfg.eval_candidates}"] = \
                evaluate(cfg.support_K, cfg.eval_candidates)["normalized_score"]
        log["eval/seconds"] = time.time() - te
        print(f"{t:>8}  " + "  ".join(f"{k.split('/')[1]} {v:.1f}" for k, v in log.items()), flush=True)
        return log

    # ------------------------------------------------------------ run
    run_dir = os.path.join(cfg.checkpoints_path, cfg.name) if cfg.checkpoints_path else None
    full = dict(asdict(cfg), critic_run=crit_dir, bc_run=bc_dir, bc_checkpoint=ccfg["bc_checkpoint"], bc_config=bc,
                data_info=data_info)
    if run_dir:   # Aware-style keys too, so diagnostics/common.py can load the EMA field as an EqM policy
        os.makedirs(run_dir, exist_ok=True)
        saved = dict(full, hidden_dim=bc["hidden_dim"], num_hidden_layers=bc["num_hidden_layers"], gamma=0.99,
                     actor=dict(opt_type=bc["actor"]["opt_type"], step_size=bc["actor"]["step_size"],
                                num_step=eval_K[0], moment=bc["actor"]["moment"], kind="eqm"),
                     agent=dict(scale_gradient=bc["scale_gradient"], weighting=f"qipo:beta{cfg.beta}:data_mult{cfg.data_mult}:bc_coef{cfg.bc_coef}"))
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(saved, f, sort_keys=False)
    wandb.init(config=full, project=cfg.project, group=cfg.group, name=cfg.name, id=str(uuid.uuid4()))

    stats = renew()
    torch.cuda.empty_cache()
    fcntl.flock(lock, fcntl.LOCK_UN)   # memory is allocated: the next run may start
    lock.close()
    print("renewal 0: " + ", ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in stats.items()), flush=True)
    log = dict(stats, **eval_log(0))
    wandb.log(log, step=0)

    t_params, e_params = list(model.parameters()), list(ema.parameters())
    rng = torch.Generator(device=dev).manual_seed(cfg.seed + 2000)
    losses, bc_losses, t_log = [], [], time.time()
    ones_b = torch.ones(cfg.batch_size, 1, device=dev)
    try:
        for t in range(1, cfg.num_train_ops):
            log = {}
            if t % cfg.renew_every == 0:
                log.update(renew())
                print(f"renewal at {t}: " + ", ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in log.items()), flush=True)
            idx = torch.randint(0, n, (cfg.batch_size,), device=dev, generator=rng)
            s = S[idx][:, None].expand(-1, M1, -1).reshape(-1, state_dim)
            a = torch.cat([A[idx][:, None], SUP[idx].float()], 1).reshape(-1, action_dim)
            loss = agent._actor_loss(s, a, weights=(G[idx] * M1).reshape(-1, 1))
            total = loss
            if cfg.bc_coef > 0:   # plain EqM-BC on a separate batch of dataset (s, a)
                jdx = torch.randint(0, n, (cfg.batch_size,), device=dev, generator=rng)
                bc_loss = agent._actor_loss(S[jdx], A[jdx], weights=ones_b)
                total = loss + cfg.bc_coef * bc_loss
            opt.zero_grad()
            total.backward()
            opt.step()
            with torch.no_grad():
                torch._foreach_mul_(e_params, 1 - cfg.ema_tau)
                torch._foreach_add_(e_params, t_params, alpha=cfg.ema_tau)
            if t % cfg.stats_every == 0:
                losses.append(loss.detach())
                if cfg.bc_coef > 0:
                    bc_losses.append(bc_loss.detach())
            do_eval = t % cfg.eval_frequency == 0 or t == cfg.num_train_ops - 1
            if t % cfg.log_frequency == 0 or do_eval or log:
                if losses:
                    log["train/actor_loss"] = torch.stack(losses).mean().item()
                    log["train/ms_per_step"] = 1000 * (time.time() - t_log) / (cfg.stats_every * len(losses))
                    losses = []
                if bc_losses:
                    log["train/bc_loss"] = torch.stack(bc_losses).mean().item()
                    bc_losses = []
                if do_eval:
                    log.update(eval_log(t))
                wandb.log(log, step=t)
                t_log = time.time()
            if run_dir and (t % cfg.checkpoint_every == 0 or t == cfg.num_train_ops - 1):
                torch.save({"model": ema.state_dict(), "model_raw": model.state_dict()},
                           os.path.join(run_dir, f"checkpoint_{t}.pt"))
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
