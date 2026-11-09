"""Experiment 15: eval-only best-of-N on OGBench with the frozen Experiment 14 critics and EqM-BC actors, no training.

For every sampler step count K in Ks and every N in Ns: at each environment step, N samples of the frozen EqM-BC
actor (K sampler steps from N(0, I), clipped to the action box) are scored by the frozen critic (min(Q1, Q2)) and
the highest is taken. Same actor, critic, sampler, eval environments and test seeds as Experiment 14's
phase-3 step-0 eval (K = 20, N = 16: humanoidmaze-medium 30 / 34 / 56 % over seeds 1-3, 50 episodes). With more
episodes the first 50 start states are the same ones (ogbench_data.evaluate seeds episode j with test_seed + 1000 k + j).

Prints one line 'RESULT {json}' per (K, N): success %, return, episode length, the chosen action's Q at the start
states (q_start), and q_gain = mean over all steps of Q(chosen) - mean Q(candidates), i.e. how much the critic
separates the candidates. diagnostics/summarize_exp15.py reads these lines from the sweep's stdout.log files.
No wandb, no checkpoints.

    python eval_best_of_n.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --seed=1
"""

import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pyrallis
import torch
import yaml

import ogbench_data
from actor_models import Actor
from train_critic import TwinQ, set_seed
from utils.networks import ConditionalMLP


@dataclass
class EvalConfig:
    group: str = "Exp15_BestN"
    name: str = "BestN"
    env_name: str = "humanoidmaze-medium-navigate-singletask-v0"
    # the frozen critic: <critic_root>/<critic_prefix>_env_name<env>_bc_actor<bc_actor>_seed<seed>-*/<critic_checkpoint>;
    # the BC actor is the checkpoint that critic was trained on (its config.yaml: bc_run, bc_checkpoint)
    critic_root: str = "/workspace/Trained_Models/checkpoints/"
    critic_prefix: str = "Exp14_Critic"
    bc_actor: str = "eqm"
    critic_checkpoint: str = "critic_1000000.pt"
    critic_key: str = "critic"            # the online weights, as Experiment 14's evals
    Ks: str = "20,30"                     # EqM sampler steps
    Ns: str = "16,32,64"                  # candidates per environment step
    n_test_episodes: int = 50             # per task
    test_seed: int = 69                   # as fit_actor.py, train_critic.py, finetune_actor.py
    eval_tasks: str = "own"               # 'own' (the dataset's task, whose rewards the critic learned), 'all' or "1,3"
    ogbench_dir: str = ogbench_data.OGBENCH_DATA
    seed: int = 1                         # picks the critic (and through it the BC actor) of this seed
    device: str = "cuda"


def find_critic(cfg: EvalConfig) -> str:
    pattern = os.path.join(cfg.critic_root, f"{cfg.critic_prefix}_env_name{cfg.env_name}_bc_actor{cfg.bc_actor}"
                                            f"_seed{cfg.seed}-{cfg.env_name}-*")
    dirs = [d for d in sorted(glob.glob(pattern)) if os.path.exists(os.path.join(d, cfg.critic_checkpoint))]
    assert len(dirs) == 1, f"expected one critic run with {cfg.critic_checkpoint} for {pattern}, found {dirs}"
    return dirs[0]


@pyrallis.wrap()
def main(cfg: EvalConfig):
    crit_dir = find_critic(cfg)
    ccfg = yaml.safe_load(open(os.path.join(crit_dir, "config.yaml")))
    bc_dir, bc = ccfg["bc_run"], ccfg["bc_config"]
    assert ccfg["env_name"] == cfg.env_name and ccfg["seed"] == cfg.seed and bc["actor"]["kind"] == "eqm", crit_dir
    print(f"critic {crit_dir}/{cfg.critic_checkpoint} ({cfg.critic_key})\nBC actor {bc_dir}/{ccfg['bc_checkpoint']}",
          flush=True)

    dev = cfg.device
    env, data, mean, std, _, _ = ogbench_data.load_ogbench(cfg.env_name, cfg.ogbench_dir)
    set_seed(cfg.seed, env)
    state_dim, action_dim = data["observations"].shape[1], data["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    del data

    critic = TwinQ(state_dim, action_dim, ccfg["hidden_dim"], ccfg["n_hidden"], ccfg.get("layer_norm", False)).to(dev)
    critic.load_state_dict(torch.load(os.path.join(crit_dir, cfg.critic_checkpoint), map_location=dev)[cfg.critic_key])
    critic.eval().requires_grad_(False)
    model = ConditionalMLP(state_dim + action_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(bc_dir, ccfg["bc_checkpoint"]), map_location=dev)["model"])
    model.eval().requires_grad_(False)

    def sampler(K):
        return Actor(model=model, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low,
                     max_action=high, ebm=None, opt_type=bc["actor"]["opt_type"], step_size=bc["actor"]["step_size"],
                     num_step=K, moment=bc["actor"]["moment"], sampler_type="implicit", ood_threshold=0.01,
                     early_stop=None)

    tasks = ogbench_data.parse_tasks(cfg.eval_tasks, env)
    eval_envs = ogbench_data.make_eval_envs(cfg.env_name, tasks, cfg.n_test_episodes)
    Ks = [int(k) for k in cfg.Ks.split(",") if k.strip()]
    Ns = [int(n) for n in cfg.Ns.split(",") if n.strip()]
    print(f"{cfg.env_name} seed {cfg.seed}: tasks {tasks}, {cfg.n_test_episodes} episodes each, K {Ks}, N {Ns}",
          flush=True)

    for K in Ks:
        act_sampler = sampler(K)
        for N in Ns:
            torch.manual_seed(cfg.seed)
            stats = {"q_start": None, "gain_sum": 0.0, "gain_n": 0}

            @torch.no_grad()
            def act(s_np):
                s = torch.tensor(s_np, device=dev)
                x = torch.randn(len(s) * N, action_dim, device=dev)
                cand = act_sampler._implicit(x=x, state=s.repeat_interleave(N, 0)).clamp(low, high)
                cand = cand.view(len(s), N, action_dim)
                q = critic(s[:, None].expand(-1, N, -1), cand)                  # (B, N)
                best = q.argmax(1)
                qb = q.max(1).values
                if stats["q_start"] is None:
                    stats["q_start"] = qb.mean().item()
                stats["gain_sum"] += (qb - q.mean(1)).sum().item()
                stats["gain_n"] += len(s)
                return cand[torch.arange(len(s), device=dev), best].cpu().numpy()

            t0 = time.time()
            ev = ogbench_data.evaluate(eval_envs, act, mean, std, cfg.test_seed, low, high)
            res = dict(env_name=cfg.env_name, seed=cfg.seed, K=K, N=N, episodes=cfg.n_test_episodes * len(tasks),
                       success=ev["normalized_score"], score=ev["score"], episode_length=ev["episode_length"],
                       q_start=stats["q_start"], q_gain=stats["gain_sum"] / max(stats["gain_n"], 1),
                       seconds=time.time() - t0,
                       **{k: v for k, v in ev.items() if k.startswith("success_task")})
            print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
