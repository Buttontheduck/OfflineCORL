import os
import random
import uuid
from typing import Optional, Tuple, Union

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional
from buffer import ReplayBuffer
from actor_models import Actor
from critic_models import DeterministicCritic, ValueNetwork
from agents import Otter
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from utils.networks import ConditionalMLP
from logger import global_logger as gl
import wandb


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        # Ensure the state matches the 32-bit float dtype standard in RL
        return ((state - state_mean) / state_std).astype(np.float32)

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config.get("project"),
        group=config.get("group"),
        name=config.get("run_name"),
        id=config.get("run_id"),
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    if getattr(cfg, "run_id", None) is None:
        cfg.run_id = str(uuid.uuid4())
    else:
        cfg.run_id = str(cfg.run_id)

    safe_env_name = cfg.env_name.replace("/", "_")
    cfg.run_name = f"{cfg.name}-{safe_env_name}-{cfg.run_id[:8]}"

    if getattr(cfg, "checkpoints_path", None) is not None:
        cfg.actual_checkpoints_path = os.path.join(cfg.checkpoints_path, cfg.run_name)
        os.makedirs(cfg.actual_checkpoints_path, exist_ok=True)
    else:
        cfg.actual_checkpoints_path = None

    OmegaConf.set_struct(cfg, True)

    env = gym.make(cfg.env_name)
    dataset = d4rl.qlearning_dataset(env)
    set_seed(cfg.seed, env, deterministic_torch=cfg.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if getattr(cfg, "normalize_reward", False):
        modify_reward(dataset, cfg.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # One env per eval episode, so the episodes run side by side (see eval_actor)
    eval_envs = [env] + [
        wrap_env(gym.make(cfg.env_name), state_mean=state_mean, state_std=state_std)
        for _ in range(cfg.n_test_episodes - 1)
    ]

    # Fixed dataset states for the sampler/* logs at each eval
    diag_idx = np.random.default_rng(0).choice(
        len(dataset["observations"]), size=min(cfg.diag_states, len(dataset["observations"])), replace=False
    )
    diag_states = torch.tensor(dataset["observations"][diag_idx], dtype=torch.float32, device=cfg.device)
    diag_actions = torch.tensor(dataset["actions"][diag_idx], dtype=torch.float32, device=cfg.device)
    diag_generator = torch.Generator(device=cfg.device).manual_seed(0)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        cfg.buffer_size,
        cfg.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    model_kwargs = {
        "input_dim" :        state_dim+action_dim,
        "output_dim":        action_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }
    
    critic_kwargs = {
        "state_dim" :        state_dim,
        "action_dim":        action_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }

    model = ConditionalMLP(**model_kwargs)
    model.to(cfg.device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    critic_1 = DeterministicCritic(**critic_kwargs)
    critic_2 = DeterministicCritic(**critic_kwargs)
    critic_1.to(cfg.device)
    critic_2.to(cfg.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=cfg.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=cfg.learning_rate)
    
    actor = Actor(
        model         = model,
        critic_1      = critic_1,
        critic_2      = critic_2,        
        action_dim    = action_dim,
        min_action    = env.action_space.low[0],
        max_action    = env.action_space.high[0],
        ebm           = cfg.actor.ebm,
        opt_type      = cfg.actor.opt_type,
        step_size     = cfg.actor.step_size, 
        num_step      = cfg.actor.num_step,
        moment        = cfg.actor.moment,
        sampler_type  = cfg.actor.sampler_type,
        ood_threshold = cfg.actor.ood_threshold,
        early_stop    = cfg.actor.early_stop
        )
    
    # IQL critic option (agent.critic: iql): a V(s) network, same size as the critics
    value, value_optimizer = None, None
    if cfg.agent.get("critic", "awac") == "iql":
        value = ValueNetwork(state_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(cfg.device)
        value_optimizer = torch.optim.Adam(value.parameters(), lr=cfg.learning_rate)

    agent = Otter(
        value               = value,
        value_optimizer     = value_optimizer,
        model               = model,
        model_optimizer     = model_optimizer,
        actor               = actor,
        critic_1            = critic_1,
        critic_1_optimizer  = critic_1_optimizer,
        critic_2            = critic_2,
        critic_2_optimizer  = critic_2_optimizer,
        **cfg.agent
        )

    @torch.no_grad()
    def eval_actor(envs: list, seed: int) -> np.ndarray:
        """One episode per env (seeds seed, seed+1, ...), run side by side: at every step the
        sampler gets the states of all unfinished episodes as one batch, so 10 episodes cost
        about as much as one."""
        actor.eval()
        states = []
        for i, e in enumerate(envs):
            # Old gym API (D4RL): seed before reset; step returns 4 values
            e.seed(seed + i)
            states.append(e.reset())
        episode_rewards = np.zeros(len(envs))
        done = np.zeros(len(envs), dtype=bool)
        while not done.all():
            active = np.flatnonzero(~done)
            actions = actor.sample(np.stack([states[i] for i in active]))
            for action, i in zip(actions, active):
                states[i], reward, done[i], _ = envs[i].step(action)
                episode_rewards[i] += reward

        actor.train()
        return episode_rewards
    

    wandb_init(OmegaConf.to_container(cfg, resolve=True))

    if getattr(cfg, "actual_checkpoints_path", None) is not None:
        print(f"Checkpoints path: {cfg.actual_checkpoints_path}")
        with open(os.path.join(cfg.actual_checkpoints_path, "config.yaml"), "w") as f:
            # Save the fully resolved Hydra config
            OmegaConf.save(config=cfg, f=f)
    try:
        for t in trange(cfg.num_train_ops, ncols=80):
            batch = replay_buffer.sample(cfg.batch_size)
            batch = [b.to(cfg.device) for b in batch]
            update_result = agent.update(batch)
            for key, val in update_result.items():
                gl.log(f"train/{key}", val)
            gl.dump(step=t)
            if (t) % cfg.eval_frequency == 0:
                eval_scores = eval_actor(eval_envs, cfg.test_seed)

                gl.log("eval/score", eval_scores.mean())
                gl.log("eval/score_std", eval_scores.std())
                gl.log("eval/score_min", eval_scores.min())
                normalized_eval_scores = env.get_normalized_score(eval_scores.mean()) * 100.0
                gl.log("eval/normalized_score", normalized_eval_scores)
                if cfg.diag_states > 0:
                    sampler_stats = actor.diagnostics(
                        diag_states, diag_actions, n_samples=cfg.diag_samples, generator=diag_generator
                    )
                    for key, val in sampler_stats.items():
                        gl.log(f"sampler/{key}", val)

                gl.dump(step=t)

                if getattr(cfg, "actual_checkpoints_path", None) is not None:
                    torch.save(
                        agent.state_dict(),
                        os.path.join(cfg.actual_checkpoints_path, f"checkpoint_{t}.pt"),
                    )
    except BaseException:
        # Ctrl-C or an error: leave the W&B run to wandb's exit hook, which marks it 'killed'
        # (KeyboardInterrupt) or 'failed'; wandb.finish() here would mark it 'finished'.
        raise
    else:
        wandb.finish()


if __name__ == "__main__":
    train()
