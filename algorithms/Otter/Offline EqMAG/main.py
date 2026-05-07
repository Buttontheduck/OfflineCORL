import os
import random
import uuid
from typing import Optional, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional
from buffer import ReplayBuffer
from minari_utils import get_ref_scores, minari_normalized_score, prepare_minari_data
from actor_models import Actor
from critic_models import Critic
from agents import Otter
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from utils.networks import ConditionalMLP

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

    # Gymnasium requires explicitly defining the new observation space.
    # Since we normalized, the bounds are technically [-inf, inf]
    new_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float32
    )

    env = gym.wrappers.TransformObservation(env, normalize_state, new_obs_space)
    return env


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.sample(state,device)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, term, trun in zip(
        dataset["rewards"], dataset["terminations"], dataset["truncations"]
    ):
        d = term or trun
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

    env, dataset, minari_dataset = prepare_minari_data(cfg.env_name)
    set_seed(cfg.seed, env, deterministic_torch=cfg.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ref_scores = get_ref_scores(minari_dataset)

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
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        cfg.buffer_size,
        cfg.device,
    )
    replay_buffer.load_minari_dataset(dataset)

    model_kwargs = {
        "input_dim" :        state_dim+action_dim,
        "output_dim":        action_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }
    
    critic_kwargs = {
        "input_dim" :        state_dim+action_dim,
        "output_dim":        1,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }

    model = ConditionalMLP(**model_kwargs)
    model.to(cfg.device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    critic_1 = ConditionalMLP(**critic_kwargs)
    critic_2 = ConditionalMLP(**critic_kwargs)
    critic_1.to(cfg.device)
    critic_2.to(cfg.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=cfg.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=cfg.learning_rate)
    
    actor = Actor(
        model      = model,
        action_dim = action_dim,
        ebm        = cfg.actor.ebm,
        opt_type   = cfg.actor.opt_type,
        step_size  = cfg.actor.step_size, 
        num_step   = cfg.actor.num_step,
        moment     = cfg.actor.moment
        )
    
    agent = Otter(
        model=model,
        model_optimizer     = model_optimizer,
        critic_1            = critic_1,
        critic_1_optimizer  = critic_1_optimizer,
        critic_2            = critic_2,
        critic_2_optimizer  = critic_2_optimizer,
        **cfg.agent
        )
    

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
            wandb.log(update_result, step=t)
            if (t + 1) % cfg.eval_frequency == 0:
                eval_scores = eval_actor(
                    env, actor, cfg.device, cfg.n_test_episodes, cfg.test_seed
                )

                wandb.log({"eval_score": eval_scores.mean()}, step=t)
                normalized_eval_scores = minari_normalized_score(
                    eval_scores.mean(), ref_scores
                )
                wandb.log({"minari_normalized_score": normalized_eval_scores}, step=t)

                if getattr(cfg, "actual_checkpoints_path", None) is not None:
                    torch.save(
                        agent.state_dict(),
                        os.path.join(cfg.actual_checkpoints_path, f"checkpoint_{t}.pt"),
                    )
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
