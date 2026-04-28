import os
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional
from agent import AdvantageWeightedActorCritic
from buffer import ReplayBuffer
from minari_utils import get_ref_scores, minari_normalized_score, prepare_minari_data
from models import Actor, Critic
from tqdm import trange

import wandb


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "AWAC-D4RL"
    # wandb run name
    name: str = "AWAC"
    # training dataset and evaluation environment
    env_name: str = "mujoco/halfcheetah/simple-v0"  # "mujoco/hopper/medium-v0" #"mujoco/halfcheetah/medium-v0"  # "halfcheetah-medium-expert-v2"
    # actor and critic hidden dim
    hidden_dim: int = 256
    # actor and critic learning rate
    learning_rate: float = 3e-4
    # discount factor
    gamma: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 5e-3
    # awac actor loss temperature, controlling balance
    # between behaviour cloning and Q-value maximization
    awac_lambda: float = 1.0
    # total number of gradient updated during training
    num_train_ops: int = 1_000_000
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # evaluation frequency, will evaluate every eval_frequency
    # training steps
    eval_frequency: int = 1000
    # number of episodes to run during evaluation
    n_test_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = (
        "/Users/batin13/Desktop/RL Projects/Trained_Models/checkpoints/"
    )
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    seed: int = 42
    # evaluation random seed
    test_seed: int = 69
    # training device
    device: str = "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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
            action = actor.act(state, device)
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
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


@pyrallis.wrap()
def train(config: TrainConfig):

    env, dataset, minari_dataset = prepare_minari_data(config.env_name)
    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ref_scores = get_ref_scores(minari_dataset)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

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
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_minari_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    for t in trange(config.num_train_ops, ncols=80):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                env, actor, config.device, config.n_test_episodes, config.test_seed
            )

            wandb.log({"eval_score": eval_scores.mean()}, step=t)
            normalized_eval_scores = minari_normalized_score(
                eval_scores.mean(), ref_scores
            )
            wandb.log({"minari_normalized_score": normalized_eval_scores}, step=t)

            if config.checkpoints_path is not None:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
