import time

import gymnasium as gym
import torch

# Import the necessary classes and functions from your training script
from Awac_Minari import Actor, compute_mean_std, wrap_env
from minari_utils import prepare_minari_data


def visualize_checkpoint():
    # 1. Configuration
    env_name = "mujoco/halfcheetah/medium-v0"
    checkpoint_path = "/Users/batin13/Desktop/RL Projects/Trained_Models/checkpoints/AWAC-mujoco/halfcheetah/medium-v0-4f779c66/checkpoint_169999.pt"
    device = "cpu"

    print("Loading dataset to compute normalization statistics...")
    # 2. Get normalization stats (Crucial: Your agent fails without this)
    _, dataset, _ = prepare_minari_data(env_name)
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    print("Initializing environment for human rendering...")
    # 3. Create the environment with rendering enabled
    # We use the underlying Gymnasium env for rendering
    env = gym.make("HalfCheetah-v4", render_mode="human")
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Loading Actor network...")
    # 4. Initialize Actor
    actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)

    # 5. Load Checkpoint
    # Your awac.state_dict() saved a dictionary with 'actor', 'critic_1', 'critic_2'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()  # Set to evaluation mode

    print("Starting simulation...")
    # 6. Evaluation Loop
    n_episodes = 10
    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            # Use your clamped act() method
            action = actor.act(state, device)

            # Step the environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1

            # Small sleep to keep the rendering at a viewable speed
            time.sleep(0.05)

        print(
            f"Episode {i + 1} finished. Total Reward: {episode_reward:.2f} | Steps: {step_count}"
        )

    env.close()


if __name__ == "__main__":
    visualize_checkpoint()
