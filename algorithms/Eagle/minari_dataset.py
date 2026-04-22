import minari
import numpy as np

# id = "mujoco/halfcheetah/simple-v0"


def prepare_minari_data(env_id):

    if env_id not in minari.list_local_datasets():
        minari.download_dataset(env_id)

    minari_dataset = minari.load_dataset(env_id)

    env = minari_dataset.recover_environment()

    observations = []
    next_observations = []
    actions = []
    rewards = []
    truncations = []
    terminations = []

    for eps in minari_dataset.iterate_episodes():
        observations.append(eps.observations[:-1])
        actions.append(eps.actions)
        next_observations.append(eps.observations[1:])
        rewards.append(eps.rewards)
        truncations.append(eps.truncations)
        terminations.append(eps.terminations)

    dataset = {
        "observations": np.concatenate(observations),
        "actions": np.concatenate(actions),
        "rewards": np.concatenate(rewards),
        "next_observations": np.concatenate(next_observations),
        "truncations": np.concatenate(truncations),
        "terminations": np.concatenate(terminations),
    }
    return env, dataset
