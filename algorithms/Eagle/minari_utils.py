import warnings

import minari
import numpy as np
from minari import MinariDataset

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
    return env, dataset, minari_dataset


MINARI_DICT_REF_SCORES = {
    "hopper": (-20.272305, 3234.3),
    "halfcheetah": (-280.178953, 12135.0),
    "walker2d": (1.629008, 4592.3),
    "pen": (137.92955108500217, 8820.50845967583),
    "door": (-45.80706024169922, 2940.578369140625),
    "hammer": (-267.074462890625, 12635.712890625),
    "relocate": (9.189092636108398, 4287.70458984375),
}


def get_ref_scores(dataset: MinariDataset) -> tuple[float, float]:
    # Pure function that returns the reference scores safely.
    metadata = dataset.storage.metadata

    # Safely extract both values using .get()
    meta_min = metadata.get("ref_min_score")
    meta_max = metadata.get("ref_max_score")

    metadata_scores = None
    if meta_min is not None and meta_max is not None:
        metadata_scores = (meta_min, meta_max)

    # Check against our hardcoded dictionary
    for key, saved_scores in MINARI_DICT_REF_SCORES.items():
        if key in dataset.id:
            if (
                metadata_scores is not None
                and np.linalg.norm(np.array(metadata_scores) - np.array(saved_scores))
                > 0.1
            ):
                warnings.warn(
                    "=== WARNING ===\nMinari reference scores found in dataset's metadata "
                    "do not match manually specified ones. Falling back to manual."
                )
            return saved_scores

    # Fallback to metadata if environment is not in our dictionary
    if metadata_scores is not None:
        return metadata_scores

    # If we reach here, we have a total failure
    raise Exception(
        f"Reference scores for the Minari dataset {dataset.id} were neither "
        f"found in the dataset metadata nor in our dict of reference scores."
    )


def minari_normalized_score(acc_reward: float, ref_scores: tuple[float, float]) -> float:
    # Takes the explicitly passed scores, avoiding global state.
    min_score, max_score = ref_scores
    return 100.0 * (acc_reward - min_score) / (max_score - min_score)
