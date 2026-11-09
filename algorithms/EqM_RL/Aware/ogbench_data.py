"""OGBench single-task data and evaluation for fit_actor.py. Run with the `ogbench` conda env
(/workspace/miniconda3/envs/ogbench: OGBench 1.2.2, gymnasium, mujoco 3); the D4RL scripts stay in hawk.

Dataset: the single-task names, e.g. antmaze-large-navigate-singletask-v0 or ...-singletask-task3-v0. The
returned dict has the keys of offline_data.load_d4rl (observations, actions, rewards, next_observations,
terminals), plus the same state normalization and the episode of every transition. OGBench's own fields differ:
  terminals  OGBench: 1 at the last step of each collected trajectory (an episode boundary, NOT the end of the
             task). Used here for the episode ids only.
  masks      0 where the task is solved (reward 0), else 1. The task terminal is 1 - masks, which is what D4RL
             calls terminals, so data["terminals"] = 1 - masks.
  rewards    mazes (navigate): -1 per step, 0 at the goal. Manipulation (cube, scene, puzzle): -(number of
             unsolved subtasks), e.g. cube-double -2 / -1 / 0. Reward 0 <=> masks 0 in both.
Behavior cloning uses only the observations and actions; the rewards are those of the dataset's own task.

Evaluation, as OGBench's reference implementations (FQL's utils/evaluation.py, 50 episodes per task):
  - task k runs in its own environment <base>-singletask-task<k>-v0, with that task's start and goal;
  - an episode succeeds if the environment's info["success"] at its last step is 1. Single-task episodes end at
    the goal (terminated) or after max_episode_steps (truncated: 1000 for the ant and point mazes, 2000-4000
    for the humanoid mazes);
  - the score OGBench reports is the success rate in %, and an environment's score is the mean over its 5 tasks;
  - actions are clipped to the action box ([-1, 1]).
Differences from OGBench's evaluation: episodes are seeded (seed + 1000 * task + i), so every eval and every run
sees the same starts (OGBench resets unseeded; the scores are the same in expectation). The seed must also go to
numpy's global RNG, which the mazes use for the start noise (seeded_reset). And all episodes of all tasks run side
by side, one batched policy call per step. A behavior-cloning policy doesn't depend on the task, so one model is
evaluated on all tasks.
"""

import os

import numpy as np

OGBENCH_DATA = "/workspace/ogbench_data"


def is_ogbench(env_name: str) -> bool:
    return "singletask" in env_name.split("-")


def task_env_name(env_name: str, task: int) -> str:
    """antmaze-large-navigate-singletask-v0 (or -singletask-task2-v0) -> antmaze-large-navigate-singletask-task<task>-v0"""
    parts = env_name.split("-")
    pos = parts.index("singletask")
    return "-".join(parts[: pos + 1] + [f"task{task}"] + parts[-1:])


def own_task(env) -> int:
    """The task a single-task environment was made for (an explicit taskN, or the dataset's default task)."""
    return int(env.unwrapped._reward_task_id)


def parse_tasks(spec: str, env) -> list:
    """'all' -> 1..num_tasks; 'own' -> the dataset's task; '1,3' -> [1, 3]."""
    if spec == "all":
        return list(range(1, env.unwrapped.num_tasks + 1))
    if spec == "own":
        return [own_task(env)]
    tasks = [int(t) for t in spec.split(",") if t.strip()]
    assert tasks and all(1 <= t <= env.unwrapped.num_tasks for t in tasks), f"bad eval_tasks {spec!r}"
    return tasks


def load_ogbench(env_name: str, dataset_dir: str = OGBENCH_DATA):
    """-> env, dataset (D4RL-style keys), state mean, state std, episode id per transition, info dict."""
    import ogbench

    env, train, _ = ogbench.make_env_and_datasets(env_name, dataset_dir=os.path.expanduser(dataset_dir),
                                                  compact_dataset=False)
    data = dict(observations=train["observations"], actions=train["actions"], rewards=train["rewards"],
                next_observations=train["next_observations"],
                terminals=(1.0 - train["masks"]).astype(np.float32))     # task terminal (D4RL meaning)
    obs = data["observations"]
    mean, std = obs.mean(0), obs.std(0) + 1e-3                            # as load_d4rl and main.py
    ends = train["terminals"].astype(bool)                                # trajectory boundaries
    ep = np.concatenate([[0], np.cumsum(ends)[:-1]])
    info = dict(benchmark="ogbench", dataset_dir=dataset_dir, transitions=len(obs), episodes=int(ends.sum()),
                state_dim=int(obs.shape[1]), action_dim=int(data["actions"].shape[1]),
                max_episode_steps=int(env.spec.max_episode_steps), num_tasks=int(env.unwrapped.num_tasks),
                dataset_task=own_task(env), task_solved_share=float(data["terminals"].mean()))
    return env, data, mean, std, ep, info


def make_eval_envs(env_name: str, tasks: list, n_per_task: int) -> list:
    """[(task, env)] with n_per_task environments for each task."""
    import ogbench

    out = []
    for k in tasks:
        name = task_env_name(env_name, k)
        out += [(k, ogbench.make_env_and_datasets(name, env_only=True)) for _ in range(n_per_task)]
    return out


def seeded_reset(env, seed: int) -> np.ndarray:
    """reset(seed=...) seeds only the environment's own RNG. OGBench's mazes draw the start-position noise (and,
    in the teleport mazes, the teleport targets) from numpy's GLOBAL RNG (locomaze/maze.py, add_noise), so
    without seeding that too, the starts would differ between evals and runs. evaluate() restores the global
    RNG afterwards."""
    np.random.seed(seed % 2 ** 32)
    return env.reset(seed=seed)[0]


def evaluate(envs: list, act, mean: np.ndarray, std: np.ndarray, seed: int, low: float, high: float) -> dict:
    """Runs one episode in every environment of make_eval_envs, all side by side. act: normalized states
    (n, state_dim) float32 -> actions (n, action_dim). Returns the log entries without a prefix (fit_actor.py
    logs the *_task<k> ones under eval_tasks/ and the others under eval/):
      normalized_score       success rate in %, mean over the tasks (OGBench's score)
      success_task<k>        success rate in % of task k
      score, score_std, score_min, return_task<k>   undiscounted return (mazes: -1 per step until the goal;
                                                    manipulation: -(unsolved subtasks) per step)
      episode_length, length_task<k>                steps until the goal or the time limit
    """
    n = len(envs)
    task = np.array([k for k, _ in envs])
    ret, length = np.zeros(n), np.zeros(n, int)
    done, success = np.zeros(n, bool), np.zeros(n)
    global_rng = np.random.get_state()      # restored below: the eval must not change the caller's numpy RNG
    try:
        obs, j = [], {}
        for i, (k, e) in enumerate(envs):
            j[k] = j.get(k, -1) + 1
            obs.append(seeded_reset(e, seed + 1000 * k + j[k]))
        obs = np.stack(obs).astype(np.float64)
        while not done.all():
            active = np.flatnonzero(~done)
            a = np.clip(act(((obs[active] - mean) / std).astype(np.float32)), low, high)
            for m, i in enumerate(active):
                ob, r, terminated, truncated, info = envs[i][1].step(a[m])
                obs[i] = ob
                ret[i] += r
                length[i] += 1
                if terminated or truncated:
                    done[i] = True
                    success[i] = float(info["success"])                   # its value at the last step
    finally:
        np.random.set_state(global_rng)
    out = {}
    for k in sorted(set(task)):
        sel = task == k
        out[f"success_task{k}"] = 100.0 * success[sel].mean()
        out[f"return_task{k}"] = ret[sel].mean()
        out[f"length_task{k}"] = length[sel].mean()
    out["normalized_score"] = float(np.mean([out[f"success_task{k}"] for k in sorted(set(task))]))
    out.update(score=ret.mean(), score_std=ret.std(), score_min=ret.min(), episode_length=length.mean())
    return out
