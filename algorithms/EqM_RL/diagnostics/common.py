"""Shared helpers: load a trained Aware run (config, dataset, EqM field, optionally the critics)
and sample actions with exactly the sampler that Actor._implicit uses (opt_type 'nag').

Works for both codes: Aware_Minari runs (Minari data, `eagle` env) and Aware runs (D4RL data,
`hawk` env). The run's env_name decides which: Minari ids contain a '/'
(mujoco/halfcheetah/simple-v0), D4RL ids don't (halfcheetah-medium-expert-v2)."""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EQM_RL = Path(__file__).resolve().parents[1]


def is_minari(env_name: str) -> bool:
    return "/" in env_name


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not ckpts:
        sys.exit(f"No checkpoint_*.pt in {run_dir}")
    return ckpts[-1]


class Policy:
    """A trained run: its config, dataset and env, state normalization, EqM field and
    (with load_critic=True) its two critics."""

    def __init__(self, run_dir: Path, checkpoint: str = None, device: str = "cuda",
                 load_critic: bool = False):
        self.run_dir, self.device = Path(run_dir), device
        self.cfg = yaml.safe_load(open(self.run_dir / "config.yaml"))
        if self.cfg["actor"]["opt_type"] != "nag":
            sys.exit("Only the 'nag' sampler is implemented here")
        self.eta, self.mu = self.cfg["actor"]["step_size"], self.cfg["actor"]["moment"]

        # The code folder of the run's algorithm; both have utils/networks.py and critic_models.py
        self.minari = is_minari(self.cfg["env_name"])
        sys.path.insert(0, str(EQM_RL / ("Aware_Minari" if self.minari else "Aware")))
        from utils.networks import ConditionalMLP

        if self.minari:
            from minari_utils import prepare_minari_data
            self.env, self.data, self.minari_dataset = prepare_minari_data(self.cfg["env_name"])
        else:
            import d4rl  # noqa: F401  (registers the D4RL envs with gym)
            import gym
            self._gym = gym
            self.env = gym.make(self.cfg["env_name"])
            self.data = d4rl.qlearning_dataset(self.env)
            self.minari_dataset = None
        obs = self.data["observations"]
        self.mean, self.std = obs.mean(0), obs.std(0) + 1e-3  # as in main.py
        self.state_dim, self.action_dim = obs.shape[1], self.data["actions"].shape[1]
        self.low = float(self.env.action_space.low[0])
        self.high = float(self.env.action_space.high[0])

        self.checkpoint = self.run_dir / checkpoint if checkpoint else latest_checkpoint(self.run_dir)
        state = torch.load(self.checkpoint, map_location=device)
        self.model = ConditionalMLP(self.state_dim + self.action_dim, self.action_dim,
                                    self.cfg["hidden_dim"], self.cfg["num_hidden_layers"]).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.critics = []
        if load_critic:
            from critic_models import DeterministicCritic
            for key in ("critic_1", "critic_2"):
                c = DeterministicCritic(self.state_dim, self.action_dim, self.cfg["hidden_dim"],
                                        self.cfg["num_hidden_layers"]).to(device)
                c.load_state_dict(state[key])
                c.eval()
                self.critics.append(c)

    def info(self) -> dict:
        return dict(run=self.run_dir.name, checkpoint=self.checkpoint.name,
                    env=self.cfg["env_name"].split("/")[-1], lam=self.cfg["agent"]["scale_gradient"],
                    seed=self.cfg["seed"], eta=self.eta, K_train=self.cfg["actor"]["num_step"],
                    temperature=self.cfg.get("temperature"),
                    weighting=self.cfg["agent"].get("weighting", "awac"))

    # ------------------------------------------------------------------ envs (both gym APIs)
    def make_env(self):
        return self.minari_dataset.recover_environment() if self.minari else self._gym.make(self.cfg["env_name"])

    def reset(self, env, seed: int) -> np.ndarray:
        if self.minari:
            return env.reset(seed=seed)[0]
        env.seed(seed)  # old gym API (D4RL): seed before reset
        return env.reset()

    def step(self, env, action: np.ndarray):
        """-> (obs, reward, done)"""
        if self.minari:
            o, r, term, trunc, _ = env.step(action)
            return o, r, term or trunc
        o, r, done, _ = env.step(action)
        return o, r, done

    # ------------------------------------------------------------------ networks
    def states(self, obs: np.ndarray) -> torch.Tensor:
        """Raw observations -> normalized state tensor."""
        return torch.tensor((obs - self.mean) / self.std, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """min(Q1, Q2)(s, a), as in the advantage weights; shape (N,)."""
        return torch.min(self.critics[0](s, a), self.critics[1](s, a)).squeeze(-1)

    @torch.no_grad()
    def sample(self, s: torch.Tensor, K: int, snaps=(), eta=None, mu=None, x0=None):
        """K steps of the NAG sampler from N(0, I), unclipped (exactly Actor._implicit).
        eta and mu default to the run's config; either can be a float or a (N, 1) tensor.
        Returns the final actions and {k: actions after k steps} for k in snaps."""
        eta = self.eta if eta is None else eta
        mu = self.mu if mu is None else mu
        x = torch.randn(s.shape[0], self.action_dim, device=self.device) if x0 is None else x0.clone()
        m = torch.zeros_like(x)
        out = {}
        for k in range(1, K + 1):
            m = self.model(x - eta * m * mu, s)
            x = x - eta * m
            if k in snaps:
                out[k] = x.clone()
        return x, out

    @torch.no_grad()
    def sample_rows(self, s: torch.Tensor, K: torch.Tensor, eta: torch.Tensor, mu: torch.Tensor):
        """Like sample, but every row has its own K, eta and mu (tensors of shape (N,)), so many
        sampler settings run as one batch. Returns each row's action after its own K steps
        (unclipped)."""
        x = torch.randn(s.shape[0], self.action_dim, device=self.device)
        out = x.clone()
        m = torch.zeros_like(x)
        eta, mu = eta[:, None], mu[:, None]
        for k in range(1, int(K.max()) + 1):
            m = self.model(x - eta * m * mu, s)
            x = x - eta * m
            hit = K == k
            if hit.any():
                out[hit] = x[hit]
        return out
