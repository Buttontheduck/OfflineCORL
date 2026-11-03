"""Shared helpers: load a trained Aware_Minari run (config, dataset, EqM field) and sample
actions with exactly the sampler that Actor._implicit uses (opt_type 'nag')."""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

AWARE_MINARI = Path(__file__).resolve().parents[1] / "Aware_Minari"
sys.path.insert(0, str(AWARE_MINARI))
from minari_utils import prepare_minari_data  # noqa: E402
from utils.networks import ConditionalMLP  # noqa: E402


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not ckpts:
        sys.exit(f"No checkpoint_*.pt in {run_dir}")
    return ckpts[-1]


class Policy:
    """A trained run: its config, Minari dataset and env, state normalization and EqM field."""

    def __init__(self, run_dir: Path, checkpoint: str = None, device: str = "cuda"):
        self.run_dir, self.device = Path(run_dir), device
        self.cfg = yaml.safe_load(open(self.run_dir / "config.yaml"))
        if self.cfg["actor"]["opt_type"] != "nag":
            sys.exit("Only the 'nag' sampler is implemented here")
        self.eta, self.mu = self.cfg["actor"]["step_size"], self.cfg["actor"]["moment"]

        self.env, self.data, self.minari_dataset = prepare_minari_data(self.cfg["env_name"])
        obs = self.data["observations"]
        self.mean, self.std = obs.mean(0), obs.std(0) + 1e-3  # as in main.py
        self.state_dim, self.action_dim = obs.shape[1], self.data["actions"].shape[1]
        self.low = float(self.env.action_space.low[0])
        self.high = float(self.env.action_space.high[0])

        self.checkpoint = self.run_dir / checkpoint if checkpoint else latest_checkpoint(self.run_dir)
        self.model = ConditionalMLP(self.state_dim + self.action_dim, self.action_dim,
                                    self.cfg["hidden_dim"], self.cfg["num_hidden_layers"]).to(device)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=device)["model"])
        self.model.eval()

    def info(self) -> dict:
        return dict(run=self.run_dir.name, checkpoint=self.checkpoint.name,
                    env=self.cfg["env_name"].split("/")[-1], lam=self.cfg["agent"]["scale_gradient"],
                    seed=self.cfg["seed"], eta=self.eta, K_train=self.cfg["actor"]["num_step"])

    def states(self, obs: np.ndarray) -> torch.Tensor:
        """Raw observations -> normalized state tensor."""
        return torch.tensor((obs - self.mean) / self.std, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def sample(self, s: torch.Tensor, K: int, snaps=()):
        """K steps of the NAG sampler from N(0, I), unclipped (exactly Actor._implicit).
        Returns the final actions and {k: actions after k steps} for k in snaps."""
        x = torch.randn(s.shape[0], self.action_dim, device=self.device)
        m = torch.zeros_like(x)
        out = {}
        for k in range(1, K + 1):
            m = self.model(x - self.eta * m * self.mu, s)
            x = x - self.eta * m
            if k in snaps:
                out[k] = x.clone()
        return x, out
