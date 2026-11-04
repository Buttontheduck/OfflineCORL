"""The Gaussian actor of fit_actor.py, in its own module so that diagnostics/common.py can rebuild
saved fits without importing the training script."""

import torch
import torch.nn as nn


class GaussianActor(nn.Module):
    """CORL awac.py's Actor with a configurable depth: a mean MLP (ReLU) and a state-independent
    log std. num_hidden_layers hidden layers of hidden_dim units (awac.py's own actor has 3)."""

    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layers, min_log_std=-20.0, max_log_std=2.0):
        super().__init__()
        layers, d = [], state_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self._mlp = nn.Sequential(*layers, nn.Linear(d, action_dim))
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std, self._max_log_std = min_log_std, max_log_std

    def log_prob(self, state, action):
        std = self._log_std.clamp(self._min_log_std, self._max_log_std).exp()
        return torch.distributions.Normal(self._mlp(state), std).log_prob(action).sum(-1)
