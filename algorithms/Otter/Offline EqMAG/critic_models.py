from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 
from utils.networks import MLP



class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


#################################################################################################
#################################################################################################
#################################################################################################

class DeterministicCritic(Module):
    def __init__(self,state_dim, action_dim, hidden_dim, num_hidden):
        super().__init__()

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.output_dim = 1

        self.input_dim = self.state_dim + self.action_dim

        self.model = MLP(
            input_dim  = self.input_dim, 
            output_dim = self.output_dim, 
            hidden_dim = self.hidden_dim,
            num_hidden = self.num_hidden
            )

    def forward(self,state,action):

        x = torch.cat([state,action],dim=1)

        value = self.model(x)

        return value
    
    
class GaussianCritic(Module):
    def __init__(self,state_dim, action_dim, hidden_dim, num_hidden, num_samples):
        super().__init__()

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.output_dim = hidden_dim

        self.num_samples = num_samples

        self.input_dim = self.state_dim + self.action_dim

        self.model = MLP(
            input_dim  = self.input_dim, 
            output_dim = self.output_dim, 
            hidden_dim = self.hidden_dim,
            num_hidden = self.num_hidden-1
            )


        self.mean_layer    = lin(in_features=self.hidden_dim,out_features=1)
        self.log_std_layer = lin(in_features=self.hidden_dim,out_features=1)

    def forward(self,state,action):

        x = torch.cat([state,action],dim=1)
        x = self.model(x)
        x = F.mish(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        dist = Normal(mean,std)

        values = dist.sample((self.num_samples,))
        value = torch.mean(values, dim=0)

        return value
    
