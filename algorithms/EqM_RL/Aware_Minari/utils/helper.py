from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 


def flatten_repeated_states(states: torch.Tensor) -> torch.Tensor:
    """
    Input:  [num_samples, batch_size, state_dim]
    Output: [batch_size * num_samples, state_dim]
    """
    return states.permute(1, 0, 2).reshape(-1, states.shape[-1])

def unflatten_repeated_tensor(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Input:  [batch_size * num_samples, dim]
    Output: [num_samples, batch_size, dim]
    """
    total, dim = tensor.shape
    assert total % num_samples == 0, f"Tensor size {total} must be divisible by {num_samples}"

    batch_size = total // num_samples
    return tensor.reshape(batch_size, num_samples, dim).permute(1, 0, 2).contiguous()

def select_lowest_ood_actions(actions: torch.Tensor, ood_scores: torch.Tensor) -> torch.Tensor:
    """
    actions:    [num_samples, batch_size, action_dim]
    ood_scores: [num_samples, batch_size, 1]
    
    Returns the actions associated with the lowest half of the OOD scores.
    """
    assert actions.ndim == 3 and ood_scores.ndim == 3
    assert actions.shape[:2] == ood_scores.shape[:2]
    assert ood_scores.shape[-1] == 1

    k = actions.shape[0] // 2


    _, indices = torch.topk(ood_scores, k=k, dim=0, largest=False, sorted=True)


    indices = indices.expand(-1, -1, actions.shape[-1])

    selected_actions = torch.gather(actions, dim=0, index=indices)
    
    return selected_actions
