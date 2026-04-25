from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 
from utils.networks import MLP



class EagleActor(nn.Module):
    def __init__(self, model, ebm, sampler, step_size, num_step, moment):
        super().__init__()
        
        self.model = model
        self.sampler = sampler
        self.step_size = step_size
        self.num_step = num_step
        self.moment = moment
        self.ebm = ebm


    def _compute_gradient(self, Xt, state):
        # We assume Xt already has requires_grad_(True) when passed in
        output = self.model(Xt,state)
            
        if self.ebm == 'dot':
            E = torch.sum(output * Xt, dim=1)     
        elif self.ebm == 'scalar':
            E = output.squeeze(-1)       
        elif self.ebm == 'l2':
            E = -0.5 * torch.sum(output**2, dim=1)        
        else:
            raise ValueError("\n During Sampling - Type of EBM set incorrectly; chose one from: l2 , scalar , dot \n ") 

        pred_grad = torch.autograd.grad(
            outputs=[E.sum()],
            inputs=[Xt],
            create_graph=False
        )[0]
        
        return pred_grad 
        

    def sample_implicit(self, x , state):

        is_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            if self.sampler == "gd":
                for _ in range(self.num_step):
                    grad = self.model(x,state)
                    x = x - self.step_size * grad

            elif self.sampler == "nag":
                m = torch.zeros_like(x)
                for _ in range(self.num_step):
                    x_lookahead = x - self.step_size * m * self.moment
                    grad = self.model(x_lookahead,state)
                    m = grad
                    x = x - self.step_size * m
            else:
                raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")
        
        if is_training:
            self.model.train()
        return x
    

    def sample_implicit_langevin(self, x , state , initial_temperature=1.0, noise_decay=0.99):

            is_training = self.model.training
            self.model.eval()

            with torch.no_grad():
                if self.sampler == "gd":
                    for i in range(self.num_step):
                        # 1. Get the deterministic gradient from the Implicit field
                        grad = self.model(x,state)

                        # 2. Anneal the temperature based on the current step
                        current_temp = initial_temperature * (noise_decay ** i)

                        # 3. Calculate Langevin noise scale: sqrt(2 * step_size * Temperature)
                        noise_scale = np.sqrt(2 * self.step_size * current_temp)
                        noise = torch.randn_like(x)

                        # 4. Update with gradient descent + thermal noise
                        x = x - self.step_size * grad + noise_scale * noise

                elif self.sampler == "nag":
                    m = torch.zeros_like(x)
                    for i in range(self.num_step):
                        # 1. Nesterov momentum lookahead
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)
                        m = grad

                        # 2. Anneal the temperature
                        current_temp = initial_temperature * (noise_decay ** i)

                        # 3. Calculate Langevin noise
                        noise_scale = np.sqrt(2 * self.step_size * current_temp)
                        noise = torch.randn_like(x)

                        # 4. Update with momentum + thermal noise
                        x = x - self.step_size * m + noise_scale * noise
                else:
                    raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")
            
            if is_training:
                self.model.train()
            return x


    def sample_implicit_ODD(self, x , state):
            
            is_training = self.model.training
            self.model.eval()
    
            with torch.no_grad():
                if self.sampler == "gd":
                    for _ in range(self.num_step):
                        grad = self.model(x,state)
                        x = x - self.step_size * grad
    
                elif self.sampler == "nag":
                    m = torch.zeros_like(x)
                    for _ in range(self.num_step):
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)
                        m = grad
                        x = x - self.step_size * m
                else:
                    raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")
              
                # --- GeCO OOD Detection Metric ---
                # Do one final forward pass at the settled location
                final_grad = self.model(x,state)
                
                # Calculate the L2 Norm of the final gradient for every particle
                # Shape will be [batch_size]
                ood_scores = torch.norm(final_grad, p=2, dim=1)
            

            if is_training:
                self.model.train()
            return x, ood_scores
    
    def sample_implicit_stop(self, x , state, tau_opt = 1.5):
            """
            GeCO Adaptive Early Stopping: Particles individually stop updating 
            once their gradient norm falls below tau_opt.
            """
            is_training = self.model.training
            self.model.eval()

            with torch.no_grad():
                if self.sampler == "gd":
                    # Create a mask tracking which particles are still moving (True = Active)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        grad = self.model(x,state)

                        # Calculate the L2 norm of the gradients
                        grad_norms = torch.norm(grad, p=2, dim=1)

                        # Update active mask: keep True ONLY if norm > tau_opt AND it was previously active
                        active_mask = active_mask & (grad_norms > tau_opt)

                        # If all particles have stopped, exit the loop completely
                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        # Apply update only to active particles (multiply grad by active_mask)
                        # active_mask.unsqueeze(1) broadcasts the 1D mask to the 2D coordinates [batch, 2]
                        grad_update = grad * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * grad_update

                elif self.sampler == "nag":
                    m = torch.zeros_like(x)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        # Lookahead
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)

                        grad_norms = torch.norm(grad, p=2, dim=1)
                        active_mask = active_mask & (grad_norms > tau_opt)

                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        m = grad
                        # Apply update only to active particles
                        m_update = m * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * m_update
                else:
                    raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")


            if is_training:
                self.model.train()
            return x
    

    def sample_implicit_ODD_stop(self, x , state, tau_opt=0.4):
            """
            Combined GeCO Sampler: 
            1. Adaptive Early Stopping (particles park when grad_norm < tau_opt)
            2. OOD Detection (returns the final gradient norms for anomaly filtering)
            """

            is_training = self.model.trainig
            self.model.eval()

            with torch.no_grad():
                if self.sampler == "gd":
                    # Create a mask tracking which particles are still moving (True = Active)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        grad = self.model(x,state)

                        # Calculate the L2 norm of the gradients
                        grad_norms = torch.norm(grad, p=2, dim=1)

                        # Update active mask: keep True ONLY if norm > tau_opt AND it was previously active
                        active_mask = active_mask & (grad_norms > tau_opt)

                        # If all particles have stopped, exit the loop completely
                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        # Apply update only to active particles (multiply grad by active_mask)
                        grad_update = grad * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * grad_update

                elif self.sampler == "nag":
                    m = torch.zeros_like(x)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        # Lookahead
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)

                        grad_norms = torch.norm(grad, p=2, dim=1)
                        active_mask = active_mask & (grad_norms > tau_opt)

                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        m = grad
                        # Apply update only to active particles
                        m_update = m * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * m_update
                else:
                    raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")

                # --- GeCO OOD Detection Metric ---
                # Do one final forward pass at the settled location to get final OOD scores
                final_grad = self.model(x,state)
                ood_scores = torch.norm(final_grad, p=2, dim=1)

            if is_training:
                self.model.train()
            return x, ood_scores


    def sample_explicit(self, x , state):

        is_training = self.model.training
        self.model.eval()
        with torch.enable_grad():     
            if self.sampler == "gd":        
                # Start with a clean, detached tensor
                x = x.detach()

                for _ in range(self.num_step):
                    # 1. Turn on tracking for this specific step
                    x.requires_grad_(True)

                    # 2. Get the gradient
                    grad = self._compute_gradient(x,state)

                    # 3. Update the state and immediately DETACH to prevent a memory leak
                    x = (x - self.step_size * grad).detach()

            elif self.sampler == "nag":     
                x = x.detach()
                m = torch.zeros_like(x)

                for _ in range(self.num_step):
                    # 1. Calculate lookahead and detach it so it's a clean starting point
                    x_lookahead = (x - self.step_size * m * self.moment).detach()

                    # 2. Turn on tracking for the lookahead position
                    x_lookahead.requires_grad_(True)

                    # 3. Get the gradient
                    grad = self._compute_gradient(x_lookahead,state)

                    # 4. Save momentum (detached to avoid tracking history)
                    m = grad.detach()

                    # 5. Update the main state and detach
                    x = (x - self.step_size * m).detach()
            else:
                raise ValueError(f"\n Sampler must be 'gd' or 'nag', got '{self.sampler}' \n ")
        
        if is_training:
            self.model.train()
        return x
    


    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        # Use .detach() to break the tensor off the gradient graph
        action = action_t[0].detach().cpu().numpy()

        return action
