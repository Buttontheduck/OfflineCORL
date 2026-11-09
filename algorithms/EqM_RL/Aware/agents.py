from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import global_logger as gl

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def expand_t_like_x(t, x):
    
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t
  

class Otter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        model_optimizer: torch.optim.Optimizer,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        temperature: float = 1.0,
        exp_adv_max: float = 100.0,
        fino: bool = False,
        flag_type: str = 'trun',
        flag_threshold: float = 0.8,
        scale_gradient: float = 4.0,
        num_actions: int = 10,
        fino_knob: float = 0.1,
        weighting: str = "awac",
        critic: str = "awac",
        iql_tau: float = 0.7,
        value: nn.Module = None,
        value_optimizer: torch.optim.Optimizer = None,
        mc_samples: int = 0,
        **kwargs

    ):
        super().__init__()
        self._model = model
        self._model_optimizer = model_optimizer
        
        self._actor = actor

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._temperature = temperature
        self._exp_adv_max = exp_adv_max
        
        self.fino = fino
        self.flag_type = flag_type
        self.flag_threshold = flag_threshold
        self.scale_gradient = scale_gradient
        self.fino_knob = fino_knob

        self.num_actions = num_actions

        # 'awac': weights exp(A/T) clipped at exp_adv_max. 'bc': every weight is 1 (behavior
        # cloning ablation); the advantage statistics are still computed and logged.
        if weighting not in ("awac", "bc"):
            raise ValueError(f"weighting must be 'awac' or 'bc', got {weighting!r}")
        self.weighting = weighting

        # 'awac': Q is trained on the actor's sampled next action, and V(s) = min Q at the
        # actor's own sample (the original Aware). 'iql': IQL's in-sample critic (Kostrikov et
        # al. 2021, as CORL's iql.py): V(s) is an upper expectile (iql_tau) of min target Q over
        # dataset actions, Q's target is r + gamma * V(s'), and A = min target Q(s, a) - V(s).
        # The critic then never sees an EqM sample.
        if critic not in ("awac", "iql"):
            raise ValueError(f"critic must be 'awac' or 'iql', got {critic!r}")
        if critic == "iql" and (value is None or value_optimizer is None):
            raise ValueError("critic 'iql' needs a value network and its optimizer")
        self.critic = critic
        self._iql_tau = iql_tau
        self._value = value
        self._value_optimizer = value_optimizer

        # mc_samples > 0: the actor's weights come from _offline_weights_MC, where V(s) is the mean of
        # min(Q1, Q2) over mc_samples actor samples per state. 0: _offline_weights with num_actions
        # samples (as before). Only for the 'awac' critic: the 'iql' critic learns V(s) as a network.
        if mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {mc_samples}")
        if mc_samples > 0 and critic != "awac":
            raise ValueError(f"mc_samples needs critic 'awac' (the 'iql' critic has its own V network), got {critic!r}")
        self.mc_samples = mc_samples




    def _expand_dims(self,low_dim_vector,reference):
        vector_matched_dim = expand_t_like_x(low_dim_vector,reference)
        return vector_matched_dim
    
    def _c_linear(self,rho):
        return 1-rho
    
    def _c_trun(self,interpolate,rho):
        

        if not (0 < interpolate < 1):
              raise ValueError(f"Interpolate must be in (0,1), got {interpolate}")

        start = 1.0
        ct = torch.minimum(start-(start-1)/(interpolate)*rho, 1/(1-interpolate)-1/(1-interpolate)*rho)
        return ct

    def _sample_gamma_and_noise(self,true_action):
        
        batch_size = true_action.shape[0]
        device = true_action.device
        dtype = true_action.dtype

        rho = torch.rand(batch_size, device=device, dtype=dtype)  
        noise = torch.randn_like(true_action)
        return rho, noise
    
    def _mix_data_and_compute_target(self,rho,noise,true_action):
    
        rho = self._expand_dims(rho,true_action)
        noisy_action = rho * (true_action) + (1 - rho) * noise
        target_gradient = noise-true_action
        return noisy_action, target_gradient
    
    def _fino_mix_data_and_compute_target(self, rho, noise, true_action, eta=0.1):
           
            """
            FINO Noise Injection: Expands the learned action space by injecting 
            a scheduled Gaussian noise and modifying the target vector.
            """
            rho = self._expand_dims(rho, true_action)

            # 1. Standard interpolation
            noisy_action = rho * true_action + (1 - rho) * noise

            # 2. FINO Variance Schedule: alpha^2 = (eta^2 - 2*eta)*rho^2 + 2*eta*rho
            alpha_sq = (eta**2 - 2*eta) * (rho**2) + 2 * eta * rho

            # Clamp at 0 for numerical safety before taking the square root
            alpha = torch.sqrt(torch.clamp(alpha_sq, min=0.0)) 

            # 3. Inject the FINO noise into the state
            fino_noise = torch.randn_like(true_action)
            noisy_action_fino = noisy_action + (alpha * fino_noise)

            # 4. Modified FINO Target Gradient
            # Standard EqM is (noise - true_action). FINO scales the noise weight by (1-eta).
            target_gradient = (1 - eta) * noise - true_action

            return noisy_action_fino, target_gradient


    
    def _actor_loss(self, states, actions, weights=None):
        
        """
        B: Batch  Dimensions
        A: Action Dimensions
        S: State  Dimensions
        K: Number of Samples
        """
        
        with torch.no_grad():

            rho, noise = self._sample_gamma_and_noise(actions)
                            
            if self.flag_type=="trun":
                ct = self._c_trun(interpolate=self.flag_threshold,rho=rho)
            elif self.flag_type=="lin":
                ct = self._c_linear(rho=rho)
            else:
                raise ValueError(f"\n Invalid Ct FlagType: {self.flag_type!r}. ""Expected either 'lin' or 'trun'. \n")

            ct = self._expand_dims(low_dim_vector=ct,reference=actions)

            
            if self.fino:
                noisy_action, target_gradient =  self._fino_mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions,eta=self.fino_knob)
            else:
                noisy_action, target_gradient =  self._mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions)
            
            scaled_target_grad =  self.scale_gradient * ct * target_gradient
            
            
        pred_grad = self._model(noisy_action,states)
        
        if weights is None:
            if self.mc_samples > 0:
                weights = self._offline_weights_MC(states=states, actions=actions, num_samples=self.mc_samples)
            else:
                weights = self._offline_weights(states=states, actions= actions, num_actions=self.num_actions)
            if self.weighting == "bc":
                weights = torch.ones_like(weights)
        # else: fixed weights (B, 1) given by the caller, e.g. Aware/fit_actor.py with frozen weights;
        # no critic is queried then.

        loss_per_sample_and_dim = F.mse_loss(pred_grad, scaled_target_grad, reduction='none') # Returns (B,A), loss for each sample and each action dimension 
        loss_per_sample = loss_per_sample_and_dim.sum(dim=-1, keepdim=True) # Returns (B,1), loss for each gradient prediction
        weighted_loss_per_sample = loss_per_sample * weights # Returns (B,1)
        loss = weighted_loss_per_sample.mean() # Scalar Loss
        
        #loss= loss_per_sample.mean() # scalar loss for BC case.
        
        return loss
    
    def _offline_weights(self, states, actions, num_actions):
            with torch.no_grad():
                if self.critic == "iql":
                  # IQL: A = min target Q(s, a) - V(s), all at dataset actions (CORL iql.py)
                  q = torch.min(self._target_critic_1(states, actions), self._target_critic_2(states, actions))
                  v = self._value(states)
                else:
                  # pi_actions shape: [num_actions, batch_size, action_dim]
                  pi_actions = self._actor(states, num_actions=num_actions)
                  N, B, A = pi_actions.shape
    
                  # Repeat states and flatten both to [N * B, dim] for the critic
                  repeated_states = states.unsqueeze(0).repeat(N, 1, 1)
                  flat_states = repeated_states.reshape(N * B, -1)
                  flat_pi_actions = pi_actions.reshape(N * B, A)
    
                  # Evaluate Q(s, a_pi) and unflatten back to [N, B, 1]
                  flat_q1 = self._critic_1(flat_states, flat_pi_actions)
                  flat_q2 = self._critic_2(flat_states, flat_pi_actions)
                
                  q1_pi = flat_q1.reshape(N, B, -1)
                  q2_pi = flat_q2.reshape(N, B, -1)
    
                  # Take min across critics, then average across actions to get V(s)
                  q_pi = torch.min(q1_pi, q2_pi)
                  v = q_pi.mean(dim=0) # shape: [B, 1]
    
                  # Evaluate dataset actions [B, action_dim] with states [B, state_dim]
                  q = torch.min(self._critic_1(states, actions), self._critic_2(states, actions))  
    
                adv = q - v
                

                weights = torch.clamp_max(input=torch.exp(adv/self._temperature), max=self._exp_adv_max)
                self._log_weighting(q, v, adv, weights)

            return weights

    def _offline_weights_MC(self, states, actions, num_samples):
        """AWAC weights w = min(exp(A / T), exp_adv_max) with A = min(Q1, Q2)(s, a) - V(s), where V(s) is a
        Monte Carlo estimate over num_samples actor samples per state:
            V(s) = (1 / num_samples) * sum_k min(Q1, Q2)(s, a_k),   a_k ~ actor(s), k = 1 .. num_samples.
        num_samples is the number of actions drawn per state, not the sampler's step count (actor.num_step).

        Why more samples: with one sample, V(s) is noisy. The weight exp(-V / T) is convex in V, so this noise
        makes the weights larger on average than with the exact V (Jensen's inequality). The noise variance,
        and with it this upward bias, shrinks about as 1 / num_samples.

        Shapes: states (B, S), actions (B, A) -> weights (B, 1). The actor draws all B * num_samples samples in
        one batched sampler call, so the compiled sampler (compile: true) records one CUDA graph for this size.

        Extra logs (besides the ones of _offline_weights):
          weighting/v_sample_std           spread of min Q over a state's samples, averaged over the batch.
                                           The noise of V(s) is this divided by sqrt(num_samples).
          weighting/weights_mean_1sample   mean weight if V(s) came from the first sample only, as with
                                           num_actions = 1. Its gap to weighting/weights_mean is the bias that
                                           the extra samples remove.
        """
        with torch.no_grad():
            pi_actions = self._actor(states, num_actions=num_samples)      # (N, B, A), clipped to the action box
            N, B, A = pi_actions.shape
            flat_states = states.unsqueeze(0).expand(N, B, states.shape[-1]).reshape(N * B, -1)
            flat_pi_actions = pi_actions.reshape(N * B, A)
            q_pi = torch.min(self._critic_1(flat_states, flat_pi_actions),
                             self._critic_2(flat_states, flat_pi_actions)).reshape(N, B, 1)
            v = q_pi.mean(dim=0)                                             # (B, 1)

            q = torch.min(self._critic_1(states, actions), self._critic_2(states, actions))   # (B, 1)
            adv = q - v
            weights = torch.clamp_max(torch.exp(adv / self._temperature), self._exp_adv_max)

            self._log_weighting(q, v, adv, weights)
            if N > 1:
                weights_1 = torch.clamp_max(torch.exp((q - q_pi[0]) / self._temperature), self._exp_adv_max)
                gl.log("weighting/v_sample_std",         q_pi.std(dim=0, unbiased=True).mean())
                gl.log("weighting/weights_mean_1sample", weights_1.mean())
        return weights

    def _log_weighting(self, q, v, adv, weights):
        # GPU tensors, no .item(): each .item() makes the CPU wait for the GPU. The logger
        # turns them into numbers once per log_frequency steps (logger.py, dump).
        gl.log("weighting/q_mean",            q.mean())
        gl.log("weighting/v_mean",            v.mean())
        gl.log("weighting/adv_mean",          adv.mean())
        gl.log("weighting/adv_std",           adv.std(unbiased=False))
        gl.log("weighting/weights_mean",      weights.mean())
        gl.log("weighting/weights_max",       weights.max())
        gl.log("weighting/weights_min",       weights.min())
        gl.log("weighting/weights_var",       weights.var(unbiased=False))
        gl.log("weighting/weights_median",    weights.median())
        gl.log("weighting/weights_clip_frac", (weights >= self._exp_adv_max).float().mean())


    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            if self.critic == "iql":
                q_next = self._value(next_states)   # in-sample: no actor action
            else:
                next_actions = self._actor(next_states).squeeze(0)

                q_next = torch.min(
                    self._target_critic_1(next_states, next_actions),
                    self._target_critic_2(next_states, next_actions),
                )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.detach()   # no .item() sync; logged as a tensor

    def _update_value(self, states, actions):
        """IQL's expectile regression of V(s) towards min target Q(s, a) at dataset actions."""
        with torch.no_grad():
            target_q = torch.min(self._target_critic_1(states, actions), self._target_critic_2(states, actions))
        diff = target_q - self._value(states)
        loss = (torch.abs(self._iql_tau - (diff < 0).float()) * diff ** 2).mean()
        self._value_optimizer.zero_grad()
        loss.backward()
        self._value_optimizer.step()
        return loss.detach()   # no .item() sync; logged as a tensor

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._model_optimizer.zero_grad()
        loss.backward()
        self._model_optimizer.step()
        return loss.detach()   # no .item() sync; logged as a tensor

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, truncations, terminals = batch
        dones = terminals
        result = {}
        if self.critic == "iql":   # CORL iql.py order: V, then Q, then the actor
            result["value_loss"] = self._update_value(states, actions)
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss  = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)
        #result = {"actor_loss": actor_loss}
        result.update(critic_loss=critic_loss, actor_loss=actor_loss)
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self._model.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
            **({"value": self._value.state_dict()} if self.critic == "iql" else {}),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._model.load_state_dict(state_dict["model"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])
        if self.critic == "iql":
            self._value.load_state_dict(state_dict["value"])





###################### CODE PARTS FROM AWAC ############################

    # def _actor_loss(self, states, actions):
    #     with torch.no_grad():
    #         pi_action, _ = self._model(states)
    #         v = torch.min(
    #             self._critic_1(states, pi_action), self._critic_2(states, pi_action)
    #         )

    #         q = torch.min(
    #             self._critic_1(states, actions), self._critic_2(states, actions)
    #         )
    #         adv = q - v
    #         weights = torch.clamp_max(
    #             torch.exp(adv / self._awac_lambda), self._exp_adv_max
    #         )

    #     action_log_prob = self._model.log_prob(states, actions)
    #     loss = (-action_log_prob * weights).mean()
    #     return loss
