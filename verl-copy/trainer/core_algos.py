# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class SPOValueTracker:
    """Persistent per-prompt value tracker for SPO with Bayesian updates and KL-adaptive forgetting.
    
    Based on the SPO paper: Single-stream Policy Optimization (arXiv:2509.13232)
    This implements a per-prompt value tracker with forgetting rates that adapt based on KL divergence.
    
    For binary rewards {0, 1}, the value function follows a Beta distribution (Equation 5):
        V(x) ~ Beta(α(x), β(x))
        V̂(x) = α(x) / (α(x) + β(x))
    
    Initialization (Equation 6, Algorithm 2):
        1. Set N_0 = 1/(1-ρ_min) = 8 (effective window size)
        2. For each sample x:
           - Collect n_0 outcomes with initial policy π_0
           - Compute v̂_0(x) = (1/n_0) * Σ r^(k)  [average of binary outcomes]
           - Initialize: α_0(x) = N_0 * v̂_0(x), β_0(x) = N_0 * (1 - v̂_0(x))
    
    Bayesian update with forgetting (Equation 7):
        α_new = ρ * α_old + r    [where r ∈ {0, 1}]
        β_new = ρ * β_old + (1 - r)
        V̂_new = α_new / (α_new + β_new)
    
    Per-prompt rho calculation (Reference implementation):
        D = KL(π_old || π_new) per prompt
        ρ = 2^(-D/D_half) where D_half = 0.06
    """
    
    def __init__(self, 
                 rho_min: float = 0.875,
                 rho_max: float = 0.96,
                 target_kl: float = 0.1,
                 v_init: float = 0.5,
                 use_per_sample_rho: bool = True,
                 d_half: float = 0.06
    ):
        """
        Args:
            rho_min: Minimum forgetting rate (corresponds to W_max=25 in paper)
            rho_max: Maximum forgetting rate (corresponds to W_min=8 in paper)
            target_kl: Target KL divergence for adaptive forgetting (global method)
            v_init: Fallback value for prompts not in initialization set (default 0.5)
            use_per_sample_rho: If True, use per-prompt rho calculation like reference implementation
            d_half: Half-life parameter for exponential decay (D_half = 0.06 in reference)
        
        Note:
            For proper initialization per Algorithm 2, call initialize_from_samples()
            after creating the tracker. The v_init parameter is only used as a fallback
            for prompts that were not in the initialization dataset.
        """
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.target_kl = target_kl
        self.v_init = v_init
        self.use_per_sample_rho = use_per_sample_rho
        self.d_half = d_half
        
        # Compute N_0 from Algorithm 2, line 1
        # N_0 = 1/(1-ρ_min) = effective window size
        self.N_0 = 1.0 / (1.0 - rho_min)
        
        # Fallback α and β for prompts not in initialization set
        # These use v_init as a default estimate
        # Proper initialization via Algorithm 2 should use initialize_from_samples()
        self.alpha_init = v_init * self.N_0
        self.beta_init = (1.0 - v_init) * self.N_0
        
        # Per-prompt Beta parameters: sample_id -> (α, β)
        # Equation 5: V(x) ~ Beta(α(x), β(x))
        self.prompt_alpha = {}
        self.prompt_beta = {}
        
        # Per-prompt storage for rho calculation
        # Store log_probs for each sample_id to calculate KL divergence
        self.prompt_log_probs = {}  # sample_id -> torch.Tensor (old log probs)
        self.prompt_D = {}  # sample_id -> float (KL divergence D)
        
        # Track recent KL divergences for adaptive forgetting (global method)
        self.recent_kl_values = []
        self.kl_window_size = 30
        
    def _get_adaptive_rho(self) -> float:
        """
        Compute adaptive forgetting rate based on recent KL divergences.
        
        When KL is high (policy changing rapidly), use faster forgetting (lower ρ, smaller window).
        When KL is low (policy stable), use slower forgetting (higher ρ, larger window).
        
        Returns:
            rho: Forgetting rate in [rho_min, rho_max]
        """
        if len(self.recent_kl_values) < 10:
            # Not enough data, use middle value
            return (self.rho_min + self.rho_max) / 2
        
        # Compute mean KL from recent history
        mean_kl = np.mean(self.recent_kl_values[-20:])
        
        # Linearly interpolate between rho_min and rho_max based on KL
        # Higher KL -> lower rho (faster forgetting)
        # Lower KL -> higher rho (slower forgetting)
        if mean_kl > self.target_kl:
            # Policy changing rapidly, use faster forgetting
            ratio = min(mean_kl / (2 * self.target_kl), 1.0)
            rho = self.rho_max - ratio * (self.rho_max - self.rho_min)
        else:
            # Policy stable, use slower forgetting
            ratio = mean_kl / self.target_kl
            rho = self.rho_min + ratio * (self.rho_max - self.rho_min)
        
        return np.clip(rho, self.rho_min, self.rho_max)
    
    def initialize_from_samples(self, sample_ids: list, outcomes_per_sample: list):
        """
        Initialize value tracker using Algorithm 2 from Appendix A.
        
        This is called automatically by RayPPOTrainer._initialize_spo_from_dataset() 
        at the start of training (when global_step=0)
        
        For each sample, collect outcomes and compute per-sample initial value estimate:
            v̂_0(x) = average of outcomes for sample x
            α_0(x) = N_0 * v̂_0(x)
            β_0(x) = N_0 * (1 - v̂_0(x))
        
        Args:
            sample_ids: List of sample identifiers (e.g., dataset indices)
                       Each sample_id uniquely identifies a (text, image) combination
            outcomes_per_sample: List of lists, where outcomes_per_sample[i] contains
                                binary outcomes for sample_ids[i]
        
        Example:
            # Dataset indices as sample IDs
            sample_ids = [0, 1, 2]  # sample 0, 1, 2 from dataset
            outcomes_per_sample = [
                [1],      # Single outcome for sample 0 → v̂_0 = 1.0
                [0],      # Single outcome for sample 1 → v̂_0 = 0.0
                [1, 0, 1] # Multiple outcomes for sample 2 → v̂_0 = 0.667
            ]
            tracker.initialize_from_samples(sample_ids, outcomes_per_sample)
        """
        for sample_id, outcomes in zip(sample_ids, outcomes_per_sample):
            # Compute v̂_0(x) = average of outcomes (Algorithm 2, line 4)
            if isinstance(outcomes, torch.Tensor):
                v_0 = torch.mean(outcomes.float()).item()
            else:
                v_0 = np.mean(outcomes)
            
            # Initialize α and β (Algorithm 2, line 5)
            self.prompt_alpha[sample_id] = self.N_0 * v_0
            self.prompt_beta[sample_id] = self.N_0 * (1 - v_0)
    
    def update(self, sample_ids: list, outcomes: torch.Tensor, kl_values: torch.Tensor = None, 
               per_prompt_rho: dict = None):
        """
        Update the value tracker for a batch of samples using Beta-Bernoulli Bayesian update.
        
        Implements Equation 7 from Section 4.1:
            α_new = ρ * α_old + r
            β_new = ρ * β_old + (1 - r)
            V̂_new = α_new / (α_new + β_new)
        
        Supports two methods for rho calculation:
        1. Global adaptive rho (original): rho adapts based on global KL history
        2. Per-prompt rho (reference implementation): rho = 2^(-D/D_half) where D is per-prompt KL
        
        Args:
            sample_ids: List of sample identifiers (e.g., dataset indices)
                       Each sample_id uniquely identifies a (text, image) combination
            outcomes: Binary outcomes (0 or 1) for each sample, shape (batch_size,)
            kl_values: Optional KL divergences per sample for adaptation, shape (batch_size,)
            per_prompt_rho: Optional dict mapping sample_id -> rho for per-prompt rho calculation
        """
        # Update KL history for adaptive forgetting (global method)
        if kl_values is not None and not self.use_per_sample_rho:
            mean_kl = torch.mean(kl_values).item()
            self.recent_kl_values.append(mean_kl)
            if len(self.recent_kl_values) > self.kl_window_size:
                self.recent_kl_values.pop(0)
        
        # Update each sample's Beta parameters (Equation 7)
        for i, (sample_id, outcome) in enumerate(zip(sample_ids, outcomes.tolist())):
            if sample_id not in self.prompt_alpha:
                # Initialize unseen sample with fallback prior
                # (Proper initialization should be done via initialize_from_samples)
                self.prompt_alpha[sample_id] = self.alpha_init
                self.prompt_beta[sample_id] = self.beta_init
            
            # Get rho for this sample
            if per_prompt_rho is not None and sample_id in per_prompt_rho:
                # Per-prompt rho (reference implementation)
                rho = per_prompt_rho[sample_id]
            else:
                # Global adaptive rho
                rho = self._get_adaptive_rho()
            
            # Bayesian update with forgetting (Equation 7)
            # For binary reward r ∈ {0, 1}:
            #   α_new = ρ * α_old + r
            #   β_new = ρ * β_old + (1 - r)
            old_alpha = self.prompt_alpha[sample_id]
            old_beta = self.prompt_beta[sample_id]
            
            self.prompt_alpha[sample_id] = rho * old_alpha + outcome
            self.prompt_beta[sample_id] = rho * old_beta + (1 - outcome)
    
    def get_values(self, sample_ids: list) -> torch.Tensor:
        """
        Get value estimates for a batch of samples.
        
        Implements Equation 5 from Section 4.1:
            V̂(x) = α(x) / (α(x) + β(x))
        
        Args:
            sample_ids: List of sample identifiers (e.g., dataset indices)
                       Each sample_id uniquely identifies a (text, image) combination
            
        Returns:
            values: Tensor of value estimates, shape (batch_size,)
        """
        values = []
        for sample_id in sample_ids:
            if sample_id in self.prompt_alpha:
                # Compute value from Beta parameters (Equation 5)
                alpha = self.prompt_alpha[sample_id]
                beta = self.prompt_beta[sample_id]
                value = alpha / (alpha + beta)
                values.append(value)
            else:
                # Use initial value for unseen samples (from Equation 6)
                # Fallback for samples not seen during initialization
                value = self.alpha_init / (self.alpha_init + self.beta_init)
                values.append(value)
        return torch.tensor(values, dtype=torch.float32)
    
    def calculate_per_prompt_rho(self, sample_ids: list, new_log_probs_list: list, 
                                   response_masks: list) -> dict:
        """
        Calculate per-prompt rho values based on KL divergence approximation.
        
        Reference implementation formula:
            D = KL(π_old || π_new) = Σ_t mask_t * (old_log_prob_t - new_log_prob_t)
            ρ = 2^(-D/D_half)
            ρ = clip(ρ, rho_min, rho_max)
        
        Args:
            sample_ids: List of sample identifiers
            new_log_probs_list: List of new log prob tensors for each sample
            response_masks: List of response masks for each sample
            
        Returns:
            per_prompt_rho: Dict mapping sample_id -> rho value
        """
        per_prompt_rho = {}
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in self.prompt_log_probs:
                # First time seeing this prompt, no old log probs yet
                # Use max rho (minimal forgetting) for new prompts
                per_prompt_rho[sample_id] = self.rho_max
                continue
            
            # Get old and new log probs
            old_log_probs = self.prompt_log_probs[sample_id].to(new_log_probs_list[i].device)
            new_log_probs = new_log_probs_list[i]
            mask = response_masks[i]
            
            # Calculate KL divergence: KL(old || new) = Σ mask * (old_log_prob - new_log_prob)
            # This measures how much the policy has changed for this specific prompt
            kl_div = ((old_log_probs - new_log_probs) * mask).sum()
            kl_div_value = kl_div.item()
            
            # Handle NaN or infinite KL values
            if not np.isfinite(kl_div_value):
                # If KL is not finite, use max rho (minimal forgetting) as a safe fallback
                per_prompt_rho[sample_id] = self.rho_max
                continue
            
            # Store D for tracking
            self.prompt_D[sample_id] = kl_div_value
            
            # Calculate rho using exponential decay: ρ = 2^(-D/D_half)
            # D_half = 0.06 is the half-life parameter (when D = 0.06, rho = 0.5)
            # Larger D (more policy change) → smaller rho (more forgetting)
            # Smaller D (less policy change) → larger rho (less forgetting)
            
            # Add numerical safeguards to prevent overflow/underflow
            exponent = -kl_div_value / self.d_half
            
            # Clip exponent to prevent overflow
            if exponent > 50:
                rho = self.rho_max  # D is very negative (shouldn't happen), use max rho
            elif exponent < -50:
                rho = self.rho_min  # D is very large, use min rho (max forgetting)
            else:
                rho = 2.0 ** exponent
                # Clip to [rho_min, rho_max]
                rho = max(self.rho_min, min(self.rho_max, rho))
            
            per_prompt_rho[sample_id] = rho
        
        return per_prompt_rho
    
    def reset(self):
        """Reset the tracker to initial state."""
        self.prompt_alpha = {}
        self.prompt_beta = {}
        self.recent_kl_values = []


class SPOPrioritizedSampler:
    """Prioritized sampling for SPO adaptive curriculum learning.
    
    Based on the SPO paper and reference implementation:
    - Weight prompts by uncertainty: sqrt(p*(1-p)) where p = V(x)
    - Samples with p ≈ 0.5 have highest uncertainty → highest weight
    - Samples with p ≈ 0 or p ≈ 1 are easy/hard → lower weight
    - This enables adaptive curriculum focusing on boundary samples
    """
    
    def __init__(self, 
                 use_uncertainty_weighting: bool = True,
                 priority_alpha: float = 1.0,
                 priority_epsilon: float = 1e-6):
        """
        Args:
            use_uncertainty_weighting: If True, use sqrt(p*(1-p)) weighting (reference impl)
                                      If False, use advantage-based prioritization
            priority_alpha: Scaling factor for priorities
            priority_epsilon: Small value to avoid zero priorities
        """
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon
        
        # Track sample weights
        self.sample_weights = {}
        
    def update_weights_from_value_tracker(self, value_tracker: SPOValueTracker, 
                                          sample_ids: list = None):
        """
        Update sample weights based on uncertainty from value tracker.
        
        Reference implementation formula:
            weight = sqrt(p * (1-p))
            where p = V(x) = α(x) / (α(x) + β(x))
        
        This gives maximum weight to samples with p ≈ 0.5 (highest uncertainty)
        and minimum weight to samples with p ≈ 0 or p ≈ 1 (very easy/hard).
        
        Args:
            value_tracker: SPOValueTracker instance
            sample_ids: Optional list of sample IDs to update. If None, update all.
        """
        if not self.use_uncertainty_weighting:
            # Skip uncertainty weighting if disabled
            return
            
        if sample_ids is None:
            sample_ids = list(value_tracker.prompt_alpha.keys())
        
        for sample_id in sample_ids:
            if sample_id in value_tracker.prompt_alpha:
                # Get value estimate p = V(x)
                alpha = value_tracker.prompt_alpha[sample_id]
                beta = value_tracker.prompt_beta[sample_id]
                p = alpha / (alpha + beta)
                
                # Calculate uncertainty weight: sqrt(p * (1-p))
                # This is the standard deviation of a Bernoulli(p) random variable
                uncertainty = (p * (1 - p)) ** 0.5
                
                # Store weight
                self.sample_weights[sample_id] = uncertainty + self.priority_epsilon
        
    def update_priorities(self, sample_ids: list, advantages: torch.Tensor):
        """
        Update priorities based on advantage magnitudes (fallback method).
        
        This is the original implementation, used when uncertainty weighting is disabled.
        
        Args:
            sample_ids: List of sample identifiers
            advantages: Tensor of advantage values (batch_size,)
        """
        if not self.use_uncertainty_weighting:
            # Skip priority updates when uncertainty weighting is disabled
            return
            
        # Compute priorities based on advantage magnitudes
        advantage_magnitudes = torch.abs(advantages)
        priorities = advantage_magnitudes + self.priority_epsilon
        
        # Update sample priorities
        for i, sample_id in enumerate(sample_ids):
            self.sample_weights[sample_id] = priorities[i].item()
    
    def get_sample_weights(self, sample_ids: list) -> torch.Tensor:
        """
        Get sampling weights for given sample IDs.
        
        Args:
            sample_ids: List of sample identifiers
            
        Returns:
            weights: Tensor of sampling weights (unnormalized)
        """
        if not self.use_uncertainty_weighting:
            # Return uniform weights when uncertainty weighting is disabled
            return torch.ones(len(sample_ids), dtype=torch.float32)
            
        weights = []
        for sample_id in sample_ids:
            if sample_id in self.sample_weights:
                weights.append(self.sample_weights[sample_id])
            else:
                # Fallback: assume p=0.5 (maximum uncertainty)
                weights.append(0.5 + self.priority_epsilon)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Apply priority alpha scaling
        weights = weights ** self.priority_alpha
        
        return weights
    
    def get_sampling_probabilities(self, sample_ids: list) -> torch.Tensor:
        """
        Get normalized sampling probabilities for given sample IDs.
        
        Args:
            sample_ids: List of sample identifiers
            
        Returns:
            probabilities: Tensor of sampling probabilities (sum to 1)
        """
        weights = self.get_sample_weights(sample_ids)
        
        # Normalize to probabilities
        probabilities = weights / torch.sum(weights)
        
        return probabilities
    
    def sample_indices(self, sample_ids: list, num_samples: int, 
                       replacement: bool = False) -> list:
        """
        Sample indices based on uncertainty weights.
        
        Args:
            sample_ids: List of all available sample IDs
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            
        Returns:
            sampled_indices: List of sampled indices into sample_ids
        """
        probabilities = self.get_sampling_probabilities(sample_ids)
        
        # Sample with torch.multinomial
        sampled_indices = torch.multinomial(
            probabilities, 
            num_samples=num_samples, 
            replacement=replacement
        ).tolist()
        
        return sampled_indices
    
    def reset(self):
        """Reset the sampler to initial state."""
        self.sample_weights = {}


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    SPO = "spo"


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns

# @torch.no_grad()
# def compute_grpo_outcome_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     index,  # Any hashable type (UUID strings, integers, etc.)
#     eps: float = 1e-6,
#     log_probs: torch.Tensor = None,
#     alpha: float = 0.4,
#     kappa: float = 2.0,
#     use_entropy_shaping: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute advantage for GRPO, operating only on Outcome reward
#     (with only one scalar reward for each response).
#     Optionally applies entropy-based advantage shaping if log_probs are provided.

#     Args:
#         token_level_rewards: `(torch.Tensor)`
#             shape: (bs, response_length)
#         response_mask: `(torch.Tensor)`
#             shape: (bs, response_length)
#         index: `(torch.Tensor)`
#             shape: (bs,)
#         eps: `(float)`
#             epsilon value to avoid division by zero
#         log_probs: `(torch.Tensor)`, optional
#             Log probabilities from policy model for entropy computation
#             shape: (bs, response_length)
#         alpha: `(float)`, default 0.4
#             Scaling factor for entropy term in advantage shaping
#         kappa: `(float)`, default 2.0
#             Denominator for advantage magnitude term in advantage shaping

#     Returns:
#         advantages: `(torch.Tensor)`
#             shape: (bs, response_length)
#         returns: `(torch.Tensor)`
#             shape: (bs, response_length)

#     """

#     scores = token_level_rewards.sum(dim=-1)
#     id2score = defaultdict(list)
#     id2mean, id2std = {}, {}

#     bsz = scores.shape[0]
#     for i in range(bsz):
#         id2score[index[i]].append(scores[i])

#     for idx in id2score:
#         assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
#         id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
#         id2std[idx] = torch.std(torch.tensor(id2score[idx]))

#     for i in range(bsz):
#         scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

#     returns = scores.unsqueeze(-1) * response_mask
#     advantages = returns
    
#     # Apply entropy-based advantage shaping if enabled and log_probs are provided
#     if use_entropy_shaping and log_probs is not None:
#         # Compute per-token entropy from log probabilities (memory-efficient)
#         entropy = compute_token_entropy(log_probs, response_mask)

#         # Apply entropy-based advantage shaping
#         entropy_shaping_term = compute_entropy_advantage_shaping(
#             advantages, entropy, response_mask, alpha, kappa
#         )
#         advantages = advantages + entropy_shaping_term
    
#     return advantages, returns

@torch.no_grad()
def compute_token_entropy(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token entropy directly from log probabilities of generated tokens.
    This is much simpler and more memory efficient than top-k approximation.

    For each token position t in an output o, the entropy is approximated as:
    H_t = -∑_{v} π_θ(v|q,o<t) log π_θ(v|q,o<t)

    Args:
        log_probs: Tensor of shape (bs, response_length)
            Log probabilities of the generated tokens from the policy model
        response_mask: Tensor of shape (bs, response_length)
            Mask indicating valid tokens

    Returns:
        entropy: Tensor of shape (bs, response_length)
            Per-token uncertainty values (negative log probability)
    """
    # Compute entropy using the correct formula: H = -P * log(P)
    # Convert log probabilities to probabilities first
    probs = torch.exp(log_probs)
    # True entropy: H = -P * log(P)
    entropy = -probs * log_probs  # (bs, response_length)

    # Apply response mask to zero out invalid positions
    entropy = entropy * response_mask

    return entropy


@torch.no_grad()
def compute_entropy_advantage_shaping(
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.4,
    kappa: float = 2.0,
) -> torch.Tensor:
    """
    Compute entropy-based advantage shaping term as described in the paper:
    "Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs"

    The entropy-based advantage term is:
    ψ(H_t) = min(α·H_t^detach, |A_t|/κ)

    Args:
        advantages: Tensor of shape (bs, response_length)
            Original advantages (used for magnitude term)
        entropy: Tensor of shape (bs, response_length)
            Per-token entropy values (will be detached)
        response_mask: Tensor of shape (bs, response_length)
            Mask indicating valid tokens
        alpha: float, default 0.4
            Scaling factor for entropy term
        kappa: float, default 2.0
            Denominator for advantage magnitude term

    Returns:
        entropy_shaping_term: Tensor of shape (bs, response_length)
            Entropy shaping term to be added to advantages
    """
    # Detach entropy to prevent gradients flowing through it
    entropy_detached = entropy.detach()

    # Compute entropy term: α·H_t^detach
    entropy_term = alpha * entropy_detached

    # Compute advantage magnitude term: |A_t|/κ
    advantage_magnitude_term = torch.abs(advantages) / kappa

    # Compute ψ(H_t) = min(α·H_t^detach, |A_t|/κ)
    psi = torch.min(entropy_term, advantage_magnitude_term)

    # Apply response mask to entropy shaping term
    psi = psi * response_mask

    return psi


@torch.no_grad()
def compute_visual_influence_weights(
    full_modality_log_probs: torch.Tensor,
    text_only_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute visual influence weights for each token based on logprob differences.
    
    This function implements a dual-stream approach to identify tokens where visual
    input has the most influence, allowing the model to focus learning on vision-relevant tokens.
    
    Algorithm:
    1. For each token, compute: logprob_diff = logprob_full - logprob_text_only
       - Positive diff: visual input increases probability (visual helps)
       - Negative diff: visual input decreases probability (visual conflicts or irrelevant)
    
    2. Apply temperature-scaled softmax per sequence:
       weight_i = exp(logprob_diff_i / temperature) / Σ_j exp(logprob_diff_j / temperature)
       
    3. These weights indicate relative visual influence across tokens in each sequence
    
    Args:
        full_modality_log_probs: Log probabilities with full input (image + text)
            shape: (batch_size, response_length)
        text_only_log_probs: Log probabilities with text-only input (no image)
            shape: (batch_size, response_length)
        response_mask: Binary mask for valid response tokens
            shape: (batch_size, response_length)
        temperature: Temperature for softmax scaling (default 1.0)
            - Lower temp: more peaked (focus on highest-influence tokens)
            - Higher temp: more uniform (spread across all tokens)
        eps: Small constant for numerical stability
    
    Returns:
        visual_weights: Normalized visual influence weights per token
            shape: (batch_size, response_length)
            These are softmax-normalized within each sequence, summing to 1.0 per sequence
        metrics: Dictionary with visual influence statistics
    """

    full_modality_probs = torch.exp(full_modality_log_probs)
    text_only_probs = torch.exp(text_only_log_probs)
    prob_diff = full_modality_probs - text_only_probs
    
    # Diagnostic check: Warn if logprobs are suspiciously similar
    valid_tokens = response_mask.sum()
    if valid_tokens > 0:
        abs_diff_mean = (torch.abs(prob_diff) * response_mask).sum() / valid_tokens
        if abs_diff_mean < 1e-6:
            print("\n" + "="*80)
            print("WARNING: Visual influence detection issue!")
            print("="*80)
            print(f"Average absolute logprob difference: {abs_diff_mean.item():.2e} (extremely small!)")
            print("This suggests full and text-only logprobs are nearly identical.")
            print("="*80 + "\n")

    # Apply masking to prob_diff before using as weights
    visual_weights = prob_diff * response_mask
    
    # Compute metrics for monitoring
    metrics = {}
    
    # Average prob difference per token (across batch)
    valid_diff = (prob_diff * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/avg_prob_diff"] = valid_diff.item()
    
    # Percentage of tokens where visual significantly increases probability
    # Threshold in probability space: 0.01 = 1% increase in probability
    visual_threshold = 0.05  # Tokens with >0.01 prob difference are "visually helpful"
    visual_helps = ((prob_diff > visual_threshold).float() * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/pct_visual_helps"] = visual_helps.item() * 100

    # Distribution stats: standard deviation of weights (how spread out are they?)
    # This is more robust than entropy for raw prob_diff values
    weight_std = ((visual_weights - valid_diff) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
    weight_std = torch.sqrt(weight_std)
    metrics["visual/weight_std"] = weight_std.item()
    
    # Maximum weight per sequence (how peaked is the distribution?)
    max_weights = visual_weights.max(dim=-1)[0]
    max_weight_mean = (max_weights * (response_mask.sum(dim=-1) > 0).float()).sum()
    max_weight_mean = max_weight_mean / (response_mask.sum(dim=-1) > 0).sum().clamp(min=1)
    metrics["visual/max_weight_per_seq"] = max_weight_mean.item()
    
    # Absolute prob difference magnitude (how much does visual input change things?)
    abs_diff = (torch.abs(prob_diff) * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/abs_prob_diff"] = abs_diff.item()
    
    return visual_weights, metrics


@torch.no_grad()
def compute_visual_influence_weights_log(
    full_modality_log_probs: torch.Tensor,
    text_only_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    visual_threshold: float = 0.1,
    eps: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute visual influence weights for each token based on logprob differences.
    
    This function implements a dual-stream approach to identify tokens where visual
    input has the most influence, allowing the model to focus learning on vision-relevant tokens.
    
    Algorithm:
    1. For each token, compute: logprob_diff = logprob_full - logprob_text_only
       - Positive diff: visual input increases probability (visual helps)
       - Negative diff: visual input decreases probability (visual conflicts or irrelevant)
    
    2. Apply threshold to filter meaningful visual influence (visual_threshold parameter)
       - Only tokens with logprob_diff > visual_threshold get non-zero weights
       - This prevents amplifying tokens with negligible visual influence
    
    3. Apply temperature-scaled softmax per sequence on above-threshold tokens:
       weight_i = exp((logprob_diff_i - threshold) / temperature) / Σ_j exp((logprob_diff_j - threshold) / temperature)
       
    4. These weights indicate relative visual influence across tokens in each sequence
    
    Args:
        full_modality_log_probs: Log probabilities with full input (image + text)
            shape: (batch_size, response_length)
        text_only_log_probs: Log probabilities with text-only input (no image)
            shape: (batch_size, response_length)
        response_mask: Binary mask for valid response tokens
            shape: (batch_size, response_length)
        temperature: Temperature for softmax scaling (default 1.0)
            - Lower temp: more peaked (focus on highest-influence tokens)
            - Higher temp: more uniform (spread across all tokens)
        visual_threshold: Minimum logprob difference to consider as "visual helps" (default 0.1)
            - logprob_diff > threshold means visual helps meaningfully
            - threshold = 0.1 means ~10.5% probability increase (exp(0.1) ≈ 1.105)
            - Higher threshold = more selective (fewer tokens amplified)
            - Lower threshold = less selective (more tokens amplified)
            Recommended: 0.05-0.2 depending on task
        eps: Small constant for numerical stability
    
    Returns:
        visual_weights: Visual influence weights per token (softmax-normalized)
            shape: (batch_size, response_length)
            - Tokens with logprob_diff > visual_threshold: weight = softmax((logprob_diff - threshold) / temp)
            - Tokens with logprob_diff <= visual_threshold: weight = 0
            - Weights sum to ~1.0 per sequence (probability distribution)
            - Larger positive logprob_diff → higher weight (proportional amplification)
            - Small positive logprob_diff → small weight (minimal amplification)
            With reweight_multiplier = 1.0 + strength * visual_weights:
            - Tokens where visual helps a lot: high weight → strong amplification
            - Tokens where visual helps a little: low weight → weak amplification
            - Tokens where visual doesn't help: weight = 0 → no amplification
        metrics: Dictionary with visual influence statistics
    """
    # Compute logprob difference: how much does visual input change token probabilities?
    # Positive = visual increases prob, Negative = visual decreases prob
    logprob_diff = full_modality_log_probs - text_only_log_probs  # (bs, seq_len)
    
    # Diagnostic check: Warn if logprobs are suspiciously similar
    valid_tokens = response_mask.sum()
    if valid_tokens > 0:
        abs_diff_mean = (torch.abs(logprob_diff) * response_mask).sum() / valid_tokens
        if abs_diff_mean < 1e-6:
            print("\n" + "="*80)
            print("WARNING: Visual influence detection issue!")
            print("="*80)
            print(f"Average absolute logprob difference: {abs_diff_mean.item():.2e} (extremely small!)")
            print("This suggests full and text-only logprobs are nearly identical.")
            print("="*80 + "\n")
    
    # Compute signed visual weights from logprob_diff (preserve sign, no softmax)
    # We'll normalize per sequence using z-score + tanh to get signed weights in range [-1, 1]
    
    # Per-sequence z-score normalization (centers at 0, scales by std)
    seq_mean_diff = (logprob_diff * response_mask).sum(dim=-1, keepdim=True) / response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    seq_std_diff = torch.sqrt(
        (((logprob_diff - seq_mean_diff) ** 2) * response_mask).sum(dim=-1, keepdim=True) / 
        response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    )
    seq_std_diff = seq_std_diff.clamp(min=eps)
    
    # Z-score normalization per sequence
    # Use softmax-like approach: only amplify tokens where visual helps (logprob_diff > 0)
    # and make amplification proportional to how much visual helps
    
    # Step 1: Create mask for tokens where visual helps MEANINGFULLY
    # Use a threshold to filter out noise and focus on tokens where visual truly matters
    # logprob_diff > 0.1 means visual input increases token probability by exp(0.1) ≈ 1.105x (10.5% increase)
    visual_helps_mask = (logprob_diff > visual_threshold).float()  # 1.0 where visual helps, 0.0 elsewhere
    
    # Step 2: Zero out values below threshold, keep only above-threshold logprob_diff
    positive_logprob_diff = torch.clamp(logprob_diff - visual_threshold, min=0.0)  # Shift and ReLU
    
    # Step 3: Apply temperature scaling to positive values
    scaled_positive_diff = positive_logprob_diff / temperature
    
    # Step 4: Compute exp for softmax, but ONLY for tokens where visual helps
    # This prevents tokens with logprob_diff <= 0 from getting non-zero weights
    exp_scaled_diff = torch.exp(scaled_positive_diff) * visual_helps_mask * response_mask
    # KEY: Multiply by visual_helps_mask to ensure tokens with logprob_diff <= 0 stay at 0
    
    # Normalize per sequence so weights sum to 1.0 (or less if some tokens masked)
    # Only tokens with positive logprob_diff contribute to the sum
    sum_per_seq = exp_scaled_diff.sum(dim=-1, keepdim=True).clamp(min=eps)
    visual_weights = exp_scaled_diff / sum_per_seq  # Softmax over ONLY positive values
    
    # Result: 
    # - logprob_diff > visual_threshold (visual helps meaningfully): weight proportional to exp((logprob_diff - threshold) / temp)
    # - logprob_diff <= visual_threshold (visual doesn't help enough): weight = EXACTLY 0 (masked out)
    # - Weights sum to ~1.0 per sequence (only counting tokens where visual helps above threshold)
    
    # Compute metrics for monitoring
    metrics = {}
    
    # Average logprob difference per token (across batch)
    valid_diff = (logprob_diff * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/avg_logprob_diff"] = valid_diff.item()
    
    # Log the threshold being used
    metrics["visual/threshold"] = visual_threshold
    
    # Percentage of tokens where visual significantly increases probability
    visual_helps = ((logprob_diff > visual_threshold).float() * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/pct_visual_helps"] = visual_helps.item() * 100
    
    # Statistics about visual weights (softmax-normalized)
    # Mean weight per token (should be less than 1/seq_len on average)
    weight_mean = (visual_weights * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/weight_mean"] = weight_mean.item()
    
    # Percentage of tokens with non-zero weights (visual helps, will be amplified)
    # Use small epsilon to handle floating point precision
    pct_nonzero_weight = ((visual_weights > eps).float() * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/pct_amplified_tokens"] = pct_nonzero_weight.item() * 100
    
    # Percentage of tokens with zero weights (visual hurts/neutral, no amplification)
    pct_zero_weight = ((visual_weights <= eps).float() * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/pct_unchanged_tokens"] = pct_zero_weight.item() * 100
    
    # Among amplified tokens, what's the average weight?
    amplified_mask = (visual_weights > eps).float() * response_mask
    if amplified_mask.sum() > 0:
        avg_amplified_weight = (visual_weights * amplified_mask).sum() / amplified_mask.sum()
        metrics["visual/avg_amplified_weight"] = avg_amplified_weight.item()
    else:
        metrics["visual/avg_amplified_weight"] = 0.0
    
    # Maximum weight per sequence (indicates concentration of visual influence)
    max_weights = visual_weights.max(dim=-1)[0]
    max_weight_mean = (max_weights * (response_mask.sum(dim=-1) > 0).float()).sum()
    max_weight_mean = max_weight_mean / (response_mask.sum(dim=-1) > 0).sum().clamp(min=1)
    metrics["visual/max_weight_per_seq"] = max_weight_mean.item()
    
    # Absolute logprob difference magnitude (how much does visual input change things?)
    abs_diff = (torch.abs(logprob_diff) * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/abs_logprob_diff"] = abs_diff.item()
    
    # Ensure visual_weights is a plain tensor, not a TensorDict
    if hasattr(visual_weights, 'batch'):
        # If it's a TensorDict, extract the actual tensor
        visual_weights = visual_weights.batch.get("visual_influence_weights", visual_weights)
    
    return visual_weights, metrics

@torch.no_grad()
def apply_visual_advantage_reweighting(
    advantages: torch.Tensor,
    visual_weights: torch.Tensor,
    response_mask: torch.Tensor,
    reweight_strength: float = 1.0,
    # eps: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reweight token-level advantages using visual influence weights.
    
    This amplifies the gradient signal for tokens where visual input has high influence,
    steering the model to better understand and utilize visual information.
    
    Algorithm:
    1. Original advantage is uniform or globally normalized
    2. Apply visual weights: reweighted_adv = advantage * (1 + strength * (visual_weight - mean))
       - Tokens with high visual influence get amplified
       - Tokens with low visual influence get reduced
    3. Optionally renormalize to preserve gradient scale
    
    Args:
        advantages: Token-level advantages from SPO
            shape: (batch_size, response_length)
        visual_weights: Visual influence weights from compute_visual_influence_weights
            shape: (batch_size, response_length)
        response_mask: Binary mask for valid response tokens
            shape: (batch_size, response_length)
        reweight_strength: Strength of reweighting (0.0 = no reweighting, 1.0 = full)
            Higher values = more aggressive focus on visual-influenced tokens
        eps: Small constant for numerical stability
    
    Returns:
        reweighted_advantages: Advantages reweighted by visual influence
            shape: (batch_size, response_length)
        metrics: Dictionary with reweighting statistics
    """
    # Ensure visual_weights is a tensor, not a TensorDict
    if hasattr(visual_weights, 'batch'):
        # If it's a TensorDict, extract the actual tensor
        visual_weights = visual_weights.batch.get("visual_influence_weights", None)
        if visual_weights is None:
            raise ValueError("visual_weights is a TensorDict but doesn't contain 'visual_influence_weights' key")
    
    # Use signed visual_weights for reweighting
    # visual_weights are now in range [-1, 1] from tanh of normalized logprob_diff
    # - Positive weights: visual helps (amplify)
    # - Negative weights: visual hurts or irrelevant (suppress)
    # - Zero weights: no visual effect
    
    # Create reweighting multiplier: 1.0 + strength * visual_weight
    # For visual_weight > 0: multiplier > 1.0 (amplify)
    # For visual_weight < 0: multiplier < 1.0 (suppress)
    # reweight_multiplier = 1.0 + reweight_strength * visual_weights
    reweight_multiplier = reweight_strength * visual_weights
    
    # Ensure multiplier is always positive and reasonable
    reweight_multiplier = torch.clamp(reweight_multiplier, min=0.01, max=10.0)
    
    # Apply reweighting to advantages
    # reweighted_advantages = advantages * reweight_multiplier
    reweighted_advantages_term = torch.abs(advantages) * reweight_multiplier

    # reweighted_advantages = advantages + reweighted_advantages_term 
    
    # # If needed for stability, we can add back a gentle normalization that preserves
    # # the reweighting effect but prevents explosion:
    # # Option 1: Only normalize if the norm increases too much (for stability)
    # new_adv_norm = torch.sqrt((reweighted_advantages ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1))
    # orig_adv_norm = torch.sqrt((advantages ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1))
    
    # Compute metrics
    metrics = {}
    
    # # Only apply safety normalization if norm increases by more than 5x
    # safety_threshold = 5.0
    # if new_adv_norm > orig_adv_norm * safety_threshold:
    #     scale_factor = (orig_adv_norm * safety_threshold) / (new_adv_norm + eps)
    #     reweighted_advantages = reweighted_advantages * scale_factor
    #     metrics["visual/safety_renorm_applied"] = 1.0
    # else:
    #     metrics["visual/safety_renorm_applied"] = 0.0
    
    # Average multiplier (should be close to 1.0 by construction)
    avg_multiplier = (reweight_multiplier * response_mask).sum() / response_mask.sum().clamp(min=1)
    metrics["visual/avg_reweight_multiplier"] = avg_multiplier.item()
    
    # Max and min multipliers
    valid_multipliers = reweight_multiplier * response_mask + (1.0 - response_mask) * 1.0
    metrics["visual/max_reweight_multiplier"] = valid_multipliers.max().item()
    metrics["visual/min_reweight_multiplier"] = valid_multipliers[response_mask.bool()].min().item() if response_mask.sum() > 0 else 1.0
    
    # Norm change ratio (to verify reweighting is actually changing magnitudes)
    # metrics["visual/adv_norm_ratio"] = (new_adv_norm / orig_adv_norm).item() if orig_adv_norm > 0 else 1.0
    
    # # Correlation between visual weights and |advantages|
    # abs_advantages = torch.abs(reweighted_advantages)
    # correlation = VF.masked_mean(visual_weights * abs_advantages, response_mask)
    # metrics["visual/weight_adv_correlation"] = correlation.item()
    
    # Scale factor applied (disabled)
    # metrics["visual/renorm_scale_factor"] = scale_factor.item()
    
    return reweighted_advantages_term, metrics


@torch.no_grad()
def compute_kl_divergence_between_streams(
    multimodal_log_probs: torch.Tensor,
    text_only_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute KL divergence between multimodal and text-only streams.
    
    This function calculates KL(text_only || multimodal), treating the text-only stream
    as an anchor/reference policy. This regularizes the multimodal stream to not deviate
    too far from the text-only baseline, providing training stability.
    
    KL(text_only || multimodal) = E[log(P_text_only) - log(P_multimodal)]
                                 = E[text_only_log_prob - multimodal_log_prob]
    
    This is added to the loss, so minimizing loss = maximizing similarity to text-only stream.
    
    Args:
        multimodal_log_probs: Log probabilities from multimodal stream (text + image)
            shape: (batch_size, response_length)
        text_only_log_probs: Log probabilities from text-only stream (anchor/reference)
            shape: (batch_size, response_length)
        response_mask: Mask indicating valid response tokens
            shape: (batch_size, response_length)
        eps: Small epsilon for numerical stability
        
    Returns:
        kl_divergence: KL divergence loss per token [batch_size, seq_len]
        metrics: Dictionary of KL divergence metrics
    """
    
    # KL divergence using PyTorch's F.kl_div
    # F.kl_div(input, target, log_target=True) computes KL(target || input)
    # We want KL(text_only || multimodal), so:
    # - input = multimodal_log_probs (what we're regularizing)
    # - target = text_only_log_probs (reference/anchor)
    kl_divergence = F.kl_div(
        input=multimodal_log_probs,   # log P_multimodal
        target=text_only_log_probs,   # log P_text_only (reference)
        log_target=True,              # target is in log space
        reduction='none'              # return per-token KL
    )
    
    # Apply response mask to only compute KL on valid tokens
    masked_kl_divergence = kl_divergence * response_mask
    
    # Compute metrics
    metrics = {}
    
    # Average KL divergence per token
    avg_kl = masked_kl_divergence.sum() / response_mask.sum().clamp(min=1)
    metrics["text_kl/avg_kl_div"] = avg_kl.item()
    
    # Total KL divergence across all tokens
    total_kl = masked_kl_divergence.sum()
    metrics["text_kl/total_kl_div"] = total_kl.item()
    
    # Max KL per token (to detect outliers)
    max_kl = masked_kl_divergence.max()
    metrics["text_kl/max_kl_div"] = max_kl.item()
    
    return masked_kl_divergence, metrics


@torch.no_grad()
def compute_spo_advantage(
    token_level_scores: torch.Tensor,
    response_mask: torch.Tensor,
    value_tracker: SPOValueTracker,
    sample_ids: list,
    kl_divergences: torch.Tensor = None,
    old_log_probs: torch.Tensor = None,
    eps: float = 1e-6,
    normalize_globally: bool = True,
    visual_weights: torch.Tensor = None,
    visual_reweight_strength: float = 0.0,
    use_entropy_shaping: bool = False,
    log_probs: torch.Tensor = None,
    entropy_alpha: float = 0.4,
    entropy_kappa: float = 2.0
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    """
    Compute advantage for Single-stream Policy Optimization (SPO).
    
    Based on the SPO paper: Single-stream Policy Optimization (arXiv:2509.13232)
    Extended with visual-aware advantage reweighting and entropy-based shaping.
    
    CRITICAL: For SPO, advantages and value tracker should use RAW scores (no KL penalty).
    The KL penalty should be handled as a separate loss term (use_kl_loss: true).
    
    Key Algorithm:
    1. Compute sequence-level scores from token_level_scores
    2. Get per-sample value baselines V(x) from tracker
    3. Compute advantage: A = score - V(x)
    4. Normalize advantages globally (optional, for policy gradient stability)
    5. Broadcast to token level
    6. Apply visual-aware reweighting if visual_weights provided
       - Tokens with high visual influence get amplified gradient
       - Tokens with low visual influence get reduced gradient
    7. Apply entropy-based advantage shaping if enabled (independent of visual reweighting)
       - Encourages exploration by adding entropy-based bonus to advantages
       - Formula: ψ(H_t) = min(α·H_t^detach, |A_t|/κ)
    8. Convert scores to binary outcomes {0,1} for value tracker update
    9. Calculate per-prompt rho based on policy change (if enabled)
    10. Update value tracker with binary outcomes using Beta-Bernoulli formula
    
    Args:
        token_level_scores: `(torch.Tensor)`
            shape: (bs, response_length) - RAW task scores (no KL penalty)
            This should be the output from your reward function (e.g., correctness)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length) - Response mask
        value_tracker: `(SPOValueTracker)`
            Persistent per-sample value tracker for baseline estimation
        sample_ids: `(list)`
            List of sample identifiers (e.g., dataset indices)
            Each sample_id identifies a unique prompt
        kl_divergences: `(torch.Tensor)`
            shape: (bs, response_length) - Optional KL divergences for adaptive forgetting
        old_log_probs: `(torch.Tensor)`
            shape: (bs, response_length) - Log probs from current batch for per-prompt rho
        eps: `(float)`
            epsilon value to avoid division by zero
        normalize_globally: `(bool)`
            whether to normalize advantages globally across the batch
        visual_weights: `(torch.Tensor)`
            shape: (bs, response_length) - Optional visual influence weights
            If provided, advantages will be reweighted to focus on visual-influenced tokens
        visual_reweight_strength: `(float)`
            Strength of visual reweighting (0.0 = disabled, 1.0 = full strength)
        use_entropy_shaping: `(bool)`
            Whether to apply entropy-based advantage shaping (independent of visual reweighting)
        log_probs: `(torch.Tensor)`
            shape: (bs, response_length) - Log probabilities for entropy computation
            Required if use_entropy_shaping=True
        entropy_alpha: `(float)`
            Scaling factor for entropy term in advantage shaping (default: 0.4)
        entropy_kappa: `(float)`
            Denominator for advantage magnitude term in entropy shaping (default: 2.0)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
        metrics: `(Dict[str, float])` [optional]
            Metrics about visual reweighting and/or entropy shaping (only if enabled)
    """
    # Compute sequence-level scores (sum over tokens)
    scores = token_level_scores.sum(dim=-1)  # shape: (bs,)
    
    # Get per-sample baselines from value tracker
    # CRITICAL: Value tracker learns E[score], so we use scores for advantage
    baseline_values = value_tracker.get_values(sample_ids)  # shape: (bs,)
    baseline_values = baseline_values.to(scores.device)
    
    # Compute advantages: A = score - V(x)
    # This is mathematically correct: advantage is the difference between
    # observed reward and expected reward (baseline)
    raw_advantages = scores - baseline_values  # shape: (bs,)
    
    # Global batch normalization for policy gradient stability
    # This normalization is ONLY for policy gradient, NOT for value tracking
    if normalize_globally and len(raw_advantages) > 1:
        adv_mean = torch.mean(raw_advantages)
        adv_var = torch.var(raw_advantages, unbiased=False)
        advantages = (raw_advantages - adv_mean) * torch.rsqrt(torch.clamp(adv_var, min=eps))
    else:
        advantages = raw_advantages
    
    # Convert scores to binary outcomes for Beta-Bernoulli update
    # For binary rewards {0, 1}: outcome = score (already binary)
    # For continuous rewards: outcome = 1 if score > 0.5, else 0
    # Since your rewards are binary {0, 1}, this is essentially identity
    binary_outcomes = (scores > 0.5).float()
    
    # Calculate per-prompt rho if enabled and old_log_probs provided
    per_prompt_rho = None
    if value_tracker.use_per_sample_rho and old_log_probs is not None:
        # Calculate rho for ALL samples in batch (not just unique ones)
        # This handles duplicates correctly by computing rho for each occurrence
        log_probs_list = [old_log_probs[i] for i in range(len(sample_ids))]
        masks_list = [response_mask[i] for i in range(len(sample_ids))]
        
        per_prompt_rho = value_tracker.calculate_per_prompt_rho(
            sample_ids, log_probs_list, masks_list
        )
    
    # Update value tracker with binary outcomes
    # IMPORTANT: Update for EACH sample independently, even if duplicates exist
    # This is the correct behavior for SPO - each rollout is an independent update
    if per_prompt_rho is not None:
        # Use per-prompt rho
        value_tracker.update(sample_ids, binary_outcomes, per_prompt_rho=per_prompt_rho)
    elif kl_divergences is not None:
        # Use global adaptive rho with KL
        kl_values = torch.tensor([
            kl_divergences[i].sum().item() / response_mask[i].sum().item()
            for i in range(len(sample_ids))
        ], dtype=kl_divergences.dtype, device=kl_divergences.device)
        value_tracker.update(sample_ids, binary_outcomes, kl_values)
    else:
        # Use global adaptive rho without KL
        value_tracker.update(sample_ids, binary_outcomes)
    
    # Store current log probs for next step's rho calculation
    if value_tracker.use_per_sample_rho and old_log_probs is not None:
        for i, sample_id in enumerate(sample_ids):
            # Always update with latest log probs (even for duplicates)
            value_tracker.prompt_log_probs[sample_id] = old_log_probs[i].detach().cpu()
    
    # Broadcast to token level
    advantages = advantages.unsqueeze(-1) * response_mask  # shape: (bs, response_length)
    
    # Apply entropy-based advantage shaping if enabled (independent of visual reweighting)
    # This encourages exploration by adding entropy-based bonus to advantages
    entropy_metrics = {}
    entropy_shaping_term = torch.zeros_like(advantages)
    if use_entropy_shaping:
        if log_probs is None:
            raise ValueError("log_probs must be provided when use_entropy_shaping=True")
        
        # Compute per-token entropy from log probabilities
        entropy = compute_token_entropy(log_probs, response_mask)
        
        # Apply entropy-based advantage shaping
        # Formula: ψ(H_t) = min(α·H_t^detach, |A_t|/κ)
        entropy_shaping_term = compute_entropy_advantage_shaping(
            advantages=advantages,
            entropy=entropy,
            response_mask=response_mask,
            alpha=entropy_alpha,
            kappa=entropy_kappa
        )
        
        # Add entropy shaping term to advantages
        # advantages = advantages + entropy_shaping_term
        
        # Compute entropy metrics for monitoring
        valid_entropy = entropy * response_mask
        valid_entropy_sum = valid_entropy.sum()
        valid_mask_sum = response_mask.sum().clamp(min=1)
        avg_entropy = valid_entropy_sum / valid_mask_sum
        
        valid_shaping = entropy_shaping_term * response_mask
        valid_shaping_sum = valid_shaping.sum()
        avg_shaping = valid_shaping_sum / valid_mask_sum
        
        entropy_metrics["entropy/avg_entropy"] = avg_entropy.item()
        entropy_metrics["entropy/avg_shaping_term"] = avg_shaping.item()
        entropy_metrics["entropy/max_shaping_term"] = entropy_shaping_term.max().item()
        
        # Compute min shaping term (only for valid tokens)
        valid_shaping_masked = entropy_shaping_term * response_mask
        valid_shaping_flat = valid_shaping_masked[response_mask > 0]
        if valid_shaping_flat.numel() > 0:
            entropy_metrics["entropy/min_shaping_term"] = valid_shaping_flat.min().item()
        else:
            entropy_metrics["entropy/min_shaping_term"] = 0.0
    
        # Apply visual-aware advantage reweighting if enabled (for multimodal learning)
    # This focuses learning on tokens where visual input has high influence
    visual_metrics = {}
    visual_advantages_term = torch.zeros_like(advantages)
    if visual_weights is not None and visual_reweight_strength > 0.0:
        visual_advantages_term, visual_metrics = apply_visual_advantage_reweighting(
            advantages=advantages,
            visual_weights=visual_weights,
            response_mask=response_mask,
            reweight_strength=visual_reweight_strength,
            # eps=eps
        )

    advantages = advantages + entropy_shaping_term + visual_advantages_term
    returns = advantages  # In SPO, returns equal advantages
    
    # Combine all metrics
    all_metrics = {}
    if visual_metrics:
        all_metrics.update(visual_metrics)
    if entropy_metrics:
        all_metrics.update(entropy_metrics)
    
    # Return with optional metrics
    if all_metrics:
        return advantages, returns, all_metrics
    else:
        return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        metrics: dict with the following keys:
            pg_clipfrac_higher: (float)
                a float number indicating the fraction of policy gradient loss being clipped to a higher value
            pg_clipfrac_lower: (float)
                a float number indicating the fraction of policy gradient loss being clipped to a lower value
            ppo_kl: (float)
                a float number indicating the mean KL divergence between the old policy and the new policy
            entropy: (float)
                a float number indicating the mean entropy (H = -P*log(P))
        entropy_loss: `a scalar torch.Tensor`
            entropy loss for optional regularization (same value as metrics["entropy"])
            can be used with entropy_coef to encourage exploration

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp negative_approx_kl to avoid nan kld
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    # negative_approx_kl = torch.clamp(negative_approx_kl, -0.01, 0.01)
    ratio = torch.exp(negative_approx_kl)
    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    # metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)
    
    # Compute proper token-level entropy: H = -P(token) * log(P(token))
    # This is an approximation of true entropy using only the sampled token's probability
    # Convert log_probs to probabilities
    probs = torch.exp(log_probs)  # P(token)
    # Token entropy: H = -P * log(P) = -P * log_prob
    token_entropy = -probs * log_probs  # (bs, response_length)
    # Average entropy: first average over tokens in each sequence, then average over batch
    # This gives the average entropy per token, averaged across all sequences
    metrics["entropy"] = average_loss(token_entropy, response_mask, mode=loss_avg_mode)

    pg_loss = -advantages * ratio  # -ratio * A
    pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
    pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    
    # Compute entropy loss for optional regularization (separate from entropy metric)
    # Entropy loss is the negative of entropy (maximize entropy = minimize negative entropy)
    entropy_loss = average_loss(token_entropy, response_mask, mode=loss_avg_mode)
    
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics, entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item()
    return vf_loss, vf_clipfrac


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
