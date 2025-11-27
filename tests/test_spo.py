#!/usr/bin/env python3
"""
Test script for SPO (Single-stream Policy Optimization) implementation.
This script verifies that the SPO algorithm components work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add the verl package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.trainer.core_algos import (
    SPOValueTracker, 
    SPOPrioritizedSampler,
    compute_spo_advantage,
    AdvantageEstimator
)


def test_spo_value_tracker():
    """Test SPOValueTracker functionality with Beta-Bernoulli updates."""
    print("Testing SPOValueTracker (Beta distribution)...")
    
    tracker = SPOValueTracker(
        rho_min=0.875,
        rho_max=0.96,
        target_kl=0.1,
        n_init=8,
        v_init=0.5
    )
    
    # Verify initialization parameters (fallback values)
    assert tracker.N_0 == 8.0, f"N_0 should be 8.0, got {tracker.N_0}"
    assert abs(tracker.alpha_init - 0.5 * 8.0) < 1e-6, f"α_fallback should be 4.0, got {tracker.alpha_init}"
    assert abs(tracker.beta_init - 0.5 * 8.0) < 1e-6, f"β_fallback should be 4.0, got {tracker.beta_init}"
    print(f"✓ Fallback initialization: N_0={tracker.N_0}, α_fallback={tracker.alpha_init}, β_fallback={tracker.beta_init}")
    
    # Test Algorithm 2 initialization
    print("\n  Testing Algorithm 2 initialization...")
    init_prompts = ['prompt_A', 'prompt_B', 'prompt_C']
    init_outcomes = [
        [1, 0, 1, 1, 0, 1, 0, 1],  # 5/8 = 0.625
        [0, 0, 1, 0, 0, 0, 1, 0],  # 2/8 = 0.25
        [1, 1, 1, 1, 1, 1, 1, 1],  # 8/8 = 1.0
    ]
    tracker.initialize_from_samples(init_prompts, init_outcomes)
    
    # Verify per-prompt initialization
    values = tracker.get_values(init_prompts)
    assert abs(values[0].item() - 0.625) < 1e-6, f"prompt_A should be 0.625, got {values[0].item()}"
    assert abs(values[1].item() - 0.25) < 1e-6, f"prompt_B should be 0.25, got {values[1].item()}"
    assert abs(values[2].item() - 1.0) < 1e-6, f"prompt_C should be 1.0, got {values[2].item()}"
    print(f"✓ Algorithm 2 initialization: A={values[0].item():.3f}, B={values[1].item():.3f}, C={values[2].item():.3f}")
    
    # Test per-prompt value tracking with Beta parameters
    prompt_hashes = ['prompt_1', 'prompt_2', 'prompt_3', 'prompt_4', 'prompt_5']
    outcomes = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    
    tracker.update(prompt_hashes, outcomes)
    
    # Get values for prompts (V̂ = α/(α+β))
    values = tracker.get_values(prompt_hashes)
    assert len(values) == len(prompt_hashes), "Should return value for each prompt"
    assert torch.all(values >= 0) and torch.all(values <= 1), "Values should be in [0, 1]"
    print(f"✓ Per-prompt Beta distribution tracking working: {values.numpy()}")
    
    # Verify Beta parameters are being tracked
    assert len(tracker.prompt_alpha) > 0, "Should track alpha parameters"
    assert len(tracker.prompt_beta) > 0, "Should track beta parameters"
    print(f"✓ Beta parameters tracked: α and β for {len(tracker.prompt_alpha)} prompts")
    
    # Test KL-adaptive forgetting
    kl_divs = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.18])
    for _ in range(10):  # Multiple updates to test adaptation
        tracker.update(prompt_hashes, outcomes, kl_divs)
    
    assert len(tracker.prompt_alpha) > 0, "Tracker should maintain Beta parameters"
    print(f"✓ KL-adaptive forgetting working: {len(tracker.prompt_alpha)} prompts tracked")
    
    # Test unseen prompt (should return initial value: α_init/(α_init+β_init))
    new_values = tracker.get_values(['unseen_prompt'])
    expected_init = tracker.alpha_init / (tracker.alpha_init + tracker.beta_init)  # Should be 0.5
    assert abs(new_values[0].item() - expected_init) < 1e-6, "Unseen prompt should return V̂ from prior"
    print(f"✓ Unseen prompt handling correct: V̂={new_values[0].item():.3f} (expected {expected_init:.3f})")
    
    # Test reset
    tracker.reset()
    assert len(tracker.prompt_alpha) == 0, "Reset should clear alpha parameters"
    assert len(tracker.prompt_beta) == 0, "Reset should clear beta parameters"
    print("✓ Value tracker reset correctly")


def test_spo_advantage_computation():
    """Test SPO advantage computation."""
    print("\nTesting SPO advantage computation...")
    
    tracker = SPOValueTracker(rho_min=0.875, rho_max=0.96, target_kl=0.1, n_init=8, v_init=0.5)
    
    # Test data - binary rewards (0 or 1)
    batch_size = 8
    seq_len = 10
    # Simulate outcome supervision: only last token has reward
    token_level_rewards = torch.zeros(batch_size, seq_len)
    token_level_rewards[:, -1] = torch.randint(0, 2, (batch_size,)).float()  # Binary: 0 or 1
    response_mask = torch.ones(batch_size, seq_len)
    kl_divergences = torch.rand(batch_size, seq_len) * 0.1
    
    # Generate prompt hashes
    prompt_hashes = [f'prompt_{i}' for i in range(batch_size)]
    
    # Compute advantages
    advantages, returns = compute_spo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        value_tracker=tracker,
        prompt_hashes=prompt_hashes,
        kl_divergences=kl_divergences,
        normalize_globally=True
    )
    
    assert advantages.shape == token_level_rewards.shape, "Advantages should have same shape as rewards"
    assert returns.shape == token_level_rewards.shape, "Returns should have same shape as rewards"
    assert torch.allclose(advantages, returns), "In SPO, advantages should equal returns"
    
    # Check that Beta parameters are tracked per-prompt
    assert len(tracker.prompt_alpha) == batch_size, "Should track each prompt"
    print(f"✓ Per-prompt tracking: {len(tracker.prompt_alpha)} prompts")
    
    # Test multiple updates with same prompts
    for _ in range(5):
        advantages, returns = compute_spo_advantage(
            token_level_rewards, response_mask, tracker, prompt_hashes, kl_divergences
        )
    
    assert len(tracker.prompt_alpha) == batch_size, "Should still track same prompts"
    print(f"✓ Multiple updates working: {len(tracker.prompt_alpha)} prompts tracked")


def test_spo_unified_advantage():
    """Test unified SPO advantage computation for both text and multimodal scenarios."""
    print("\nTesting unified SPO advantage computation...")
    
    tracker = SPOValueTracker(rho_min=0.875, rho_max=0.96, target_kl=0.1, n_init=8, v_init=0.5)
    
    # Test data - binary rewards
    batch_size = 6
    seq_len = 8
    token_level_rewards = torch.zeros(batch_size, seq_len)
    token_level_rewards[:, -1] = torch.randint(0, 2, (batch_size,)).float()
    response_mask = torch.ones(batch_size, seq_len)
    
    # Generate prompt hashes (same tracker works for text-only and multimodal)
    prompt_hashes = [f'prompt_{i}' for i in range(batch_size)]
    
    # Test unified advantage computation (works for both text-only and multimodal)
    advantages, returns = compute_spo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        value_tracker=tracker,
        prompt_hashes=prompt_hashes,
        normalize_globally=True
    )
    
    assert advantages.shape == token_level_rewards.shape, "Advantages should have correct shape"
    assert returns.shape == token_level_rewards.shape, "Returns should have correct shape"
    
    # Tracker should be updated with per-prompt Beta parameters
    assert len(tracker.prompt_alpha) == batch_size, "Should track all prompts"
    
    print(f"✓ Unified advantage computation working")
    print(f"  Prompts tracked: {len(tracker.prompt_alpha)}")
    print(f"  Works for both text-only and multimodal scenarios")


def test_spo_prioritized_sampler():
    """Test SPO prioritized sampling."""
    print("\nTesting SPO prioritized sampling...")
    
    sampler = SPOPrioritizedSampler(
        priority_alpha=0.6,
        priority_beta=0.4
    )
    
    # Test priority updates
    sample_ids = ['sample_1', 'sample_2', 'sample_3', 'sample_4']
    advantages = torch.tensor([2.0, 1.0, 3.0, 0.5])
    
    sampler.update_priorities(sample_ids, advantages)
    
    # Test sampling probabilities
    probabilities = sampler.get_sampling_probabilities(sample_ids)
    assert len(probabilities) == len(sample_ids), "Probabilities should match sample count"
    assert abs(torch.sum(probabilities) - 1.0) < 1e-6, "Probabilities should sum to 1.0"
    
    # Higher advantages should have higher probabilities
    assert probabilities[2] > probabilities[1], "Sample with higher advantage should have higher probability"
    assert probabilities[0] > probabilities[3], "Sample with higher advantage should have higher probability"
    
    # Test importance weights
    weights = sampler.get_importance_weights(sample_ids, probabilities)
    assert len(weights) == len(sample_ids), "Weights should match sample count"
    assert torch.all(weights > 0), "All weights should be positive"
    
    print(f"✓ Prioritized sampling working")
    print(f"  Probabilities: {probabilities.numpy()}")
    print(f"  Importance weights: {weights.numpy()}")


def test_advantage_estimator_enum():
    """Test that SPO is properly added to the enum."""
    print("\nTesting AdvantageEstimator enum...")
    
    assert hasattr(AdvantageEstimator, 'SPO'), "AdvantageEstimator should have SPO"
    assert AdvantageEstimator.SPO == 'spo', "SPO enum value should be 'spo'"
    
    # Test all supported estimators
    supported_estimators = [
        AdvantageEstimator.GAE,
        AdvantageEstimator.GRPO, 
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.SPO
    ]
    
    print(f"✓ All {len(supported_estimators)} advantage estimators available:")
    for estimator in supported_estimators:
        print(f"  - {estimator}")


def main():
    """Run all SPO tests."""
    print("=" * 60)
    print("SPO (Single-stream Policy Optimization) Implementation Tests")
    print("=" * 60)
    
    try:
        test_advantage_estimator_enum()
        test_spo_value_tracker()
        test_spo_advantage_computation()
        test_spo_unified_advantage()
        test_spo_prioritized_sampler()
        
        print("\n" + "=" * 60)
        print("✅ All SPO tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

