import pytest
import numpy as np
import torch
from deeptrade.env.termination_fns import make_single_instrument_termination_fn
from typing import Tuple


@pytest.fixture
def term_fn_setup() -> Tuple[callable, float]:
    """Fixture providing termination function and initial margin."""
    initial_margin = 1000.0
    term_fn = make_single_instrument_termination_fn(initial_margin)
    return term_fn, initial_margin


def test_basic_environment_step(term_fn_setup):
    """Test basic environment step with positive and negative PnL."""
    term_fn, _ = term_fn_setup
    
    # Setup simple test case
    actions = torch.tensor([[5.0], [-3.0]])  # Long and short positions
    next_obs = torch.zeros((2, 9))
    next_obs[:, -1] = torch.tensor([0.1, -0.2])  # 10% gain and 20% loss
    
    # First step
    done = term_fn(actions, next_obs)
    assert done.shape == (2,)
    assert not done.any(), "Should not terminate with positive margin"
    
    # Reset and test large loss
    term_fn.reset()
    next_obs[:, -1] = torch.tensor([-2.5, -02.5])  # 250% loss
    actions = torch.tensor([[20000.0], [200000.0]])  # Large positions
    done = term_fn(actions, next_obs)
    assert done.any(), "Should terminate with negative margin"
 
    
def test_planning_mode(term_fn_setup):
    """Test planning mode with multiple trajectories."""
    term_fn, initial_margin = term_fn_setup
    
    # Create large batch for planning
    batch_size = 1000
    actions = torch.randn(batch_size, 1)
    next_obs = torch.zeros((batch_size, 9))
    next_obs[:, -1] = torch.randn(batch_size) * 0.1  # Random returns
    
    done = term_fn(actions, next_obs)
    assert done.shape == (batch_size,)
    
    # Check that original margin is preserved after planning
    small_batch = torch.ones((2, 1))
    next_obs_small = torch.zeros((2, 9))
    next_obs_small[:, -1] = torch.tensor([0.1, 0.1])
    done_small = term_fn(small_batch, next_obs_small)
    assert not done_small.any(), "Planning should not affect actual margin"    
    
    
def test_reset_behavior(term_fn_setup):
    """Test reset functionality."""
    term_fn, initial_margin = term_fn_setup
    
    # Create loss that reduces margin
    actions = torch.tensor([[10.0]])
    next_obs = torch.zeros((1, 9))
    next_obs[0, -1] = -0.05  # 5% loss
    
    # First step
    term_fn(actions, next_obs)
    
    # Reset and try again with small position
    term_fn.reset()
    actions = torch.tensor([[1.0]])
    next_obs[0, -1] = -0.1  # 10% loss
    done = term_fn(actions, next_obs)
    assert not done.any(), "Should not terminate after reset with small position"
    
    
def test_edge_cases(term_fn_setup):
    """Test edge cases and boundary conditions."""
    term_fn, _ = term_fn_setup
    
    # Test zero actions
    actions = torch.zeros((1, 1))
    next_obs = torch.ones((1, 9))
    done = term_fn(actions, next_obs)
    assert not done.any(), "Zero position should not terminate"
    
    # Test very small PnL
    actions = torch.tensor([[0.001]])
    next_obs = torch.zeros((1, 9))
    next_obs[0, -1] = 0.001
    done = term_fn(actions, next_obs)
    assert not done.any(), "Small PnL should not cause numerical issues"
    
    # Test exact zero PnL
    actions = torch.zeros((1, 1))
    next_obs = torch.zeros((1, 9))
    done = term_fn(actions, next_obs)
    assert not done.any(), "Zero PnL should not terminate"
    
    
if __name__ == "__main__":
    pytest.main([__file__])