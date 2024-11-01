import math
from typing import Callable
import torch

def make_single_instrument_termination_fn(initial_margin: float = 1000.0) -> Callable:
    """Creates a termination function for trading environments that tracks margin/equity.
    
    Args:
        initial_margin: Starting margin/cash amount. Defaults to 1000.0.
    
    Returns:
        A termination function that takes (actions, next_observations) and returns
        a boolean tensor indicating termination.
    
    Example:
        >>> term_fn = make_single_instrument_termination_fn(initial_margin=10000.0)
        >>> done = term_fn(actions, next_obs)
        >>> term_fn.reset()  # Reset margin state
    """
    margin = torch.tensor(initial_margin, dtype=torch.float32)
    
    def termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Determines if the trading episode should terminate based on margin requirements.
        
        Args:
            act: Action tensor of shape (batch_size, action_dim)
            next_obs: Next observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Boolean tensor indicating termination for each batch element
        """
        nonlocal margin
        
        # Ensure tensors are on same device
        if margin.device != act.device:
            margin = margin.to(act.device)
            
        # Determine if we're in planning mode by checking batch size
        is_planning = act.shape[0] > 2
        
        if is_planning:
            # Planning mode: evaluate trajectories without updating state
            margin_base = margin.mean() if margin.dim() > 0 else margin
            margin_expanded = margin_base.expand(act.shape[0])
            
            latest_returns = next_obs[..., -1]
            positions = act[..., -1]
            pnl = positions * latest_returns
            
            margin_after_pnl = margin_expanded + pnl
            done = margin_after_pnl < 0
            
        else:
            # Environment step mode: update actual margin state
            if margin.dim() == 0:
                margin = margin.expand(act.shape[0])
            elif margin.shape[0] != act.shape[0]:
                margin = margin.mean().expand(act.shape[0])
            
            latest_returns = next_obs[..., -1]
            positions = act[..., -1]
            pnl = positions * latest_returns
            
            margin = margin + pnl
            done = margin < 0
        
        return done
    
    def reset() -> None:
        """Resets the margin to its initial value."""
        nonlocal margin
        margin = torch.tensor(initial_margin, dtype=torch.float32)
    
    termination_fn.reset = reset
    return termination_fn

def margin_call(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    margin = next_obs[:, -1]
    done = margin < 0
    return done

def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done

def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done
