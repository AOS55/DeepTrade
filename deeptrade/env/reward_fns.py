import torch
from typing import Tuple
from . import termination_fns


def single_instrument_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """Calculate reward as position * price_change.

    Args:
        act: Tensor containing positions
        next_obs: Tensor containing returns/price data

    Returns:
        Tensor of rewards
    """
    delta_price = next_obs[..., -1] - next_obs[..., -2]
    position = act[..., 0]
    rewards = position * delta_price
    return rewards.unsqueeze(-1)

# def make_single_instrument_reward_fn(action_bounds: Tuple[float, float] = (-10.0, 10.0)):
#     """Creates a reward function for the trading environment model that handles position limits."""
#     positions = torch.tensor(0.0)

#     def reward_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
#         nonlocal positions
#         if positions.device != act.device:
#             positions = positions.to(act.device)

#         delta_price = next_obs[..., -1] - next_obs[..., -2]
#         rewards = positions * delta_price

#         # Handle batch dimensions for planning
#         if act.dim() > positions.dim() + 1:  # If we have extra batch dimensions
#             # For planning, we'll use the first observation's position as initial
#             positions_expanded = positions.expand(*act.shape[:-1])
#             position_changes = act[..., 0]
#             new_positions = positions_expanded + position_changes
#             new_positions = torch.clamp(new_positions, action_bounds[0], action_bounds[1])

#             if act.dim() == 2:
#                 positions = new_positions[-1]

#         else:
#             # Normal environment step
#             position_changes = act[..., 0]
#             new_positions = positions + position_changes
#             new_positions = torch.clamp(new_positions, action_bounds[0], action_bounds[1])
#             positions = new_positions

#         return rewards.unsqueeze(-1)

#     def reset():
#         nonlocal positions
#         positions = torch.tensor(0.0)

#     reward_fn.reset = reset
#     return reward_fn


# def single_instrument(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
#     last_price = next_obs[:, -4]
#     current_price = next_obs[:, -3]
#     delta_price = current_price - last_price
#     position = next_obs[:, -2]
#     position = position + act.squeeze()
#     reward = position * delta_price
#     return reward.view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)
