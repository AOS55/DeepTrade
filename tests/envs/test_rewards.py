import pytest
import torch
from deeptrade.env.reward_fns import single_instrument_reward

def test_zero_position_gives_zero_reward():
    position = torch.tensor([[0.0]])  # zero position
    next_obs = torch.tensor([[1.0, 2.0]])  # price went up by 1
    reward = single_instrument_reward(position, next_obs)

    assert reward.shape == (1, 1)
    assert torch.allclose(reward, torch.tensor([[0.0]]))


def test_positive_position_positive_price_change():
    position = torch.tensor([[5.0]])  # long position
    next_obs = torch.tensor([[10.0, 12.0]])  # price went up by 2
    reward = single_instrument_reward(position, next_obs)

    assert reward.shape == (1, 1)
    assert torch.allclose(reward, torch.tensor([[10.0]]))  # 5 * 2 = 10


def test_negative_position_positive_price_change():
    position = torch.tensor([[-5.0]])  # short position
    next_obs = torch.tensor([[10.0, 12.0]])  # price went up by 2
    reward = single_instrument_reward(position, next_obs)

    assert reward.shape == (1, 1)
    assert torch.allclose(reward, torch.tensor([[-10.0]]))  # -5 * 2 = -10


def test_batched_inputs():
    positions = torch.tensor([[5.0], [-3.0], [0.0]])
    next_obs = torch.tensor([
        [10.0, 12.0],  # +2 change
        [12.0, 11.0],  # -1 change
        [11.0, 11.0],  # 0 change
    ])
    reward = single_instrument_reward(positions, next_obs)

    expected = torch.tensor([[10.0], [3.0], [0.0]])  # [5*2, -3*-1, 0*0]
    assert reward.shape == (3, 1)
    assert torch.allclose(reward, expected)
