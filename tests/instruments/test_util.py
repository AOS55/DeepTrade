import numpy as np
import pytest

from deeptrade.util import create_handler_from_str


def _freeze_instrument_gym_env(env_name):
    handler = create_handler_from_str(env_name)
    env = handler.make_env_from_str(env_name)
    env.reset(seed=0)

    seen_obses = []
    seen_rewards = []
    actions = []
    num_steps = 100

    with handler.freeze(env):
        for _ in range(num_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            seen_obses.append(next_obs)
            seen_rewards.append(reward)
            actions.append(action)
            if terminated or truncated:
                break

    print(env.unwrapped.time)
    for a in actions:
        next_obs, reward, _, _, _ = env.step(a)
        ref_obs = seen_obses.pop(0)
        ref_reward = seen_rewards.pop(0)
        np.testing.assert_array_almost_equal(next_obs, ref_obs)
        assert reward == pytest.approx(ref_reward)


def _get_and_set_state(env_name):
    """Test state getter and setter run without errors."""
    handler = create_handler_from_str(env_name)
    env = handler.make_env_from_str(env_name)
    env.reset(seed=0)
    state = handler.get_current_state(env)
    handler.set_env_state(state, env)
    # Test if restore works multiple times
    handler.set_env_state(state, env)


def _transfer_state(env_name):
    """Test that states can be transferred between environments."""
    handler = create_handler_from_str(env_name)
    env0 = handler.make_env_from_str(env_name)
    env0.reset(seed=0)
    state = handler.get_current_state(env0)
    env1 = handler.make_env_from_str(env_name)
    env1.reset(seed=0)
    handler.set_env_state(state, env1)


def test_freeze():
    _freeze_instrument_gym_env("SingleInstrument-v0")


def test_get_and_set_state():
    _get_and_set_state("SingleInstrument-v0")


if __name__=="__main__":
    test_get_and_set_state()
