from typing import Tuple, Dict, Union

import gymnasium as gym
import numpy as np

from deeptrade.env import SingleInstrumentEnv
from deeptrade.env import MultiInstrumentEnv
from deeptrade.util.env import EnvHandler, Freeze


def _is_instrument_env(env: gym.wrappers.TimeLimit) -> bool:
    env = env.unwrapped
    return isinstance(env, Union[SingleInstrumentEnv, MultiInstrumentEnv])


# TODO: Add a test for this to make sure behaves as expected
class FreezeInstrumentEnv(Freeze):
    """Provides a context to freeze an instrument environment."""

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env.unwrapped
        self._prices = None
        self._position = None
        self._margin = None
        self._time = None

        if not isinstance(self._env, (SingleInstrumentEnv, MultiInstrumentEnv)):
            raise ValueError("env must be a SingleInstrument or MultiInstrument environment.")

    def __enter__(self):
        # Store the current state
        self._prices = self._env.prices.copy()
        self._position = self._env.account.position.copy() if hasattr(self._env.account.position, 'copy') else self._env.account.position
        self._margin = self._env.account.margin
        self._time = self._env.time

    def __exit__(self, *args):
        # Restore the saved state
        self._env.prices = self._prices
        self._env.account.position = self._position
        self._env.account.margin = self._margin
        self._env.time = self._time

class InstrumentEnvHandler(EnvHandler):
    """Env handler for the Instrument gym environment."""

    freeze = FreezeInstrumentEnv

    @staticmethod
    def is_correct_env_type(env):
        return _is_instrument_env(env)

    @staticmethod
    def make_env_from_str(env_name) -> gym.Env:

        if env_name == "SingleInstrument-v0":
            env = gym.make("SingleInstrument-v0")
        if env_name == "MultiInstrument-v0":
            env = gym.make("MultiInstrument-v0")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

        return env

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment"""
        return (
            env.unwrapped.prices.copy(),
            env.unwrapped.account.position,
            env.unwrapped.account.margin,
            env.unwrapped.time
        )

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the environment to a specific state."""
        prices, position, margin, time = state
        env.unwrapped.prices = prices
        env.unwrapped.account.position = position
        env.unwrapped.account.margin = margin
        env.unwrapped.time = time
