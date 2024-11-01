from typing import Tuple, Dict

import gymnasium as gym
import numpy as np

from deeptrade.env import SingleInstrumentEnv
from deeptrade.env import MultiInstrumentEnv
from deeptrade.util.env import EnvHandler, Freeze


def _is_instrument_env(env: gym.wrappers.TimeLimit) -> bool:
    env = env.unwrapped
    return isinstance(env, SingleInstrumentEnv)

def _is_multiinstrument_env(env: gym.wrappers.TimeLimit) -> bool:
    env = env.unwrapped
    return isinstance(env, MultiInstrumentEnv)


# TODO: Add a test for this to make sure behaves as expected
class FreezeInstrumentEnv(Freeze):
    """Provides a context to freeze an instrument environment."""

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._returns = None
        self._position = None
        self._margin = None
        self._time = None

        if not _is_instrument_env(env):
            raise ValueError("env must be a SingleInstrument environment.")

    def __enter__(self):
        # Store the current state
        self._returns = self._env.unwrapped.returns.copy()
        self._position = self._env.unwrapped.account.position
        self._margin = self._env.unwrapped.account.margin
        self._time = self._env.unwrapped.time

    def __exit__(self, *args):
        # Restore the saved state
        self._env.unwrapped.returns = self._returns
        self._env.unwrapped.account.position = self._position
        self._env.unwrapped.account.margin = self._margin
        self._env.unwrapped.time = self._time


class FreezeMultiInstrumentEnv(Freeze):

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state = None
        self._step_count: int = 0
        self._time: int = 0

        if not _is_multiinstrument_env(env):
            raise ValueError("env must be a MultiInstrument environment.")

    def __enter__(self):
        self._init_state = self._env.unwrapped.state
        self._time = self._env.unwrapped.time

    def __exit__(self, *args):
        self._env.unwrapped.account.positions = self._init_state["positions"]
        self._env.unwrapped.account.margin = self._init_state["margin"]
        self._env.unwrapped.time = self._time


class InstrumentEnvHandler(EnvHandler):
    """Env handler for the SingleInstrument gym environment."""

    freeze = FreezeInstrumentEnv

    @staticmethod
    def is_correct_env_type(env):
        return _is_instrument_env(env)

    @staticmethod
    def make_env_from_str(env_name) -> gym.Env:

        if env_name == "SingleInstrument-v0":
            env = gym.make("SingleInstrument-v0")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

        return env

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment"""
        return (
            env.unwrapped.returns.copy(),
            env.unwrapped.account.position,
            env.unwrapped.account.margin,
            env.unwrapped.time
        )

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the environment to a specific state."""
        returns, position, margin, time = state
        env.unwrapped.returns = returns
        env.unwrapped.account.position = position
        env.unwrapped.account.margin = margin
        env.unwrapped.time = time


class MultiInstrumentEnvHandler(EnvHandler):

    freeze = FreezeMultiInstrumentEnv

    @staticmethod
    def is_correct_env_type(env):
        return _is_multiinstrument_env(env)

    @staticmethod
    def make_env_from_str(env_name) -> gym.Env:

        if env_name == "MultiInstrument-v0":
            env = gym.make("MultiInstrument-v0")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

        return env

    @staticmethod
    def get_current_state(env):

        return (env.unwrapped.state, env.unwrapped.time)

    @staticmethod
    def set_env_state(state, env):

        env.unwrapped.account.positions = state[0]["positions"]
        env.unwrapped.account.margin = state[0]["margin"]
        env.unwrapped.time = state[1]
