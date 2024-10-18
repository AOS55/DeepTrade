from typing import Tuple

import gymnasium as gym
import numpy as np

from deeptrade.env import SingleInstrumentEnv
from deeptrade.util.env import EnvHandler, Freeze


def _is_instrument_env(env: gym.wrappers.TimeLimit) -> bool:
    env = env.unwrapped
    return isinstance(env, SingleInstrumentEnv)


# TODO: Add a test for this to make sure behaves as expected
class FreezeInstrumentEnv(Freeze):

    """
    Provides a context to freeze an instrument environment. 
    
    This context allows the user to manipulate the state of the environment and return it to its
    original state when the context is exited.
    
    Example usage:
    
    ..code-block:: python
    
        env = gym.make("SingleInstrument-v0")
        env.reset()
        action = env.action_space.sample()
        with FreezeSingleInstrument(env):
            env.step(action)
            env.step(action)
            env.step(action)
        
    Args:
        env: (:class: `gym.wrappers.TimeLimit`): The environment to freeze.
    
    """
    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._step_count: int = 0
        self._time: int = 0

        if not _is_instrument_env(env):
            raise ValueError("env must be a SingleInstrument environment.")

    def __enter__(self):
        self._init_state = self._env.unwrapped.state
        self._time = self._env.unwrapped.time

    def __exit__(self, *args):
        self._env.unwrapped.account.position = self._init_state[-2]
        self._env.unwrapped.account.margin = self._init_state[-1]
        self._env.unwrapped.time = self._time


class InstrumentEnvHandler(EnvHandler):

    """
    Env handler for the SingleInstrument gym environment.
    """

    freeze = FreezeInstrumentEnv

    # TODO: Find a way to check this
    @staticmethod
    def is_correct_env_type(env):
        return _is_instrument_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:

        if env_name == "SingleInstrument-v0":
            env = gym.make("SingleInstrument-v0")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

        return env

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """
        Returns the internal state of the environment
        
        Returns a tuple with information that can be passed to :func:`set_env_state` to manually
        set the environment to a the same state when the function was called.
        
        Args:
            env: (:class: `gym.wrappers.TimeLimit`): the environment to get the state of.
        
        Returns:
            tuple: A tuple containing the state of the environment.
        
        """
        return (env.unwrapped.state, env.unwrapped.time)

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """
        Sets the environment to a specific state.
        Assumes the `state` was generated using func `get_current_state`. 
        
        Args:
            state: (tuple): The state to set the environment to.
            env: (:class: `gym.wrappers.TimeLimit`): The environment to set the state of.
        
        """
        env.unwrapped.account.position = state[0][-1]
        env.unwrapped.account.margin = state[0][-2]
        env.unwrapped.time = state[1]
