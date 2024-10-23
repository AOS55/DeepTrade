from typing import Optional
import gymnasium as gym
import numpy as np

class HoldAgent:

    """Simple agent that holds its position for the entire episode or until bust."""

    def __init__(self,
                 env: gym.Env,
                 pos_size: float = 5.0,
                 instrument: Optional[int] = None):

        self._env = env
        self._env_type = type(env.unwrapped)
        self.instrument = instrument
        self.pos_size = pos_size

    def act(self, state: np.ndarray) -> np.ndarray:
        from deeptrade.env import MultiInstrumentEnv
        if self._env_type == MultiInstrumentEnv:
            actions = np.zeros_like(state["positions"])
            if self.instrument is None:
                for idp in range(len(state["positions"])):
                    if state["positions"][idp] < self.pos_size:
                        action = self.pos_size - state["positions"][idp]
                        action = min(action, self._env.action_space.high[idp])
                    else:
                        action = 0.0
                    actions[idp] = action
            else:
                if state["positions"][self.instrument] < self.pos_size:
                    action = self.pos_size - state["positions"][self.instrument]
                    action = min(action, self._env.action_space.high[self.instrument])
                else:
                    action = 0.0
                actions[self.instrument] = action
            return actions
        else:
            if state[-2] < self.pos_size:
                action = self.pos_size - state[-2]
                action = min(action, self._env.action_space.high[0])
            else:
                action = 0.0
            return np.array([action])
