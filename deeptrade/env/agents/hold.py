import gymnasium as gym
import numpy as np


class HoldAgent:

    """Simple agent that holds its position for the entire episode or until bust."""

    def __init__(self,
                 env: gym.Env,
                 pos_size: float = 5.0):

        self._env = env
        self.pos_size = pos_size

    def act(self, state: np.ndarray) -> int:
        if state[-2] < self.pos_size:
            action = self.pos_size - state[-2]
            action = min(action, self._env.action_space.high[0])
        else:
            action = 0.0
        return np.array([action])
