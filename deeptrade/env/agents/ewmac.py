import numpy as np
import gymnasium as gym

class EWMACAgent:
    
    def __init__(self,
                 env: gym.Env,
                 fast_period: int = 10,
                 slow_period: int = 40):
        
        self._env = env
        self.fast_period = fast_period
        self.slow_period = slow_period 
        
    def act(self, state: np.ndarray) -> int:
        time = self._env.unwrapped.time
        fast = np.array(self._env.unwrapped.price_data[time-self.fast_period:time]).mean()
        slow = np.array(self._env.unwrapped.price_data[time-self.slow_period:time]).mean()
        if (fast > slow) and state[-2] < 1.0:
            return np.array([10.0])
        elif (fast < slow) and state[-2] > -1.0:
            return np.array([-10.0])
        else:
            return np.array([0.0])
