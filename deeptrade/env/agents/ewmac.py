from typing import Optional
import gymnasium as gym
import numpy as np


class EWMACAgent:

    def __init__(self,
                 env: gym.Env,
                 fast_period: int = 10,
                 slow_period: int = 40,
                 pos_size: float = 10.0,
                 instrument: Optional[int] = 0):

        self._env = env
        self._env_type = type(env.unwrapped)
        self.instrument = instrument
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.pos_size = pos_size

    def act(self, state: np.ndarray) -> int:
        
        time = self._env.unwrapped.time
        
        from deeptrade.env import MultiInstrumentEnv
        if self._env_type == MultiInstrumentEnv:
            
            price_data = self._env.unwrapped.prices_data
            actions = np.zeros_like(state["positions"])
            if self.instrument is None:
                for idp in range(len(state["positions"])):
                    fast = np.array(price_data[idp][time-self.fast_period:time]).mean()
                    slow = np.array(price_data[idp][time-self.slow_period:time]).mean()
                    if (fast > slow) and state["positions"][idp] < 1.0:
                        actions[idp] = self.pos_size
                    elif (fast < slow) and state["positions"][idp] > -1.0:
                        actions[idp] = -self.pos_size
                    else:
                        actions[idp] = 0.0
            else:
                fast = np.array(price_data[self.instrument][time-self.fast_period:time]).mean()
                slow = np.array(price_data[self.instrument][time-self.slow_period:time]).mean()
                if (fast > slow) and state["positions"][self.instrument] < 1.0:
                    actions[self.instrument] = self.pos_size
                elif (fast < slow) and state["positions"][self.instrument] > -1.0:
                    actions[self.instrument] = -self.pos_size
                else:
                    actions[self.instrument] = 0.0
            return actions
        else:
            price_data = self._env.unwrapped.price_data
            fast = np.array(price_data[time-self.fast_period:time]).mean()
            slow = np.array(price_data[time-self.slow_period:time]).mean()
            if (fast > slow) and state[-2] < 1.0:
                return np.array([self.pos_size])
            elif (fast < slow) and state[-2] > -1.0:
                return np.array([self.pos_size])
            else:
                return np.array([0.0])
