from typing import Optional

import gymnasium as gym
import numpy as np


class BreakoutAgent:

    def __init__(self,
                 env: gym.Env,
                 lookback_period: int = 10,
                 smooth: Optional[int] = None,
                 pos_size: float = 1.0,  # TODO: Add scalable functionality
                 instrument: Optional[int] = 0):

        self._env = env
        self._env_type = type(env.unwrapped)
        self.instrument = instrument
        self.lookback_period = lookback_period
        self.pos_size = pos_size

        # Setup smoothing
        if smooth is None:
            smooth = max(int(lookback_period / 4.0), 1)
        assert smooth < lookback_period, "Smooth must be less than lookback period."
        self.smooth = smooth

    def calculate_rolling_extremes(self, time, prices: np.ndarray):
        
        window = prices[max(0, time - self.lookback_period + 1):time + 1]
        roll_max = np.max(window)
        roll_min = np.min(window)

        return roll_max, roll_min

    @staticmethod
    def numpy_ewma(data, window):
        """https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm"""
        alpha = 2 /(window + 1.0)
        alpha_rev = 1-alpha
        n = data.shape[0]

        pows = alpha_rev**(np.arange(n+1))

        scale_arr = 1/pows[:-1]
        offset = data[0]*pows[1:]
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        return out
    
    def act(self, state: np.ndarray) -> np.array:
        time = self._env.unwrapped.time
        
        from deeptrade.env import MultiInstrumentEnv
        if self._env_type == MultiInstrumentEnv:
            
            prices = self._env.unwrapped.prices_data
            actions = np.zeros_like(state["positions"])
            if self.instrument is None:
                for idp in range(len(state["positions"])):
                    roll_max, roll_min = self.calculate_rolling_extremes(time, prices[idp])
                    roll_mean = (roll_max + roll_min) / 2.0
                    output = self.pos_size * 40.0 * ((prices[idp][time] - roll_mean) / (roll_max - roll_min))
                    # smoothed_output = self.numpy_ewma(output, self.smooth)
                    actions[idp] = output
            else:
                roll_max, roll_min = self.calculate_rolling_extremes(time, prices[self.instrument])
                roll_mean = (roll_max + roll_min) / 2.0
                output = self.pos_size * 40.0 * ((prices[self.instrument][time] - roll_mean) / (roll_max - roll_min))
                # smoothed_output = self.numpy_ewma(output, self.smooth)
                actions[self.instrument] = output
            return actions
    
        else:
            prices = self._env.unwrapped.price_data
            roll_max, roll_min = self.calculate_rolling_extremes(time, prices)
            roll_mean = (roll_max + roll_min) / 2.0
            output = self.pos_size * 40.0 * ((prices[time] - roll_mean) / (roll_max - roll_min))
            # smoothed_output = self.numpy_ewma(output, self.smooth)
            return np.array([output.item()])
