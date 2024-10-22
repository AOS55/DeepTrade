from typing import Optional

import gymnasium as gym
import numpy as np


class BreakoutAgent:

    def __init__(self,
                 env: gym.Env,
                 lookback_period: int = 10,
                 smooth: Optional[int] = None):

        self._env = env
        self.lookback_period = lookback_period

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
        prices = self._env.unwrapped.price_data
        roll_max, roll_min = self.calculate_rolling_extremes(time, prices)
        roll_mean = (roll_max + roll_min) / 2.0
        output = 40.0 * ((prices[time] - roll_mean) / (roll_max - roll_min))
        # smoothed_output = self.numpy_ewma(output, self.smooth)
        return np.array([output.item()])
