from typing import Optional
import gymnasium as gym
import numpy as np


class EWMACAgent:

    def __init__(self,
                 fast_period: int = 10,
                 slow_period: int = 40,
                 pos_size: float = 10.0):

        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("Periods must be positive")
        if pos_size <= 0:
            raise ValueError("Position size must be positive")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.pos_size = pos_size

        # Calculate decay factors
        self.fast_alpha = 2.0 / (fast_period + 1)
        self.slow_alpha = 2.0 / (slow_period + 1)

    @staticmethod
    def _calculate_ewma(data: np.ndarray, alpha: float) -> np.ndarray:
        """Vectorized EWMA calculation"""
        weights = (1-alpha)**np.arange(data.shape[1])
        weights = weights[::-1]
        weights /= weights.sum()

        # Calculate weighted sum for each row
        result = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'), 1, data)
        return result[:,-1]

    def act(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:

        actions = np.zeros_like(positions)

        if len(prices.shape) == 1:
            prices = prices.reshape(1, -1)

        if prices.shape[1] < self.slow_period:
            return np.zeros_like(positions)

        fast = self._calculate_ewma(prices, self.fast_alpha)
        slow = self._calculate_ewma(prices, self.slow_alpha)
        trends = fast > slow  # boolean array True if long, False if short

        actions[trends] = self.pos_size
        actions[~trends] = -self.pos_size

        return actions
