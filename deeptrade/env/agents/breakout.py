from typing import Optional
import gymnasium as gym
import numpy as np


class BreakoutAgent:

    def __init__(self,
                 lookback_period: int = 10,
                 smooth: Optional[int] = None,
                 pos_size: float = 1.0,
                 vol_target: float = 0.02,
                 vol_scale: str = 'linear',  # 'linear' or 'exponential' scaling
                 threshold: float = 0.5,  # minimum position change to trigger a trade
    ):

        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive.")
        if pos_size <= 0:
            raise ValueError("Position size must be positive.")
        if vol_target <= 0:
            raise ValueError("Volatility target must be positive.")
        if vol_scale not in ['linear', 'exponential']:
            raise ValueError("Volatility scaling must be 'linear' or 'exponential'")

        self.lookback_period = lookback_period
        self.pos_size = pos_size
        self.vol_target = vol_target
        self.vol_scale = vol_scale
        self.threshold = threshold

        # Setup smoothing
        if smooth is None:
            smooth = max(int(lookback_period / 4.0), 1)
        if smooth >= lookback_period:
            raise ValueError("Smooth must be less than lookback period.")
        self.smooth = smooth

    def _calculate_volatility(self, price_window: np.ndarray) -> float:
        """Calculate range-based volatility"""
        high = np.max(price_window)
        low = np.min(price_window)
        mean = np.mean(price_window)
        return (high - low) / mean

    def _scale_by_volatility(self, vol: float, signal: float) -> float:
        """Scale position size based on volatility"""
        vol_ratio = vol / self.vol_target

        if self.vol_scale == 'linear':
            # Linear decay: 1/vol_ratio (capped at 1.0)
            scale = min(1.0, 1.0 / vol_ratio)
        else:  # exponential
            # Exponential decay: exp(-vol_ratio)
            scale = np.exp(-vol_ratio)

        return signal * scale

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

        if prices.shape[1] < self.lookback_period:
            return np.zeros_like(positions)

        for idp in range(len(positions)):
            price_window = prices[idp, -self.lookback_period:]

            # Calculate directional signal
            roll_max = np.max(price_window)
            roll_min = np.min(price_window)
            roll_mean = (roll_max + roll_min) / 2.0
            curr_price = prices[idp, -1]

            # Normalize to -1 to +1 range
            raw_signal = 2.0 * ((curr_price - roll_mean) / (roll_max - roll_min))

            # Calculate and apply volatility scaling
            volatility = self._calculate_volatility(price_window)
            position = self._scale_by_volatility(volatility, raw_signal)

            # Apply final position size scaling
            output = self.pos_size * position

            # Apply smoothing if needed
            if self.smooth > 1:
                alpha = 2.0 / (self.smooth + 1)
                output = self._calculate_ewma(np.array([output]).reshape(1,-1), alpha)[0]

            position_delta = np.abs(output - positions[idp])
            actions[idp] = output if position_delta > self.threshold else positions[idp]

        return actions
