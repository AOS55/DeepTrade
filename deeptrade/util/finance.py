import numpy as np
import pandas as pd

from typing import Union

def calculate_drawdowns(prices: np.ndarray) -> np.ndarray:
    """Calculate the drawdowns."""
    max_prices = np.maximum.accumulate(prices)
    drawdowns = (prices - max_prices) / max_prices
    return drawdowns


def calculate_ewma_volatility(prices: np.ndarray, decay_factor: float = 0.94) -> np.ndarray:
    """Calculate the exponentially weighted moving average (EWMA) volatility."""
    log_returns = np.log(prices[1:] / prices[:-1])
    ewma_variance = np.zeros_like(log_returns)
    ewma_variance[0] = log_returns[0] ** 2
    for t in range(1, len(log_returns)):
        ewma_variance[t] = decay_factor * ewma_variance[t - 1] + (1 - decay_factor) * log_returns[t] ** 2
    ewma_vol = np.sqrt(ewma_variance)
    return ewma_vol


def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
    """Calculate the Sortino ratio."""
    excess_returns = returns - risk_free_rate
    downside_returns = np.where(returns < target_return, returns - target_return, 0)
    downside_deviations = np.sqrt(np.mean(np.square(downside_returns)))
    mean_excess_returns = np.mean(excess_returns)
    sortino_ratio = mean_excess_returns / downside_deviations
    return sortino_ratio
