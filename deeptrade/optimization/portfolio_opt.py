from typing import Optional

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class PortfolioWeights:
    weights: np.ndarray  # Weights for risky assets
    risk_free_weight: float     # Weight for risk-free asset
    sharpe: float
    vol: float
    ret: float


class PortfolioOptimizer:
    def __init__(self,
                 risk_free_rate: float = 0.0,
                 target_vol: Optional[float] = None,
                 dt: float = 1/252):
        self.risk_free_rate = risk_free_rate
        self.target_vol = target_vol
        self.dt = dt

    def calculate_portfolio_volatility(self, weights: np.ndarray, returns: np.ndarray) -> float:
            """Calculate portfolio volatility

            Args:
                weights: shape [n_instruments]
                returns: shape [n_instruments, n_times]
            """
            if isinstance(weights, PortfolioWeights):
                weights = weights.weights

            # Calculate covariance matrix [n_instruments, n_instruments]
            cov = np.cov(returns)

            # Calculate portfolio volatility (annualized)
            vol = np.sqrt(weights.T @ cov @ weights) * np.sqrt(1/self.dt)  # Assuming daily data
            return vol

    def calculate_sharpe_ratio(self, weights: np.ndarray, returns: np.ndarray) -> float:
            """Calculate Sharpe ratio

            Args:
                weights: shape [n_instruments]
                returns: shape [n_instruments, n_times]
            """
            if isinstance(weights, PortfolioWeights):
                weights = weights.weights

            # Calculate portfolio returns [n_times]
            port_returns = returns.T @ weights

            # Calculate annualized return and volatility
            ann_return = np.mean(port_returns) * (1/self.dt)  # Assuming daily data
            ann_vol = self.calculate_portfolio_volatility(weights, returns)

            return (ann_return - self.risk_free_rate) / ann_vol

    def optimize_portfolio(self, returns: np.ndarray) -> PortfolioWeights:
            """Optimize portfolio weights

            Args:
                returns: shape [n_instruments, n_times]
            """
            n_instruments = returns.shape[0]

            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
            ]

            # Define bounds
            bounds = tuple((0, 1) for _ in range(n_instruments))  # no short-selling

            # Initial guess - equal weights
            x0 = np.array([1/n_instruments] * n_instruments)

            # Optimize
            result = minimize(
                lambda w: -self.calculate_sharpe_ratio(w, returns),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            weights = result.x
            vol = self.calculate_portfolio_volatility(weights, returns)

            # Calculate portfolio returns and mean return
            port_returns = returns.T @ weights
            ann_return = np.mean(port_returns) * 252  # Assuming daily data

            sharpe = self.calculate_sharpe_ratio(weights, returns)

            return PortfolioWeights(
                weights=weights,
                risk_free_weight=0.0,
                sharpe=sharpe,
                vol=vol,
                ret=ann_return
            )
