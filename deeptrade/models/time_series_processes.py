from typing import Optional, Tuple
import abc
import numpy as np
import pandas as pd
from gymnasium.utils import seeding


class MultiVariateTimeSeriesGenerator(abc.ABC):
    """
    Base abstract class for for all mulitvariate time series datasets.
    All parameters are scaled based on `dt` the time step size.

    All classes derived from `MultiVariateTimeSeriesGenerator` must implement:
        - ``generate``: method to generate a time series dataset

    Args:
        seed (int, optional): Random seed for reproducibility, based on gymnasium.utils.seeding. Defaults to None and random seed.

    """

    def __init__(self, seed: Optional[int] = None):
        self.np_random, _ = seeding.np_random(seed)

    @staticmethod
    def validate_params(mu: np.ndarray, cov_matrix: np.ndarray):
        """
        Validate the dimensions of the mean vector and covariance matrix.

        Args:
            mu (np.ndarray): Mean vector
            cov_matrix (np.ndarray): Covariance matrix

        """
        # Allow for scalar if single instrument
        if mu.shape[0] == 1 and cov_matrix.shape[0] == 1:
            return
        if mu.shape[0] != cov_matrix.shape[0] or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Dimensions of mu and covariance matrix are inconsistent.")

    def generate_correlated_noise(self, cov_matrix: np.ndarray, steps: int) -> np.ndarray:
        """
        Generate correlated noise using the Cholesky decomposition method.

        Args:
            cov_matrix (np.ndarray): Covariance matrix
            steps (int): Number of steps to generate

        Returns:
            np.ndarray: Correlated noise array of shape [steps, d]

        """
        d = cov_matrix.shape[0]
        if d == 1:
            return self.np_random.normal(size=(steps, 1))
        L = np.linalg.cholesky(cov_matrix)
        Z = self.np_random.normal(size=(steps, d))
        return np.dot(Z, L.T)

    @abc.abstractmethod
    def generate(self, dt: float, n_steps: int) -> np.ndarray:
        """
        Abstract method to generate a multivariate time series dataset.

        Args:
            dt (float): update size for each step, e.g., 1/252 for daily steps
            n_steps (int): Number of steps to generate

        Returns:
            np.ndarray: Time series dataset of shape [n_instruments, n_steps+1]
        """
        pass

    def summary_stats(self, time_series: np.ndarray):
        """
        Summary of statistics for the time series dataset.

        Args:
            time_series (np.ndarray): Time series dataset of shape [n_instruments, n_steps+1]

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and variance of the time series dataset

        """
        mean = np.mean(time_series, axis=1)
        variance = np.var(time_series, axis=1)
        return mean, variance


class GBM(MultiVariateTimeSeriesGenerator):
    """
    Geometric Brownian Motion (GBM) model for generating multivariate time series data.
    This follows the standard GBM process for each instrument. Modelled as follows:

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Args:
        S0 (np.ndarray): Initial prices for all instruments
        mu (np.ndarray): Drift vector for each instrument
        cov_matrix (np.ndarray): Covariance matrix for correlated noise
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """

    def __init__(self, S0: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray, seed: Optional[int] = None, **kwargs):
        super().__init__(seed)
        self.S0 = S0
        self.mu = mu
        self.cov_matrix = cov_matrix
        self.validate_params(mu, cov_matrix)

    def generate(self, dt: float, n_steps: int) -> np.ndarray:
        """
        Rollout a multivariate GBM process.

        Args:
            dt (float): Time step
            n_steps (int): Number of steps to generate

        Returns:
            np.ndarray: Time series dataset of shape [n_instruments, n_steps+1]

        """

        d = len(self.S0)
        prices = np.zeros((d, n_steps + 1))
        prices[:, 0] = self.S0
        dW = self.generate_correlated_noise(self.cov_matrix, n_steps) * np.sqrt(dt)

        # Simulate each instrument's price evolution over time
        for t in range(1, n_steps + 1):
            prices[:, t] = prices[:, t - 1] * np.exp(
                (self.mu - 0.5 * np.diag(self.cov_matrix)) * dt + dW[t - 1]
            )

        return prices


class OU(MultiVariateTimeSeriesGenerator):

    """
    Ornstein-Uhlenbeck (OU) process with growing mean. This follows the standard OU process:

    X_t+1 = X_t + theta * (mu_t - X_t) * dt + sigma * dW_t * sqrt(dt)

    Args:
        x0 (np.ndarray): Initial values for the OU process (for each instrument/variable)
        mu0 (np.ndarray): Initial mean for the OU process
        delta (np.ndarray): Drift vector for the growing mean (for each instrument)
        theta (np.ndarray): Mean reversion speeds (for each instrument)
        sigma (np.ndarray): Volatility for each instrument
        cor (np.ndarray): Correlation matrix for the correlated noise
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    """

    def __init__(self, x0: np.ndarray, mu0: np.ndarray, delta: np.ndarray, theta: np.ndarray, sigma: np.ndarray, corr: np.ndarray, seed: Optional[int] = None, **kwargs):
        super().__init__(seed)
        self.x0 = x0
        self.mu0 = mu0
        self.delta = delta
        self.theta = theta
        self.sigma = sigma
        self.corr = corr
        self.validate_params(mu0, self.corr)  # TODO: validate as correlation matrix, not covariance

    def generate(self, dt: float, n_steps: int) -> np.ndarray:
        """
        Rollout a multivariate OU process with growing mean.

        Args:
            dt (float): Time step
            n_steps (int): Number of steps to generate

        Returns:
            np.ndarray: Time series dataset of shape [n_instruments, n_steps+1]

        """
        d = len(self.x0)
        prices = np.zeros((d, n_steps + 1))
        prices[:, 0] = self.x0

        # Cholesky decomposition for correlated noise
        if len(self.corr) == 1:
            L = np.array([1.0])
        else:
            L = np.linalg.cholesky(self.corr)

        for t in range(1, n_steps + 1):
            mu_t = self.mu0 + self.delta * t * dt  # growing mean
            random_shocks = self.np_random.normal(size=d)
            correlated_shocks = L @ random_shocks
            prices[:, t] = prices[:, t - 1] + self.theta * (mu_t - prices[:, t - 1]) * dt + (self.sigma * correlated_shocks * np.sqrt(dt))

        return prices


class JDM(MultiVariateTimeSeriesGenerator):

    """
    Jump Diffusion Model (JDM) for generating multivariate time series data. This model combines a GBM with jumps.

    Args:
        S0 (np.ndarray): Initial prices for all instruments
        mu (np.ndarray): Drift vector for each instrument
        cov_matrix (np.ndarray): Covariance matrix for correlated noise
        jump_lambda (float): Jump intensity (lambda)
        mu_J (np.ndarray): Mean of jumps in log-normal space (for each instrument)
        sigma_J (np.ndarray): Volatility of jumps in log-normal space (for each instrument)
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    """

    def __init__(self, S0: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray, jump_lambda: float, mu_J: np.ndarray, sigma_J: np.ndarray, seed: Optional[int] = None, **kwargs):
        super().__init__(seed)
        self.S0 = S0
        self.mu = mu
        self.cov_matrix = cov_matrix
        self.jump_lambda = jump_lambda
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.validate_params(mu, cov_matrix)

    def generate(self, dt: float, n_steps: int) -> np.ndarray:
        """
        Rollout a multivariate Jump Diffusion Model (JDM) process.

        Args:
            dt (float): Time step
            n_steps (int): Number of steps to generate

        """
        d = len(self.S0)
        prices = np.zeros((d, n_steps + 1))
        prices[:, 0] = self.S0  # Set initial prices

        dW = self.generate_correlated_noise(self.cov_matrix, n_steps) * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            N = self.np_random.poisson(self.jump_lambda * dt, size=d)

            dS_cont = (self.mu - 0.5 * np.diag(self.cov_matrix)) * dt + dW[t - 1]  # GBM component

            # Apply jumps if any occur
            J = np.ones(d)  # Initialize jump sizes
            for i in range(d):
                if N[i] > 0:
                    # If jumps occur, calculate the product of the jump sizes
                    J[i] = self.np_random.lognormal(self.mu_J[i], self.sigma_J[i], size=N[i]).prod()

            # Update prices: combine the continuous GBM and jump components
            prices[:, t] = prices[:, t - 1] * np.exp(dS_cont) * J

        return prices


class Sine(MultiVariateTimeSeriesGenerator):

    """
    Sine function generator, for testing functions

    Args:
        amp (np.ndarray): Max prices for all signals
        freq (np.ndarray): Frequency for each signal
    """

    def __init__(self, amp: np.ndarray, freq: np.ndarray):
        self.amp = amp
        self.freq = freq

    def generate(self, dt: float, n_steps: int) -> np.ndarray:

        d = len(self.amp)
        prices = np.zeros((d, n_steps + 1))
        t = np.arange(n_steps + 1) * dt

        for idd in range(d):
            prices[idd, :] = self.amp[idd] * np.sin(2 * np.pi * self.freq[idd] * t)

        return prices
