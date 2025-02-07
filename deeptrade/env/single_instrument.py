from typing import Optional, Union, Dict, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from deeptrade.util.finance import calculate_simple_returns


class Account:
    """Simple container for tracking account state."""

    def __init__(self, cash: float, position: float = 0.0):
        self.margin = cash
        self.position = position

    def get_info(self) -> dict:
        """Returns the account state information."""
        return {
            "position": float(self.position),
            "margin": float(self.margin)
        }

    def update(self, position_change: float, pnl: float) -> None:
        """Updates account state with new position and P&L."""
        self.position += position_change
        self.margin += pnl

    def reset(self, cash: float) -> None:
        """Resets the account to initial state."""
        self.margin = cash
        self.position = 0.0


class SingleInstrumentEnv(gym.Env):
    """
    Single instrument trading environment that follows gym interface.

    The observation is the returns array over the specified window.
    Account information (position, margin) is provided in the info dict.

    Attributes:
        observation_space: Box space containing returns array
        action_space: Box space for position changes in range [-10, 10]
    """

    def __init__(
        self,
        prices_data: Optional[np.ndarray] = None,
        period: int = 1,
        starting_cash: float = 1000.0,
        start_time: int = 11,
        window: int = 10,
        end_time: Optional[int] = None,
        seed: Optional[int] = None,
        dt: float = 1.0,
        price_gen_info: dict = {
            "name": "GBM",
            "S0": np.array([100.0]),
            "mu": np.array([1.0]),
            "cov_matrix": np.array([1.0]),
            "n_steps": 1000
        }
    ):
        super().reset(seed=seed)

        # Validate input parameters
        if window > start_time - 1:
            raise ValueError(f"window {window} must be less than start time {start_time}")
        if window < period:
            raise ValueError(f"window {window} must be greater than period {period}")

        self.period = period
        self.dt = dt
        self._window = window
        self.starting_cash = starting_cash
        self._start_time = start_time

        # Initialize price data
        if prices_data is None:
            self.prices_data = self._create_prices_data(price_gen_info)
        else:
            self.prices_data = prices_data

        self._end_time = end_time if end_time is not None else len(self.prices_data) - 1

        # Initialize account and time
        self.account = Account(cash=self.starting_cash)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)

        # Define observation and action spaces
        # Observation is just the returns array
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window,),
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float64
        )

    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action: Desired absolute position

        Returns:
            observation: Returns array
            reward: P&L for the step
            terminated: Whether episode ended (e.g. bankruptcy)
            truncated: Whether episode was artificially terminated
            info: Additional information including account state
        """
        # Calculate position change and clip to bounds
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        new_position = action
        position_change = new_position - self.account.position

        # Advance time and calculate reward
        self.time += self.period
        self.prices = self._observe_prices_data(self.time)
        delta_price = self.prices[-1] - self.prices[-1-self.period]
        reward = self.account.position * delta_price

        # Update account
        self.account.update(position_change, reward)

        # Check termination conditions
        terminated = self.account.margin < 0  # Bankruptcy
        truncated = self.time >= self._end_time  # Time limit

        # Get info including account state
        info = self.account.get_info()
        info.update({
            "time": self.time,
            "current_price": self.prices[-1],
            "delta_price": delta_price,
        })

        return self.prices, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for environment reset
            options: Additional options for resetting the environment

        Returns:
            observation: Initial returns array
            info: Additional information including account state
        """

        options = options or {}

        if 'start_time' in options:
            self._start_time = options['start_time']
        if 'end_time' in options:
            self._end_time = options['end_time']
        if 'starting_cash' in options:
            self.starting_cash = options['starting_cash']
        if 'prices_data' in options:
            self.prices_data = options['prices_data']

        super().reset(seed=seed, options=options)

        self.account.reset(self.starting_cash)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)

        info = self.account.get_info()
        info.update({
            "time": self.time,
            "current_price": self.prices[-1]
        })

        return self.prices, info

    def _create_prices_data(self, price_gen_info: dict) -> np.ndarray:
        """Create synthetic price data using specified generator."""
        from deeptrade.models import GBM, OU, JDM

        generators = {
            "GBM": GBM,
            "OU": OU,
            "JDM": JDM
        }

        if price_gen_info["name"] not in generators:
            raise ValueError(f"Time Series process {price_gen_info['name']} not found")

        generator = generators[price_gen_info["name"]](**price_gen_info)
        return generator.generate(self.dt, price_gen_info["n_steps"])[0]

    def _observe_prices_data(self, time: int) -> np.ndarray:
        """Get windowed price data at given time.

        Args:
            time: Current time step

        Returns:
            Array of shape [n_instruments, window] containing price history
        """
        return np.array(self.prices_data[time-self._window:time])

    def render(self, mode: str = 'human'):
        """
        Render the environment with a matplotlib plot.
        Shows price history, positions, and account value.
        Args:
            mode: Rendering mode. Only 'human' is supported.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        # Create figure on first render call
        if not hasattr(self, 'fig'):
            plt.ion()  # Enable interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            self.fig.suptitle('Trading Environment Visualization')

            # Initialize lines
            self.price_line, = self.ax1.plot([], [], 'b-', label='Price')
            self.position_line, = self.ax2.plot([], [], 'g-', label='Position')
            self.margin_line, = self.ax2.plot([], [], 'r-', label='Account Value')

            # Set labels
            self.ax1.set_ylabel('Price')
            self.ax2.set_ylabel('Value')
            self.ax2.set_xlabel('Time')

            # Add legends
            self.ax1.legend()
            self.ax2.legend()

            # Initialize data storage
            self.render_prices = []
            self.render_positions = []
            self.render_margins = []
            self.render_times = []

        # Update data
        self.render_prices.append(self.prices[-1])
        self.render_positions.append(self.account.position)
        self.render_margins.append(self.account.margin)
        self.render_times.append(self.time)

        # Update line data
        self.price_line.set_data(self.render_times, self.render_prices)
        self.position_line.set_data(self.render_times, self.render_positions)
        self.margin_line.set_data(self.render_times, self.render_margins)

        # Adjust axes limits
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()

        # Draw and pause briefly
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        """Clean up matplotlib resources."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # type: ignore
            plt.ioff()  # type: ignore


if __name__ == "__main__":
    # Example usage
    env = SingleInstrumentEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial account info: {info}")

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nAfter step:")
    print(f"Observation shape: {next_obs.shape}")
    print(f"Reward: {reward}")
    print(f"Account info: {info}")
