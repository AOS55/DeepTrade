from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt

from deeptrade.util.finance import calculate_simple_returns

class Account:
    """Simple container for tracking multi-instrument account state."""

    def __init__(self, cash: float, n_instruments: int):
        self.margin = cash
        self.position = np.zeros(n_instruments)  # Using numpy array instead of list
        self.n_instruments = n_instruments

    def get_info(self) -> dict:
        """Returns the account state information."""
        return {
            "position": self.position.tolist(),  # Convert to list for serialization
            "margin": float(self.margin)
        }

    def update(self, position_change: np.ndarray, pnl: float) -> None:
        """Updates account state with new positions and P&L."""
        self.position += position_change
        self.margin += pnl

    def reset(self, cash: float) -> None:
        """Resets the account to initial state."""
        self.margin = cash
        self.position = np.zeros(self.n_instruments)


class MultiInstrumentEnv(gym.Env):

    def __init__(self,
                 n_instruments: int = 3,
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
                     "S0": np.array([100, 150, 200]),
                     "mu": np.array([0.05, 0.07, 0.04]),
                     "cov_matrix": np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]),
                     "dt": 1/252,
                     "n_steps": 1000}
    ):

        super().reset(seed=seed)

        if window > start_time-1:
            raise ValueError(f"window {window} must be less than start time {start_time}")
        if window < period:
            raise ValueError(f"window {window} must be greater than period {period}")

        self.n_instruments = n_instruments
        self.period = period
        self.dt = dt
        self._window = window
        self._starting_cash = starting_cash
        self._start_time = start_time

        self.period = period
        self.dt = dt  # Update frequency of the price_data measured in fractions of a day

        # Initialize price data
        if prices_data is None:
            self.prices_data = self._create_prices_data(price_gen_info)
        else:
            self.prices_data = prices_data

        if self.prices_data.shape[0] != self.n_instruments:
            raise ValueError(
                f"prices_data shape {self.prices_data.shape[0]} "
                f"must equal n_instruments {self.n_instruments}"
            )

        self._n_instruments = n_instruments
        if self.prices_data.shape[0] != self._n_instruments:
            raise ValueError(f"prices_data shape {self.prices_data.shape[0]} must be equal to n_instruments {self._n_instruments}")

        self._end_time = end_time if end_time is not None else len(self.prices_data[0]) - 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_instruments, window),  # [n_instruments, window]
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.full(n_instruments, -10.0),
            high=np.full(n_instruments, 10.0),
            dtype=np.float64
        )

        # Initialize account and state
        self.account = Account(cash=self._starting_cash, n_instruments=self.n_instruments)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)

    def _observe_prices_data(self, time: int) -> np.ndarray:
        """Get windowed price data at given time.

        Args:
            time: Current time step

        Returns:
            Array of shape [n_instruments, window] containing price history
        """
        return self.prices_data[:, time-self._window:time]

    def step(self,
            action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: Desired absolute positions for each instrument
        """
        # Clip actions to valid position ranges
        new_positions = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        # Calculate position change for P&L tracking
        position_change = new_positions - self.account.position

        # Advance time and get new prices
        self.time += self.period
        self.prices = self._observe_prices_data(self.time)

        # Calculate price changes and P&L from previous positions
        delta_prices = (
            self.prices_data[:, self.time] -
            self.prices_data[:, self.time-self.period]
        )
        reward = float(np.sum(self.account.position * delta_prices))

        # Update account with new positions and P&L
        self.account.update(position_change, reward)

        # Check termination conditions
        terminated = self.account.margin < 0
        truncated = self.time >= self._end_time

        # Get info including account state
        info = self.account.get_info()
        info.update({
            "time": self.time,
            "current_prices": self.prices[:, -1].tolist(),
            "delta_prices": delta_prices.tolist(),
        })

        return self.prices, reward, terminated, truncated, info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for environment reset
            options: Additional options for resetting the environment including:
                - start_time: New start time for the episode
                - end_time: New end time for the episode
                - starting_cash: New initial cash amount
                - prices_data: New price data to use

        Returns:
            observation: Initial state dictionary containing returns, positions, and margin
            info: Additional information including account state
        """
        options = options or {}

        if 'start_time' in options:
            self._start_time = options['start_time']
        if 'end_time' in options:
            self._end_time = options['end_time']
        if 'starting_cash' in options:
            self._starting_cash = options['starting_cash']
        if 'prices_data' in options:
            self.prices_data = options['prices_data']

        super().reset(seed=seed)

        # Reset account and time
        self.account.reset(self._starting_cash)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)

        # Get info including account state
        info = self.account.get_info()
        info.update({
            "time": self.time,
            "current_prices": self.prices[:, -1].tolist()
        })

        return self.prices, info

    def _create_prices_data(self, price_gen_info: dict) -> np.ndarray:
        """Create synthetic price data using specified generator.

        Args:
            price_gen_info: Dictionary containing generator parameters

        Returns:
            Array of shape [n_instruments, n_steps] containing price paths
        """
        from deeptrade.models import GBM, OU, JDM

        generators = {
            "GBM": GBM,
            "OU": OU,
            "JDM": JDM
        }

        if price_gen_info["name"] not in generators:
            raise ValueError(f"Time Series process {price_gen_info['name']} not found")

        generator = generators[price_gen_info["name"]](**price_gen_info)
        prices = generator.generate(self.dt, price_gen_info["n_steps"])

        # Ensure prices have shape [n_instruments, n_steps]
        if len(prices.shape) == 1:
            prices = prices.reshape(1, -1)

        return prices

    def render(self, mode: str = 'human'):
        """
        Render the environment with matplotlib plots.
        Shows price histories, positions, and account value.

        Args:
            mode: Rendering mode. Only 'human' is supported.
        """
        import matplotlib.pyplot as plt

        # Create figure on first render call
        if not hasattr(self, 'fig'):
            plt.ion()  # Enable interactive mode
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
            self.fig.suptitle('Multi-Instrument Trading Environment')

            # Initialize price lines
            self.price_lines = []
            for i in range(self.n_instruments):
                line, = self.ax1.plot([], [], label=f'Instrument {i+1}')
                self.price_lines.append(line)

            # Initialize position lines
            self.position_lines = []
            for i in range(self.n_instruments):
                line, = self.ax2.plot([], [], label=f'Position {i+1}')
                self.position_lines.append(line)

            # Initialize margin line
            self.margin_line, = self.ax3.plot([], [], 'r-', label='Account Value')

            # Set labels
            self.ax1.set_ylabel('Prices')
            self.ax2.set_ylabel('Positions')
            self.ax3.set_ylabel('Account Value')
            self.ax3.set_xlabel('Time')

            # Add legends
            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()

            # Initialize data storage
            self.render_prices = [[] for _ in range(self.n_instruments)]
            self.render_positions = [[] for _ in range(self.n_instruments)]
            self.render_margins = []
            self.render_times = []

        # Update data
        self.render_times.append(self.time)
        for i in range(self.n_instruments):
            self.render_prices[i].append(self.prices[i, -1])
            self.render_positions[i].append(self.account.position[i])
        self.render_margins.append(self.account.margin)

        # Update line data
        for i in range(self.n_instruments):
            self.price_lines[i].set_data(self.render_times, self.render_prices[i])
            self.position_lines[i].set_data(self.render_times, self.render_positions[i])
        self.margin_line.set_data(self.render_times, self.render_margins)

        # Adjust axes limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        # Draw and pause briefly
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        """Clean up matplotlib resources."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.ioff()
            # Clear render data
            if hasattr(self, 'render_prices'):
                del self.render_prices
                del self.render_positions
                del self.render_margins
                del self.render_times


if __name__=="__main__":
    env = MultiInstrumentEnv()
    env.step(env.action_space.sample())
