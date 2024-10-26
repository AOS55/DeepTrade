from typing import Optional, List, Dict

import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces

from deeptrade.util.finance import calculate_simple_returns

class Account:

    def __init__(self,
                 cash: float,
                 n_instruments: int,
                 positions: Optional[List[float]] = None):
        self._margin = cash
        self._n_instruments = n_instruments
        if positions is None:
            self._positions = [0.0] * n_instruments
        else:
            self._positions = positions

    @property
    def positions(self):
        return np.array(self._positions)

    @positions.setter
    def positions(self, values: List[float]):
        self._positions = values

    @property
    def margin(self):
        return np.array(self._margin)

    @margin.setter
    def margin(self, value: float):
        self._margin = value


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
                     "n_steps": 1000}):
        
        super().reset(seed=seed)
        if window > start_time-1:
            raise ValueError(f"window {window} must be less than start time {start_time}")
        if window < period:
            raise ValueError(f"window {window} must be greater than period {period}")
        
        self.period = period
        self.dt = dt  # Update frequency of the price_data measured in fractions of a day
        
        # If price data is not provided, create price data
        if prices_data is None:
            self.prices_data = self._create_prices_data(price_gen_info)
        else:
            self.prices_data = prices_data
        
        self._n_instruments = n_instruments
        if self.prices_data.shape[0] != self._n_instruments:
            raise ValueError(f"prices_data shape {self.prices_data.shape[0]} must be equal to n_instruments {self._n_instruments}")
        
        self._window = window
        self._end_time = end_time if end_time is not None else len(self.prices_data[0]) - 1
        
        self._starting_cash = starting_cash
        self._start_time = start_time
        self.account = Account(cash=self._starting_cash, n_instruments=self._n_instruments)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)
        self.update_state()
        
        self.observation_space = spaces.Dict({
            "returns": spaces.Box(low=-np.inf, high=np.inf, shape=(n_instruments,), dtype=np.float64),"positions": spaces.Box(low=-np.inf, high=np.inf, shape=(n_instruments,), dtype=np.float64), "margin": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        })
        
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(n_instruments,), dtype=np.float64)
        
    
    def step(self, action: np.ndarray) -> tuple:
        
        # Update position with action
        positions = self.account.positions + action
        positions = np.clip(positions, self.action_space.low, self.action_space.high)
        self.account.positions = positions
        
        # Advance price and action
        self.time += self.period
        self.prices = self._observe_prices_data(self.time)
        delta_prices = self.prices[:, -1] - self.prices[:, -1-self.period]
        reward = self.account.positions * delta_prices
        
        # Update margin
        self.account.margin = self.account.margin + reward.sum()
        
        # Terminate if bankrupt (no negative balance)
        if self.account.margin < 0:
            terminated = True
        else:
            terminated = False
        
        if self.time >= self._end_time:
            truncated = True
        else:
            truncated = False
            
        self.update_state()
        
        return self.state, reward, terminated, truncated, {}
    
    def reset(self, 
              seed: Optional[int] = None,
              start_time: Optional[int] = None,
              end_time: Optional[int] = None,
              options: dict = {}) -> tuple:
        
        if seed:
            super().reset(seed=seed)
        
        if start_time is not None:
            self._start_time = start_time
        if end_time is not None:
            self._end_time = end_time
            
        self.account = Account(cash=self._starting_cash, n_instruments=self._n_instruments)
        self.time = self._start_time
        self.prices = self._observe_prices_data(self.time)
        self.update_state()
            
        return self.state, {}
    
    def _create_prices_data(self, price_gen_info: dict) -> np.ndarray:
        """Create price data"""
        from deeptrade.models import GBM, OU, JDM 
        if price_gen_info["name"] == "GBM":
            generator = GBM(**price_gen_info)
        elif price_gen_info["name"] == "OU":
            generator = OU(**price_gen_info)
        elif price_gen_info["name"] == "JDM":
            generator = JDM(**price_gen_info)
        else:
            raise ValueError(f"Time Series process {price_gen_info['name']} not found")

        return generator.generate(self.dt, price_gen_info["n_steps"])
    
    def _observe_prices_data(self, time: int) -> np.ndarray:
        return np.array(self.prices_data)[:, time-self._window:time+1]
    
    def update_state(self):
        simple_returns = self.prices[:, 1:] / self.prices[:, :-1] - 1
        self.state = {
            "returns": simple_returns,
            "positions": self.account.positions,
            "margin": self.account.margin
        }
    
    def render(self):
        pass
    
    def close(self):
        pass


if __name__=="__main__":
    env = MultiInstrumentEnv()
    env.step(env.action_space.sample())
