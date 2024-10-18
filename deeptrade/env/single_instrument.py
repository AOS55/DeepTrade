from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from deeptrade.util.finance import calculate_log_returns, calculate_simple_returns

class Account:
    
    def __init__(self, cash: float, position: float = 0.0):
        self._margin = cash
        self._position = position
        
    @property
    def position(self):
        return np.array([self._position])

    @position.setter
    def position(self, value: float):
        self._position = value
    
    @property
    def margin(self):
        return np.array([self._margin])

    @margin.setter
    def margin(self, value: float):
        self._margin = value


class SingleInstrumentEnv(gym.Env):
    
    def __init__(self,
                 price_data: Optional[np.ndarray] = None,
                 period: int = 1,
                 starting_cash: float = 1000.0,
                 start_time: int = 11,
                 window: int = 10,
                 end_time: Optional[int] = None,
                 seed: Optional[int] = None,
                 price_gen_info: dict = {"starting_price": 0.0, "mean": 0.0, "std": 1.0, "n_days": 1000}):
        
        super().reset(seed=seed)
        if window > start_time-1:
            raise ValueError(f"window {window} must be less than start time {start_time}")
        if window < period:
            raise ValueError(f"window {window} must be greater than period {period}")
        
        # If price data is not provided, create price data
        if price_data is None:
            self.price_data = self._create_price_data(price_gen_info)
        else:
            self.price_data = price_data
        
        self._window = window
        self._end_time = end_time if end_time is not None else len(self.price_data) - 1  # end time step
        
        self.starting_cash = starting_cash
        self._start_time = start_time
        self.account = Account(cash=self.starting_cash)
        self.time = self._start_time  # current world time
        self.prices = self._observe_price_data(self.time)  # prices of the instrument at the current time step
        self.period = period  # how long the position is held over before the next action
        self.update_state()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state.shape[0],), dtype=np.float64)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float64)
    
    def step(self, action: float):
        
        # Update position with action
        position = self.account.position[0] + action[0]
        if position > self.action_space.high[0]:
            position = self.action_space.high[0]
        if position < self.action_space.low[0]:
            position = self.action_space.low[0]
        self.account.position = position  # position just moves based on action
        
        # Advance price and action
        self.time += self.period
        self.prices = self._observe_price_data(self.time)
        delta_price = self.prices[-1] - self.prices[-1-self.period]
        reward = self.account.position[0] * delta_price
        
        # Update margin
        self.account.margin = self.account.margin[0] + reward
        
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
              options: dict = {}): 
        
        if seed:
            super().reset(seed=seed)
        
        if start_time is not None:
            self._start_time = start_time
        if end_time is not None:
            self._end_time = end_time
        
        self.account = Account(cash=self.starting_cash)
        self.time = self._start_time
        self.prices = self._observe_price_data(self.time)
        self.update_state()
        
        return self.state, {}
    
    def _create_price_data(self, price_gen_info: dict):
        """Create price data from random walk."""
        # rng = np.random.default_rng(price_gen_info["price_seed"])
        y_data = [price_gen_info["starting_price"]]
        for _ in range(1, price_gen_info["n_days"]):
            y_data.append(y_data[-1] + self.np_random.normal(price_gen_info["mean"], price_gen_info["std"]))
        return y_data
    
    def _observe_price_data(self, time: int):
        """Observe price data at a given time over the window size."""
        return np.array(self.price_data[time-self._window:time])
    
    def update_state(self):
        # log_returns = calculate_log_returns(self.prices)
        simple_returns = calculate_simple_returns(self.prices)
        self.state = np.concat([simple_returns, self.account.position, self.account.margin])
    
    def render(self):
        pass
    
    def close(self):
        pass
    

if __name__=="__main__":
    env = SingleInstrumentEnv()
    env.step(env.action_space.sample())