import hydra
import wandb
from omegaconf.omegaconf import OmegaConf
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import deeptrade.env
from deeptrade.util.finance import calculate_log_returns
from deeptrade.diagnostics.plotting import Plotter
from typing import Dict, List, Optional

from deeptrade.optimization import PortfolioOptimizer, PortfolioWeights


class OptPortfolioWorkspace:

    def __init__(self, cfg):

        # Setup directory configs
        self.work_dir = Path.cwd()
        self.cfg = cfg
        self.n_instruments = cfg.n_instruments  # store for easy access

        # Setup logging/plotting
        self.plotter = Plotter(self.cfg, self.work_dir / "plots")

        # Setup wandb if enabled
        if cfg.use_wandb:
            run_name = f"simple_rule_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            wandb.init(project='deep_trade',
                       name=run_name,
                       config=OmegaConf.to_container(cfg))

        # TODO: Move this to a more appropriate place
        def as_numpy_array(value):
            return np.array(value)
        OmegaConf.register_new_resolver("as_numpy_array", as_numpy_array)

        # Generate price data
        generator = hydra.utils.instantiate(cfg.price_model)
        self.price_data = generator.generate(dt = cfg.dt, n_steps=cfg.n_steps)

        # Create Environment
        env_dict = {
            "starting_cash": cfg.starting_cash,
            "window": cfg.window,
            "start_time": cfg.start_time,
            "prices_data": self.price_data,
            "seed": cfg.seed,
            "dt": cfg.dt
        }
        self.env = gym.make("MultiInstrument-v0", **env_dict)

        # Create agents
        self.agents = []
        for ida in range(self.n_instruments):
            agent = hydra.utils.instantiate(cfg.agent)
            self.agents.append(agent)
        self.agents = (self.agents)  # keep order to avoid confusion, need n_agents to preserve copies.

        # Create portfolio weights
        self.portfolio_weights = None

    def train(self):
        """Train agent and calculate optimal weights"""

        if self.cfg.train_agents:
            # Optimize the agents weights based on backtest training data
            pass

        if self.cfg.optimize_portfolio:
            # Optimize the portfolio weights based on backtest training data

            one_hot_returns = []

            for idi in range(self.n_instruments):
                agent = self.agents[idi]

                obs, info = self.env.reset(options = {"end_time": self.cfg.start_backtest})
                terminated, truncated = False, False
                returns = []

                while not terminated and not truncated:
                    actions = np.zeros(self.n_instruments)
                    position = info['position'][idi]
                    actions[idi] = agent.act(obs[idi], np.array([position], dtype=float))
                    obs, rewards, terminated, truncated, info = self.env.step(np.array(actions))
                    returns.append(rewards)

                one_hot_returns.append(np.array(returns))
            one_hot_returns = np.array(one_hot_returns)
            # Create Portfolio Optimizer
            optimizer = PortfolioOptimizer(
                risk_free_rate=self.cfg.risk_free_rate,
                target_vol=self.cfg.target_vol
            )

            # Store weights
            self.portfolio_weights = optimizer.optimize_portfolio(one_hot_returns)
            pass

        return

    def eval(self):

        return

    def run(self):

        self.train()
        self.eval()


@hydra.main(config_path="../configs", config_name="02-portfolio_optimization.yaml", version_base="1.2")
def main(cfg):
    workspace = OptPortfolioWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
