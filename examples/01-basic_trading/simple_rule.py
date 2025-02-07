import hydra
import wandb
from omegaconf.omegaconf import OmegaConf
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

import deeptrade.env  # import to register envs
from deeptrade.util.finance import calculate_log_returns
from deeptrade.diagnostics.plotting import Plotter

from typing import Dict, List, Optional
# from plotting import Plotter

class RuleWorkspace:

    def __init__(self, cfg):

        # Setup directory configs
        self.work_dir = Path.cwd()
        self.cfg = cfg

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
        self.price_data = generator.generate(dt = cfg.dt, n_steps=cfg.n_steps)[0, :]

        # Create environment
        env_dict = {
            "starting_cash": cfg.starting_cash,
            "window": cfg.window,
            "start_time": cfg.start_time,
            "prices_data": self.price_data,
            "seed": cfg.seed,
            "dt": cfg.dt
        }
        self.env = gym.make("SingleInstrument-v0", **env_dict)

        # Create agent
        self.agent = hydra.utils.instantiate(cfg.agent)

    def train(self):
        """Train the agent"""

        from deeptrade.env.agents import BreakoutAgent, EWMACAgent
        if type(self.agent) == BreakoutAgent:
            dimensions = [
                        Integer(5, 100, name='lookback_period')
                    ]
        elif type(self.agent) == EWMACAgent:
            dimensions = [
                        Integer(20, 100, name='slow_period'),
                        Integer(5, 50, name='fast_period')
                    ]
        else:
            raise ValueError(f"Agent type: {type(self.agent)}, not recognized")

        @use_named_args(dimensions)
        def objective(**params):

            for param_name, param_value in params.items():
                setattr(self.agent, param_name, param_value)

            # Run episode
            terminated, truncated = False, False
            obs, info = self.env.reset(options = {"end_time": self.cfg.start_backtest})
            account_values = [self.cfg.starting_cash]
            position = np.array([0.0])

            while not terminated and not truncated:
                action = self.agent.act(obs, position)
                obs, reward, terminated, truncated, info = self.env.step(action)
                account_values.append(info['margin'])
                position = action

            # Calculate total return (negative since we're minimizing)
            total_return = -(account_values[-1] - account_values[0]) / account_values[0]

            if self.cfg.use_wandb:
                wandb.log({
                    'train_return': -total_return,
                    **params
                })

            return total_return

        # Bayesian optimization
        n_calls = 50
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            noise=1e-10,
            random_state=self.cfg.seed
        )

        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        for param_name, param_value in best_params.items():
            setattr(self.agent, param_name, param_value)

        if self.cfg.use_wandb:
                wandb.log({
                    'best_params': best_params,
                    'best_return': -result.fun
                })

        print(f"Best parameters found: {best_params}")
        print(f"Best return achieved: {-result.fun:.2%}")

        return

    def eval(self):
        """Evaluate the agent"""

        prices = []
        positions = []
        account_values = []

        terminated, truncated = False, False
        backtest_config = {"start_time": self.cfg.start_backtest, "end_time": self.cfg.n_steps}
        obs, info = self.env.reset(options=backtest_config)
        positions = []
        position = np.array([0.0])

        while not terminated and not truncated:
            action = self.agent.act(obs, position)
            obs, reward, terminated, truncated, info = self.env.step(action)
            position = action

            prices.append(info['current_price'])
            positions.append(info['position'])
            account_values.append(info['margin'])

        prices = np.array(prices)
        positions = np.array(positions)
        account_values = np.array(account_values)

        strategy_returns = np.diff(account_values) / account_values[:-1]
        benchmark_returns = self.agent.pos_size * np.diff(prices) / account_values[:-1]  # buy and hold

        self.plotter.plot_performance_overview(
            price_data=prices,
            positions=positions,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns
        )

        self.plotter.plot_trade_analysis(
            positions=positions,
            returns=strategy_returns,
            price_data=prices
        )

        self.plotter.plot_risk_metrics(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns
        )

        if self.cfg.use_wandb:
            metrics = {
                'final_value': account_values[-1],
                'total_return': (account_values[-1] - self.cfg.starting_cash) / self.cfg.starting_cash,
                'sharpe_ratio': np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns),
                'max_leverage': np.max(np.abs(positions)),
                'avg_position': np.mean(np.abs(positions)),
                'num_trades': np.sum(np.diff(positions) != 0),
                'avg_trade_size': np.mean(np.abs(np.diff(positions)[np.diff(positions) != 0])) if any(np.diff(positions) != 0) else 0
            }
            wandb.log(metrics)

        return

    def run(self):
        """Main execution flow"""
        if self.cfg.train:
            self.train()
        self.eval()

@hydra.main(config_path="../configs", config_name="01-simple_rule.yaml")
def main(cfg):
    workspace = RuleWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
