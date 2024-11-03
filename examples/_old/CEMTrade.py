import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any, Union, Optional
import gymnasium as gym
from datetime import datetime
import wandb

from deeptrade.env import reward_fns, termination_fns
from deeptrade.models import ModelEnv, GaussianMLP, BasicEnsemble
from deeptrade.planning import RandomAgent, TrajectoryOptimizerAgent
import deeptrade.planning as planning
import deeptrade.models as models
import deeptrade.util.common as common_util
from deeptrade.util import Logger
from deeptrade.models.time_series_processes import Sine
from deeptrade.util.replay_buffer import ReplayBuffer


class CEMTradeWorkspace:

    def __init__(self, cfg):
        # Setup directory configs
        self.work_dir = Path.cwd()
        self.cfg = cfg

        # Seeding
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # Device setup
        self.device = 'cuda:0' if torch.cuda.is_available() and cfg.device == "cuda" else 'cpu'
        self.generator = torch.Generator(device=self.device)
        if cfg.seed is not None:
            self.generator.manual_seed(cfg.seed)

        # Setup wandb if enabled
        if cfg.use_wandb:
            run_name = f"model_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="deep_trade_planning",
                      name=run_name,
                      config=OmegaConf.to_container(cfg))  # type: ignore

        # Create environment
        self.env = self._create_environment()
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        # Setup dynamics model and training components
        self.dynamics_model = self._create_dynamics_model()
        self.replay_buffer = self._create_replay_buffer()
        self.model_trainer = self._create_model_trainer()

    def _create_environment(self):
        """Create and return the trading environment."""
        env_dict = {
            'starting_cash': self.cfg.env.starting_cash,
            'window': self.cfg.env.window,
            'seed': self.cfg.seed,
            'dt': 1/252,
        }

        env_dict["price_gen_info"] = {
            "name": "GBM",
            "S0": np.array([100.0]),
            "mu": np.array([0.1]),
            "cov_matrix": np.array([0.2]),
            "n_steps": 500,
        }

        # Use sine wave for deterministic testing
        # prices_data = Sine(
        #     amp=np.array([1.0]),
        #     freq=np.array([1e-3])
        # ).generate(dt=1.0, n_steps=1000)[0]
        # env_dict["prices_data"] = prices_data

        return gym.make("SingleInstrument-v0", **env_dict)

    def _create_dynamics_model(self):
        """Create the dynamics model."""
        in_size = self.obs_shape[0]
        out_size = (1,)
        reward_fn = reward_fns.single_instrument_reward

        member_cfg = OmegaConf.create({
            "_target_": "deeptrade.models.GaussianMLP",
            "device": self.device,
            "in_size": in_size,
            "out_size": out_size,
            "num_layers": self.cfg.dynamics_model.num_layers,
            "hid_size": self.cfg.dynamics_model.hid_size,
            "activation_fn_cfg": {"_target_": "torch.nn.SiLU"}
        })

        ensemble = BasicEnsemble(self.cfg.dynamics_model.ensemble_size, self.device, member_cfg)

        if self.cfg.use_wandb:
            wandb.watch(ensemble, log="all", log_freq=100)

        return models.OneDTransitionRewardModel(
            ensemble,
            target_is_delta=self.cfg.algorithm.target_is_delta,
            normalize=self.cfg.algorithm.normalize,
            reward_fn=reward_fn,
        )

    def _create_model_trainer(self):
        """Create the model trainer."""
        return models.ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay
        )

    def _create_replay_buffer(self):
            """Create and return the replay buffer using the existing utility function."""
            return ReplayBuffer(10000, (self.env.unwrapped.observation_space.shape), (0,))

    def collect_data(self):
            """Collect random exploration data using existing utility function."""

            for _ in range(100):
                terminated, truncated = False, False
                obs, info = self.env.reset(seed=self.cfg.seed)
                idt = 0
                while (not terminated) and (not truncated):
                    idt += 1
                    action = np.array([0.0])  # placeholder action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    # print(next_obs.shape, obs.shape)
                    self.replay_buffer.add(obs, 0, next_obs, 0, terminated, truncated)
                    obs = next_obs

            # print("\nCollecting random exploration data...")

            # random_agent = RandomAgent(self.env)

            # common_util.rollout_agent_trajectories(
            #     self.env,
            #     steps_or_trials_to_collect=self.cfg.training.exploration_steps,
            #     agent=random_agent,
            #     agent_kwargs={},
            #     replay_buffer=self.replay_buffer,
            #     trial_length=self.cfg.trial_length
            # )

            print(f"\nData collection complete:")
            print(f"Buffer size: {len(self.replay_buffer)}")

    def train_model(self):
        """Train the dynamics model."""
        print("\nTraining dynamics model...")

        # Update normalizer with collected data
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())

        # Create training and validation datasets
        dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
            self.replay_buffer,
            batch_size=self.cfg.training.batch_size,
            val_ratio=0,
            ensemble_size=self.cfg.dynamics_model.ensemble_size,
            shuffle_each_epoch=True,
            bootstrap_permutes=False  # Don't use bootstrapping, train on the actual price data!
        )

        train_losses, val_losses = self.model_trainer.train(
            dataset_train,
            None,
            num_epochs=self.cfg.training.epochs
        )

        return train_losses, val_losses

    @staticmethod
    def get_action(predicted_states, predicted_stds, prices):
        """Improved trading strategy."""
        window_short = 5
        window_long = 10

        if len(predicted_states) < window_long:
            return np.array([0.0])

        # Calculate moving averages
        ma_short = np.mean(prices[-window_short:])
        ma_long = np.mean(prices[-window_long:])

        # Calculate momentum
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Calculate volatility
        vol = np.std(prices[-window_long:]) if len(prices) >= window_long else np.inf

        # Calculate prediction confidence
        recent_prediction_error = np.mean(np.abs(np.array(predicted_states[-5:]) - np.array(prices[-5:])))
        confidence = 1.0 / (1.0 + recent_prediction_error)

        # Position sizing based on multiple factors
        base_position = 5.0
        position_size = base_position * confidence * (1.0 / (1.0 + vol))

        # Trading signals
        trend_signal = ma_short > ma_long
        momentum_signal = momentum > 0
        uncertainty_signal = np.mean(predicted_stds[-5:]) < 0.1

        # Combined trading decision
        if trend_signal and momentum_signal and uncertainty_signal:
            return np.array([position_size])  # Long
        elif not trend_signal and not momentum_signal and uncertainty_signal:
            return np.array([-position_size])  # Short
        else:
            # Reduce position size or go neutral when signals conflict
            return np.array([0.0])

    def trade(self):
        """Trade using the learned model."""
        state, info = self.env.reset()
        terminated, truncated = False, False

        # Lists to store data for plotting
        predicted_states = []
        predicted_stds = []
        actual_states = []
        positions = []
        margins = []
        prices = []
        timestamps = []
        t = 0

        while not terminated and not truncated:
            # Get current state
            x_in = torch.tensor(state).unsqueeze(0).float().to(self.device)
            x_in = self.dynamics_model.input_normalizer.normalize(x_in)

            with torch.no_grad():
                predicted_state, predicted_logvar = self.dynamics_model(x_in)
                pred_mean = predicted_state[:, :, -1].mean().cpu().item()
                pred_std = torch.exp(0.5 * predicted_logvar[:, :, -1]).mean().cpu().item()

            # Store prediction before action
            predicted_states.append(pred_mean)
            predicted_stds.append(pred_std)
            prices.append(state[-1].item())

            # Take action based on raw gradient
            # if len(predicted_states) > 10 and predicted_states[-1] - predicted_states[-9] > 0:
            #     action = np.array([9.0])
            # else:
            #     action = np.array([-9.0])

            # print(predicted_states[-1] - predicted_states[-2], action)
            # Step environment
            action = self.get_action(predicted_states, predicted_stds, state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Store actual values after action
            actual_states.append(next_state[-1].item())
            positions.append(info.get('position', 0))
            margins.append(info.get('margin', 0))
            timestamps.append(t)

            state = next_state
            t += 1

        # Convert lists to numpy arrays for easier manipulation
        predicted_states = np.array(predicted_states)
        predicted_stds = np.array(predicted_stds)
        actual_states = np.array(actual_states)
        timestamps = np.array(timestamps)

        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

        # Plot 1: Predicted vs Actual States with uncertainty
        ax1.plot(timestamps, predicted_states, label='Predicted', linestyle='--', color='blue')
        ax1.plot(timestamps, actual_states, label='Actual', color='red')

        # Add uncertainty envelope (±2 standard deviations = 95% confidence interval)
        ax1.fill_between(timestamps,
                        predicted_states - 2*predicted_stds,
                        predicted_states + 2*predicted_stds,
                        color='blue', alpha=0.2, label='95% Confidence')

        ax1.set_title('Predicted vs Actual States (with uncertainty)')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Position vs Price
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(timestamps, prices, 'b-', label='Price')[0]
        line2 = ax2_twin.plot(timestamps, positions, 'r-', label='Position')[0]
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price', color='b')
        ax2_twin.set_ylabel('Position', color='r')
        ax2.set_title('Position vs Price')

        # Add legend for both lines
        lines = [line1, line2]
        ax2.legend(lines, [line.get_label() for line in lines])
        ax2.grid(True)

        # Plot 3: Margin Through Time
        ax3.plot(timestamps, margins)
        ax3.set_title('Margin Through Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Margin')
        ax3.grid(True)

        # Add some statistics as text
        stats_text = (f'Mean Prediction Error: {np.mean(np.abs(predicted_states - actual_states)):.4f}\n'
                        f'Mean Uncertainty: {np.mean(predicted_stds):.4f}')
        fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')

        # Adjust layout and save
        fig.tight_layout()
        fig.savefig('trading_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Trade Performance: {info}")
        print(f"Final Margin: {margins[-1]}")
        print(f"\nPrediction Statistics:")
        print(f"Mean Absolute Prediction Error: {np.mean(np.abs(predicted_states - actual_states)):.4f}")
        print(f"Mean Model Uncertainty: {np.mean(predicted_stds):.4f}")

    def run(self):
        """Main execution flow."""
        # Collect exploration data
        self.collect_data()

        # Train the model
        train_losses, val_losses = self.train_model()
        plt.plot(train_losses, label="train")
        plt.savefig("train_losses.png")

        # Optimize to get P&L
        self.trade()

@hydra.main(config_path="configs/", config_name="planning")
def main(cfg):
    workspace = CEMTradeWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
